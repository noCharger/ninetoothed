#!/usr/bin/env python3
"""
rubric_scorer.py — deterministic six-sub-score rubric scorer for one episode.

Mirrors tests/verifier_spec.md (rules §4.4): 0–10 per task across six sub-scores.
The SAME scorer runs for no_skill / v0 / a / b so deltas are meaningful.

It scores an *episode workspace*: the directory an agent produced when solving one
proxy task. It does not run the agent — run_episode.py does that and then calls here.

Sub-scores (max):
  completion (4) : correctness. operator → run_correctness_matrix on the test file;
                   full pass = 4, partial = 1–3 by fraction of matrix cases passed,
                   none = 0. diagnosis → fraction of expected_findings hit, scaled to 4.
  test       (2) : matrix ran and loop closed. all cases pass + a results record
                   present = 2; partial = 1; not run / errored = 0.
  perf       (1) : a benchmark.csv (>=3 sizes) with a bound verdict, OR
                   inspect_generated_source evidence, is present.
  minimality (1) : git diff (or changed-file set) touches only allowed files, no mass
                   reformat / unrelated refactor.
  style      (1) : produced python is ruff/black clean (best-effort; skipped-clean if
                   tools absent) and matches kernel layout conventions.
  compliance (1) : no eval(/exec(/network in produced scripts; no test scaffold deleted;
                   reward_hacking_guard static pass (if a generated source is present).

Everything is best-effort and degrades to a conservative score with an explicit
reason string, so the scorer never crashes an evolution loop. All reasons are
recorded so a human can audit any score (verifier_spec "auto-checkable gates").

Import-safe without torch: the correctness gate shells out to
scripts/run_correctness_matrix.py in a subprocess, so torch lives only in that
child on the GPU host.
"""
from __future__ import annotations

import ast
import json
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

# rubric maxima, shared with the journal
RUBRIC_MAX = {"completion": 4, "test": 2, "perf": 1, "minimality": 1, "style": 1, "compliance": 1}

# files an episode is allowed to create/modify (minimality gate). Anything else
# touched counts against minimality.
_ALLOWED_EDIT_BASENAMES = {
    "kernel.py", "kernel_fixed.py", "wrapper.py", "test_correctness.py",
    "bench.csv", "benchmark.csv", "bench_result.txt", "notes.md",
    "matrix.csv", "generated_source.txt", "diagnosis.md",
    # legitimate process artifacts a real agent may emit while following the workflow
    "trace.md", "bench.py", "conftest.py",
}

# the subset of _ALLOWED_EDIT_BASENAMES that is actual *implementation* (as opposed
# to a test/bench/notes artifact). The legality gate (_banned_fallback_hits) only
# scans these: test_correctness.py legitimately computes a torch reference value to
# assert against, and flagging that would be a false positive, not a real cheat.
_SOLUTION_BASENAMES = {"kernel.py", "kernel_fixed.py", "wrapper.py"}

_NETWORK_TOKENS = ("socket", "urllib", "requests", "http.client", "httpx", "aiohttp")


@dataclass
class SubScore:
    value: int
    max: int
    reason: str


@dataclass
class RubricResult:
    task_id: str
    mode: str
    subscores: dict = field(default_factory=dict)   # name -> SubScore
    matrix_pass_frac: float = 0.0
    total: int = 0
    completion_error: str = ""    # real failure detail (traceback/assertion) for attribution

    def as_scores(self) -> dict:
        return {k: v.value for k, v in self.subscores.items()}

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "mode": self.mode,
            "total": self.total,
            "matrix_pass_frac": round(self.matrix_pass_frac, 4),
            "subscores": {k: {"value": v.value, "max": v.max, "reason": v.reason}
                          for k, v in self.subscores.items()},
        }


# --------------------------------------------------------------------------- helpers
def _find(workspace: pathlib.Path, *names: str) -> Optional[pathlib.Path]:
    for name in names:
        hits = sorted(workspace.rglob(name))
        if hits:
            return hits[0]
    return None


def _python_files(workspace: pathlib.Path) -> list[pathlib.Path]:
    return [p for p in workspace.rglob("*.py") if "__pycache__" not in p.parts]


def _run_correctness_matrix(test_file: pathlib.Path, matrix_csv: pathlib.Path,
                            skill_root: pathlib.Path) -> tuple[float, str]:
    """
    Shell out to scripts/run_correctness_matrix.py. Returns (pass_fraction, reason).
    pass_fraction in [0,1]; a torch/CUDA-absent SKIP is treated as "unknown" (0.0)
    with an explicit reason so offline runs don't fake a pass.
    """
    script = skill_root / "scripts" / "run_correctness_matrix.py"
    if not script.exists():
        return 0.0, f"run_correctness_matrix.py not found at {script}"
    if not test_file.exists():
        return 0.0, "no test file produced"
    # NOTE: run_correctness_matrix uses argparse.REMAINDER, so --csv MUST precede the
    # positional test file, else it is swallowed as a pytest arg.
    cmd = [sys.executable, str(script), "--csv", str(matrix_csv), str(test_file)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                              cwd=str(test_file.parent))
    except subprocess.TimeoutExpired:
        return 0.0, "correctness matrix timed out (600s)"
    out = proc.stdout + "\n" + proc.stderr
    # Prefer parsing the CSV the script writes; fall back to stdout counts.
    frac, reason = _parse_matrix_csv(matrix_csv)
    if frac is None:
        frac, reason = _parse_matrix_stdout(out, proc.returncode)
    detail = _first_failure_detail(out) if frac < 0.999 else ""
    return frac, reason, detail


def _first_failure_detail(out: str) -> str:
    """Pull the most informative failure line (Error/assert) from pytest output, for
    the evolution loop's failure attribution."""
    lines = out.splitlines()
    # prefer an explicit exception/assertion line
    for ln in lines:
        s = ln.strip()
        if s.startswith("E   ") and any(k in s for k in
                ("Error", "assert", "Exception", "MERE", "not defined", "argument",
                 "mismatch", "no attribute", "nan")):
            return s[4:].strip()[:300]
    for ln in lines:
        if "Error" in ln or "error" in ln:
            return ln.strip()[:300]
    return ""


def _parse_matrix_csv(matrix_csv: pathlib.Path) -> tuple[Optional[float], str]:
    if not matrix_csv.exists():
        return None, "no matrix csv"
    import csv
    rows = list(csv.DictReader(matrix_csv.open()))
    if not rows:
        return None, "empty matrix csv"
    # a row is a pass if a status/result column says pass, or mere/mare within bound
    passed = 0
    total = 0
    for r in rows:
        status = (r.get("status") or r.get("result") or "").strip().lower()
        if not status:
            continue
        total += 1
        if status in ("pass", "passed", "ok", "true"):
            passed += 1
    if total == 0:
        return None, "matrix csv had no status column"
    return passed / total, f"matrix {passed}/{total} cases passed"


def _parse_matrix_stdout(out: str, returncode: int) -> tuple[float, str]:
    # pytest-style "N passed, M failed"
    m_pass = re.search(r"(\d+) passed", out)
    m_fail = re.search(r"(\d+) (?:failed|error)", out)
    m_skip = re.search(r"(\d+) skipped", out)
    npass = int(m_pass.group(1)) if m_pass else 0
    nfail = int(m_fail.group(1)) if m_fail else 0
    nskip = int(m_skip.group(1)) if m_skip else 0
    denom = npass + nfail
    if denom == 0:
        if nskip > 0:
            return 0.0, f"all {nskip} cases skipped (CUDA absent?) — correctness unknown"
        return 0.0, f"no pass/fail parsed (returncode={returncode})"
    return npass / denom, f"stdout {npass} passed / {nfail} failed"


# --------------------------------------------------------------------------- sub-scorers
def score_completion(workspace: pathlib.Path, task_meta: dict, skill_root: pathlib.Path,
                     out_dir: pathlib.Path) -> tuple[SubScore, float]:
    kind = task_meta.get("kind", "operator")
    if kind == "diagnosis":
        return _score_completion_diagnosis(workspace, task_meta)
    # Prefer the harness-supplied oracle test (tamper-proof) over the agent's own test.
    test_file = _find(workspace, "oracle_test.py") or \
        _find(workspace, "test_correctness.py", f"test_{task_meta.get('name','')}.py")
    if test_file is None:
        return SubScore(0, 4, "no oracle/test file present"), 0.0, "no oracle/test file produced"
    matrix_csv = out_dir / f"matrix_{task_meta['id']}_{workspace.name}.csv"
    frac, reason, detail = _run_correctness_matrix(test_file, matrix_csv, skill_root)
    # map fraction -> 0..4 : full=4, partial 1..3, none=0
    if frac >= 0.999:
        val = 4
    elif frac <= 0.0:
        val = 0
    else:
        val = max(1, min(3, round(frac * 4)))
    # The task is to implement a NineToothed operator. A correct-output solution that
    # does NOT actually use the DSL (pure torch, or a hallucinated fake API), or that
    # reaches a banned torch/aten matmul-conv-reduction fallback (MusaCoder/KernelBench
    # legality convention), has not fulfilled the task even though the numbers match —
    # cap its completion so the skill's real-API value is what counts.
    if val >= 2:
        legality_reason = _legality_violation(workspace, skill_root)
        if legality_reason:
            val = 1
            reason += f" (capped: {legality_reason})"
            if not detail:
                detail = legality_reason
    return SubScore(val, 4, reason), frac, detail


def _uses_ninetoothed(workspace: pathlib.Path) -> bool:
    """True if the produced code genuinely builds a NineToothed kernel (import + make)."""
    for p in _python_files(workspace):
        if p.name == "oracle_test.py":
            continue
        src = p.read_text(encoding="utf-8", errors="ignore")
        if "ninetoothed" in src and re.search(r"\bmake\s*\(", src):
            return True
    return False


def _legality_violation(workspace: pathlib.Path, skill_root: pathlib.Path) -> str:
    """MusaCoder/KernelBench legality gate: a solution that never touches NineToothed,
    or that reaches for a banned torch/aten fallback (matmul/conv/reduction/attention),
    has not completed the task even if its output happens to be numerically correct.
    Returns a reason string, or "" if legal."""
    if not _uses_ninetoothed(workspace):
        return "solution does not use NineToothed (no import+make)"
    banned = _banned_fallback_hits(workspace, skill_root)
    if banned:
        return f"banned aten/torch fallback used — {banned[0]}"
    return ""


def _banned_fallback_hits(workspace: pathlib.Path, skill_root: pathlib.Path) -> list[str]:
    guard = _load_reward_hacking_guard(skill_root)
    if guard is None:
        return []
    hits: list[str] = []
    for p in _python_files(workspace):
        if p.name not in _SOLUTION_BASENAMES:
            continue
        try:
            hits.extend(guard.banned_fallback_analysis(source_code=p.read_text(
                encoding="utf-8", errors="ignore")))
        except Exception:  # noqa: BLE001  (guard is best-effort)
            continue
    return hits


def _load_reward_hacking_guard(skill_root: pathlib.Path):
    try:
        sys.path.insert(0, str(skill_root / "evaluation"))
        from skill_eval import reward_hacking_guard  # type: ignore
        return reward_hacking_guard
    except Exception:  # noqa: BLE001  (guard is best-effort)
        return None


def _score_completion_diagnosis(workspace: pathlib.Path, task_meta: dict) -> tuple[SubScore, float, str]:
    findings = [f.lower() for f in task_meta.get("expected_findings", [])]
    if not findings:
        return SubScore(0, 4, "task has no expected_findings"), 0.0, ""
    diag = _find(workspace, "diagnosis.md", "notes.md")
    text = diag.read_text(encoding="utf-8", errors="ignore").lower() if diag else ""
    hit = sum(1 for f in findings if _finding_hit(f, text))
    frac = hit / len(findings)
    val = round(frac * 4)
    detail = "" if frac >= 0.999 else f"missed findings: {[f for f in findings if not _finding_hit(f, text)][:3]}"
    return SubScore(val, 4, f"diagnosis hit {hit}/{len(findings)} expected findings"), frac, detail


_CJK = r"一-鿿"


def _finding_hit(finding: str, text: str) -> bool:
    """Robust to mixed Chinese/ASCII findings. A finding is 'covered' when enough of its
    salient tokens appear in the diagnosis: discriminative ASCII/technical tokens
    (fp16, exp, num_warps, gb/s, contiguous, .ninetoothed) plus Chinese 2-grams. ASCII
    tech tokens are weighted double since they are the strongest signal."""
    t = text.lower()
    f = finding.lower()
    ascii_tokens = [w for w in re.findall(r"[a-z0-9_./%]{2,}", f) if w not in _STOP]
    cjk_bigrams = []
    for seg in re.findall(rf"[{_CJK}]+", finding):
        cjk_bigrams += [seg[i:i + 2] for i in range(len(seg) - 1)]
    if not ascii_tokens and not cjk_bigrams:
        return f in t
    score = sum(2 for k in ascii_tokens if k in t) + sum(1 for k in set(cjk_bigrams) if k in text)
    need = max(2, int(0.4 * (2 * len(ascii_tokens) + len(set(cjk_bigrams)))))
    return score >= need


_STOP = {"the", "and", "for", "with", "use", "you", "are", "can"}


def score_test(completion_frac: float, workspace: pathlib.Path, out_dir: pathlib.Path,
               task_id: str) -> SubScore:
    matrix = _find(out_dir, f"matrix_{task_id}_*.csv") or _find(workspace, "matrix.csv")
    ran = matrix is not None
    if completion_frac >= 0.999 and ran:
        return SubScore(2, 2, "all matrix cases pass and results recorded")
    if ran and completion_frac > 0:
        return SubScore(1, 2, "matrix ran, partial pass")
    if ran:
        return SubScore(1, 2, "matrix ran but no cases passed")
    return SubScore(0, 2, "correctness matrix not run / no results recorded")


def score_perf(workspace: pathlib.Path, task_meta: dict) -> SubScore:
    bench = _find(workspace, "bench.csv", "benchmark.csv")
    if bench is not None:
        try:
            import csv
            rows = list(csv.reader(bench.open()))
            data_rows = [r for r in rows if r and not r[0].lower().startswith(("size", "shape", "#"))]
            n_sizes = len(data_rows)
        except Exception:  # noqa: BLE001
            n_sizes = 0
        txt = bench.read_text(encoding="utf-8", errors="ignore").lower()
        has_verdict = any(v in txt for v in ("memory-bound", "compute-bound", "memory bound", "compute bound"))
        # verdict may live in a sibling notes file
        if not has_verdict:
            notes = _find(workspace, "notes.md", "bench_result.txt")
            if notes:
                has_verdict = any(v in notes.read_text(errors="ignore").lower()
                                  for v in ("memory-bound", "compute-bound"))
        if n_sizes >= 3 and has_verdict:
            return SubScore(1, 1, f"benchmark with {n_sizes} sizes + bound verdict")
        if n_sizes >= 3:
            return SubScore(1, 1, f"benchmark with {n_sizes} sizes (verdict not detected)")
    gensrc = _find(workspace, "generated_source.txt")
    if gensrc is not None and gensrc.stat().st_size > 0:
        return SubScore(1, 1, "generated-source inspection evidence present")
    # perf work is only *required* for bench-carrying tasks; absence => 0 but not fatal
    return SubScore(0, 1, "no benchmark.csv or generated-source evidence")


def score_minimality(changed_files: list[str]) -> SubScore:
    if not changed_files:
        return SubScore(1, 1, "no change set provided — assume minimal")
    offenders = [f for f in changed_files
                 if pathlib.Path(f).name not in _ALLOWED_EDIT_BASENAMES]
    if offenders:
        return SubScore(0, 1, f"touched non-allowed files: {offenders[:4]}")
    return SubScore(1, 1, f"only allowed files touched ({len(changed_files)})")


def score_style(workspace: pathlib.Path) -> SubScore:
    pys = _python_files(workspace)
    if not pys:
        return SubScore(0, 1, "no python produced")
    # all produced python must at least parse
    for p in pys:
        try:
            ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError as e:
            return SubScore(0, 1, f"syntax error in {p.name}: {e}")
    # best-effort ruff; if absent, syntax-clean is enough for the point
    ruff = _try_ruff([str(p) for p in pys])
    if ruff is None:
        return SubScore(1, 1, "syntax-clean (ruff not installed)")
    if ruff == 0:
        return SubScore(1, 1, "ruff clean")
    return SubScore(0, 1, f"ruff reported {ruff} issue(s)")


def _try_ruff(paths: list[str]) -> Optional[int]:
    try:
        proc = subprocess.run(["ruff", "check", "--quiet", *paths],
                              capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode == 0:
        return 0
    return len([l for l in proc.stdout.splitlines() if l.strip()]) or 1


def score_compliance(workspace: pathlib.Path, skill_root: pathlib.Path,
                     generated_source: Optional[pathlib.Path]) -> SubScore:
    reasons = []
    for p in _python_files(workspace):
        src = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"\beval\s*\(", src) or re.search(r"\bexec\s*\(", src):
            reasons.append(f"{p.name}: eval/exec")
        for tok in _NETWORK_TOKENS:
            if re.search(rf"\b(import|from)\s+{re.escape(tok)}", src):
                reasons.append(f"{p.name}: network import {tok}")
    # optional reward-hacking static analysis on the generated triton source
    if generated_source is not None and generated_source.exists():
        static_reasons = _try_static_guard(generated_source, skill_root)
        reasons.extend(static_reasons)
    # legality gate: banned torch/aten matmul-conv-reduction fallback anywhere in the
    # solution (MusaCoder/KernelBench convention — also caps completion, see
    # score_completion / _legality_violation; a hit here docks compliance too).
    reasons.extend(_banned_fallback_hits(workspace, skill_root))
    if reasons:
        return SubScore(0, 1, "; ".join(reasons[:4]))
    return SubScore(1, 1, "no eval/exec/network; static guard clean")


def _try_static_guard(generated_source: pathlib.Path, skill_root: pathlib.Path) -> list[str]:
    guard = _load_reward_hacking_guard(skill_root)
    if guard is None:
        return []
    try:
        flags = guard.static_analysis(source_path=generated_source)
        return [f"reward-hacking: {f}" for f in (flags or [])]
    except Exception:  # noqa: BLE001  (guard is best-effort)
        return []


# --------------------------------------------------------------------------- top-level
def score_episode(workspace: str | pathlib.Path, task_meta: dict, skill_root: str | pathlib.Path,
                  out_dir: str | pathlib.Path, changed_files: Optional[list[str]] = None) -> RubricResult:
    """
    Score one episode workspace against one proxy task.

    Args:
        workspace     : directory holding the agent's produced files for this task.
        task_meta     : a manifest task dict (id, kind, name, expected_findings, ...).
        skill_root    : ninetoothed-operator-dev root (for scripts/ + evaluation/).
        out_dir       : where to write matrix_<id>.csv artifacts.
        changed_files : optional list of file paths the episode modified (git diff).
    """
    workspace = pathlib.Path(workspace)
    skill_root = pathlib.Path(skill_root)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    completion, frac, completion_error = score_completion(workspace, task_meta, skill_root, out_dir)
    gensrc = _find(workspace, "generated_source.txt")
    subs = {
        "completion": completion,
        "test": score_test(frac, workspace, out_dir, task_meta["id"]),
        "perf": score_perf(workspace, task_meta),
        "minimality": score_minimality(changed_files or []),
        "style": score_style(workspace),
        "compliance": score_compliance(workspace, skill_root, gensrc),
    }
    res = RubricResult(task_id=task_meta["id"], mode=task_meta.get("_mode", "?"),
                       subscores=subs, matrix_pass_frac=frac, completion_error=completion_error)
    res.total = sum(min(s.value, s.max) for s in subs.values())
    return res


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Score one episode workspace.")
    p.add_argument("workspace")
    p.add_argument("--task-meta", required=True, help="JSON file or inline JSON of the task dict")
    p.add_argument("--skill-root", required=True)
    p.add_argument("--out-dir", default="./_rubric_out")
    p.add_argument("--changed", nargs="*", default=None)
    args = p.parse_args(argv)

    tm_arg = args.task_meta
    task_meta = json.loads(pathlib.Path(tm_arg).read_text()) if pathlib.Path(tm_arg).exists() \
        else json.loads(tm_arg)
    res = score_episode(args.workspace, task_meta, args.skill_root, args.out_dir, args.changed)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
