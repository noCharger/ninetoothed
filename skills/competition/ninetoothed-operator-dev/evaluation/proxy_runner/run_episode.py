#!/usr/bin/env python3
"""
run_episode.py — solve ONE proxy task under ONE mode with a `claude -p` agent,
collect artifacts, score with the rubric, compute the shaped reward, and return an
EpisodeRecord (the caller appends it to the journal).

This is the fixed agent harness the proposal calls for: same model, same time
budget, same repo — the ONLY thing that varies across modes is whether (and which)
skill text is available:

    mode="no_skill"  : task prompt only, no skill files in context.
    mode="v0"|"a"|"b": the skill package at --skill-root is copied into the
                        episode sandbox and the agent is told to read it first.

The agent runs in an isolated sandbox dir, so parallel episodes never collide and
the change-set (for the minimality sub-score) is exactly the files it created.

Offline-safe: pass --dry-run to build the sandbox and print the command without
calling claude; or inject a fake solver (solver=...) for unit tests. The real
solver shells out to `claude -p --output-format json` and parses usage/cost.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))            # evaluation/  (for skill_eval, proxy_tasks)
sys.path.insert(0, str(_HERE.parent / "journal"))

from rubric_scorer import score_episode                      # noqa: E402
from reward import EpisodeOutcome, RewardConfig, compute_reward, reward_from_rubric  # noqa: E402

try:
    from journal import EpisodeRecord                        # noqa: E402
except Exception:  # pragma: no cover
    EpisodeRecord = None  # type: ignore


# what the agent is asked to produce, by task kind
_OPERATOR_DELIVERABLES = (
    "kernel.py (the NineToothed kernel), wrapper.py, and test_correctness.py (pytest "
    "cases across shapes and the task's dtypes). wrapper.py MUST expose "
    "`def solve(*inputs)` returning the output tensor(s) — the grader calls solve(). "
    "If the task mentions performance, also emit bench.csv with >=3 input sizes and "
    "state a memory-bound / compute-bound verdict."
)
_DIAGNOSIS_DELIVERABLES = (
    "diagnosis.md containing your root-cause analysis and the concrete fix. "
    "List each finding explicitly."
)


@dataclass
class SolverResult:
    """What a solver returns after producing files in the sandbox."""
    tokens_in: float = 0.0
    tokens_out: float = 0.0
    cost_usd: float = 0.0
    wall_seconds: float = 0.0
    gpu_seconds: float = 0.0
    compiled: bool = True
    compile_attempts: int = 1
    raw: dict | None = None


Solver = Callable[[str, pathlib.Path, dict], SolverResult]


# --------------------------------------------------------------------------- prompt
def build_prompt(task_meta: dict, mode: str) -> str:
    kind = task_meta.get("kind", "operator")
    deliv = _OPERATOR_DELIVERABLES if kind == "operator" else _DIAGNOSIS_DELIVERABLES
    lines = []
    if mode != "no_skill":
        lines.append(
            "A NineToothed operator-development skill is available at ./skill/. "
            "Read ./skill/SKILL.md and the relevant file(s) under ./skill/references/ "
            "BEFORE writing any code, and follow its workflow and pitfalls."
        )
    lines.append(f"Task ({task_meta.get('category','?')}/{task_meta.get('difficulty','?')}): "
                 f"{task_meta.get('prompt','').strip()}")
    if kind == "diagnosis" and task_meta.get("scenario"):
        lines.append(f"Scenario: {task_meta['scenario']}")
    lines.append(f"Deliverables: produce {deliv}")
    lines.append("Write files into the current working directory. Do not access the network.")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- sandbox
def prepare_sandbox(out_dir: pathlib.Path, run_id: str, generation: int,
                    task_id: str, mode: str, skill_root: Optional[pathlib.Path]) -> pathlib.Path:
    ws = out_dir / "episodes" / run_id / f"gen{generation}" / f"{task_id}_{mode}"
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    if mode != "no_skill" and skill_root is not None:
        _seed_skill(ws / "skill", skill_root)
    return ws


def _seed_skill(dest: pathlib.Path, skill_root: pathlib.Path) -> None:
    """Copy the skill text the agent should read. Only docs + references + SKILL.md;
    scripts are copied too since the workflow invokes them."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("SKILL.md", "README.md", "REFERENCE.md"):
        src = skill_root / name
        if src.exists():
            shutil.copy2(src, dest / name)
    for sub in ("references", "scripts"):
        s = skill_root / sub
        if s.exists():
            shutil.copytree(s, dest / sub, dirs_exist_ok=True)


def changed_files(ws: pathlib.Path) -> list[str]:
    """Files the episode produced: everything under ws except the seeded skill/ and
    harness-generated bookkeeping (names starting with '_', e.g. _prompt.txt)."""
    out = []
    for p in ws.rglob("*"):
        rel = p.relative_to(ws)
        if not p.is_file():
            continue
        if "skill" in rel.parts or "__pycache__" in rel.parts:
            continue
        if any(part.startswith("_") for part in rel.parts):   # harness artifacts
            continue
        if rel.name in ("oracle_test.py",):                    # harness-supplied grader
            continue
        out.append(str(rel))
    return out


# --------------------------------------------------------------------------- solvers
def default_claude_solver(prompt: str, ws: pathlib.Path, opts: dict) -> SolverResult:
    """Invoke `claude -p --output-format json` inside the sandbox."""
    model = opts.get("model") or os.environ.get("NINETOOTHED_EPISODE_MODEL", "")
    timeout = int(opts.get("timeout", 900))
    cmd = ["claude", "-p", "--output-format", "json"]
    if model:
        cmd += ["--model", model]
    # allow the tools the workflow needs; keep network off via the sandbox + prompt.
    cmd += ["--permission-mode", opts.get("permission_mode", "acceptEdits")]
    cmd += [prompt]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(ws))
    except subprocess.TimeoutExpired:
        return SolverResult(wall_seconds=timeout, compiled=False, compile_attempts=0,
                            raw={"error": "timeout"})
    wall = time.time() - t0
    return _parse_claude_json(proc.stdout, wall)


def _parse_claude_json(stdout: str, wall: float) -> SolverResult:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        # streaming or plain text — still count wall time, tokens unknown
        return SolverResult(wall_seconds=wall, raw={"stdout_head": stdout[:200]})
    usage = data.get("usage", {}) or {}
    tin = usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
    tout = usage.get("output_tokens", 0)
    return SolverResult(
        tokens_in=tin, tokens_out=tout,
        cost_usd=data.get("total_cost_usd", 0.0),
        wall_seconds=data.get("duration_ms", wall * 1000) / 1000.0,
        raw={"num_turns": data.get("num_turns"), "subtype": data.get("subtype")},
    )


def resolve_solver(name: str, model_dir: Optional[str] = None,
                   fake_files: Optional[dict] = None) -> Optional[Solver]:
    """Map a --solver name to a solver callable.
    'claude' -> None (run_episode uses default_claude_solver);
    'qwen'   -> local Qwen coder (needs --model-dir, torch+transformers on host);
    'fake'   -> stub writer for offline wiring checks."""
    if name == "claude":
        return None
    if name == "qwen":
        if not model_dir:
            raise SystemExit("--solver qwen requires --model-dir")
        from qwen_solver import make_qwen_solver
        return make_qwen_solver(model_dir)
    if name == "fake":
        from run_matrix import _FAKE_FILES
        return make_fake_solver(fake_files or _FAKE_FILES)
    raise SystemExit(f"unknown solver {name!r}")


def make_fake_solver(files: dict, tokens=(3000, 1200), compiled=True) -> Solver:
    """Test/offline solver: writes given {relpath: content} into the sandbox."""
    def _solver(prompt: str, ws: pathlib.Path, opts: dict) -> SolverResult:
        for rel, content in files.items():
            fp = ws / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
        return SolverResult(tokens_in=tokens[0], tokens_out=tokens[1],
                            wall_seconds=1.0, compiled=compiled)
    return _solver


# --------------------------------------------------------------------------- episode
def run_episode(task_meta: dict, mode: str, skill_root: str | pathlib.Path,
                out_dir: str | pathlib.Path, run_id: str = "baseline", generation: int = 0,
                strategy: str = "", prompt_id: str = "", solver: Optional[Solver] = None,
                solver_opts: Optional[dict] = None, reward_cfg: Optional[RewardConfig] = None,
                dry_run: bool = False, prompt_override: Optional[str] = None):
    """
    Returns (EpisodeRecord | dict, RubricResult). The caller logs the record.
    prompt_override lets version A inject a bandit-chosen strategy/prompt framing;
    when None the default task+skill prompt is used.
    """
    skill_root = pathlib.Path(skill_root)
    out_dir = pathlib.Path(out_dir)
    ws = prepare_sandbox(out_dir, run_id, generation, task_meta["id"], mode, skill_root)
    prompt = prompt_override or build_prompt(task_meta, mode)
    (ws / "_prompt.txt").write_text(prompt, encoding="utf-8")

    if dry_run:
        print(f"[dry-run] sandbox={ws}")
        print(f"[dry-run] would run claude -p in {ws} with prompt:\n{prompt}")
        return None, None

    solver = solver or default_claude_solver
    sres = solver(prompt, ws, solver_opts or {})

    # For operator tasks, drop in the harness oracle test so completion is tamper-proof.
    if task_meta.get("kind", "operator") == "operator":
        try:
            from oracle import write_oracle_test
            write_oracle_test(task_meta, ws, skill_root / "evaluation" / "proxy_tasks")
        except Exception:  # noqa: BLE001  (oracle is best-effort; agent test is fallback)
            pass

    tmeta = dict(task_meta, _mode=mode)
    rubric = score_episode(ws, tmeta, skill_root, out_dir / "rubric", changed_files(ws))

    # shaped reward: prefer latency/roofline outcome if a bench result exists, else rubric-based
    correct = rubric.subscores["completion"].value >= 3
    tokens = sres.tokens_in + sres.tokens_out
    reward = reward_from_rubric(rubric.total, tokens, sres.gpu_seconds or sres.wall_seconds,
                                sres.compile_attempts, reward_cfg)
    if not sres.compiled:
        reward = compute_reward(EpisodeOutcome(compiled=False, correct=False), reward_cfg)

    rec = None
    if EpisodeRecord is not None:
        rec = EpisodeRecord(
            run_id=run_id, generation=generation, task_id=task_meta["id"], mode=mode,
            strategy=strategy, prompt_id=prompt_id,
            scores=rubric.as_scores(), reward=round(reward, 4),
            cost={"tokens_in": sres.tokens_in, "tokens_out": sres.tokens_out,
                  "cost_usd": sres.cost_usd, "gpu_seconds": sres.gpu_seconds,
                  "wall_seconds": round(sres.wall_seconds, 2),
                  "compile_attempts": sres.compile_attempts},
            classification="none" if correct else "failed",
            artifacts={"workspace": str(ws),
                       "changed": ",".join(changed_files(ws)[:12])},
            notes=json.dumps(rubric.to_dict()["subscores"], ensure_ascii=False),
        )
    return rec, rubric


# --------------------------------------------------------------------------- CLI
def _load_task(manifest: pathlib.Path, task_id: str) -> dict:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for t in data["tasks"]:
        if t["id"] == task_id:
            return t
    raise SystemExit(f"task {task_id} not in {manifest}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run one proxy-task episode.")
    p.add_argument("task_id")
    p.add_argument("--mode", default="v0", choices=["no_skill", "v0", "a", "b"])
    p.add_argument("--skill-root", required=True)
    p.add_argument("--manifest", default=None, help="proxy_tasks/manifest.json")
    p.add_argument("--out-dir", default="./evaluation/results/_adhoc")
    p.add_argument("--run-id", default="adhoc")
    p.add_argument("--generation", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--journal", default=None, help="append the record to this JSONL journal")
    args = p.parse_args(argv)

    skill_root = pathlib.Path(args.skill_root)
    manifest = pathlib.Path(args.manifest) if args.manifest else \
        skill_root / "evaluation" / "proxy_tasks" / "manifest.json"
    task_meta = _load_task(manifest, args.task_id)

    rec, rubric = run_episode(task_meta, args.mode, skill_root, args.out_dir,
                              run_id=args.run_id, generation=args.generation,
                              dry_run=args.dry_run)
    if args.dry_run:
        return 0
    print(json.dumps(rubric.to_dict(), ensure_ascii=False, indent=2))
    if args.journal and rec is not None:
        from journal import Journal
        Journal(args.journal).append_episode(rec)
        print(f"[journal] appended episode to {args.journal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
