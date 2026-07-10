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


# what the agent is asked to produce, by task kind (legacy flat-sandbox convention)
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

# repo-aware convention: contribute a real test file matching the project's own style
# (see tests/test_add.py, tests/test_pow.py — import ninetoothed, define the kernel,
# write a pytest test using tests.utils.get_available_devices).
def _operator_deliverables_repo(task_id: str) -> str:
    return (
        f"a new file `tests/test_contrib_{task_id}.py`, following the style of the "
        "EXISTING files in tests/ (e.g. tests/test_add.py, tests/test_pow.py — import "
        "ninetoothed, define the kernel with ninetoothed.make() or @ninetoothed.jit, and "
        "write a pytest test using tests.utils.get_available_devices, matching this "
        "project's own conventions). This file MUST ALSO expose a top-level function "
        "named exactly `solve` that takes the input tensors as positional arguments and "
        "returns the output tensor — this is the stable entrypoint an independent grader "
        "calls (in addition to whatever pytest test you write for yourself). "
        "Do not modify any other existing file in the repository."
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
def build_prompt(task_meta: dict, mode: str, repo_aware: bool = False) -> str:
    kind = task_meta.get("kind", "operator")
    lines = []
    if repo_aware:
        lines.append(
            "You are working inside a full clone of the `ninetoothed` repository (the "
            "actual project this operator belongs to) — not an isolated sandbox. Explore "
            "it: `tests/` has real examples of NineToothed operators (e.g. tests/test_add.py, "
            "tests/test_pow.py) and shared helpers (tests/utils.py, tests/conftest.py); "
            "`src/ninetoothed/` is the framework source; `CONTRIBUTING.md` has style "
            "conventions. Your work should fit this codebase's existing patterns."
        )
        if mode != "no_skill":
            lines.append(
                "A `ninetoothed-operator-dev` skill is also installed (auto-discovered "
                "under .claude/skills/, and present at skills/competition/"
                "ninetoothed-operator-dev/ in the repo). USE IT: read its SKILL.md + the "
                "relevant references, imitate its worked examples under examples/, and run "
                "its scripts (scripts/run_correctness_matrix.py, scripts/debug_arrangement.py, "
                "scripts/inspect_generated_source.py) — follow its workflow and pitfalls."
            )
        deliv = _operator_deliverables_repo(task_meta["id"]) if kind == "operator" \
            else _DIAGNOSIS_DELIVERABLES
    else:
        if mode != "no_skill":
            lines.append(
                "A `ninetoothed-operator-dev` skill is available (installed as a Claude Code "
                "skill, and mirrored under ./skill/). USE IT: read its SKILL.md + the relevant "
                "references, imitate its worked examples under examples/, and run its scripts "
                "(scripts/run_correctness_matrix.py, scripts/debug_arrangement.py, "
                "scripts/inspect_generated_source.py) to verify — follow its workflow and pitfalls."
            )
        deliv = _OPERATOR_DELIVERABLES if kind == "operator" else _DIAGNOSIS_DELIVERABLES

    lines.append(f"Task ({task_meta.get('category','?')}/{task_meta.get('difficulty','?')}): "
                 f"{task_meta.get('prompt','').strip()}")
    if kind == "diagnosis" and task_meta.get("scenario"):
        lines.append(f"Scenario: {task_meta['scenario']}")
    lines.append(f"Deliverables: produce {deliv}")
    lines.append("Write files into the current working directory. Do not access the network.")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- sandbox
_SKILL_RELPATH = ("skills", "competition", "ninetoothed-operator-dev")  # within repo_root


def prepare_sandbox(out_dir: pathlib.Path, run_id: str, generation: int,
                    task_id: str, mode: str, skill_root: Optional[pathlib.Path],
                    repo_root: Optional[pathlib.Path] = None) -> pathlib.Path:
    ws = out_dir / "episodes" / run_id / f"gen{generation}" / f"{task_id}_{mode}"
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    if repo_root is not None:
        # THE FAITHFUL TEST: the agent gets the whole repo it's contributing to, not just
        # an isolated skill package. no_skill mode gets the bare repo (skills/ removed);
        # v0/a/b get the repo with the skill present + installed for auto-discovery.
        _copy_full_repo(ws, repo_root, mode)
        if mode != "no_skill" and skill_root is not None:
            _seed_skill(ws / "skill", skill_root)   # back-compat mirror for inline solvers
            _install_skill(ws, skill_root)          # .claude/skills/ for claude-code discovery
        _write_baseline_manifest(ws)                # snapshot BEFORE the agent touches anything
    elif mode != "no_skill" and skill_root is not None:
        _seed_skill(ws / "skill", skill_root)                 # for inline solvers (qwen/glm-api)
        _install_skill(ws, skill_root)                        # for the real claude-code harness
    return ws


# top-level entries of the repo that are safe/useful to give the agent. Excludes
# .git (avoid real git/network temptation), the skill's evaluation/ harness (contains
# the ANSWER KEY in proxy_tasks/ — must never be visible to an episode), and generated
# result directories.
_REPO_COPY_INCLUDE = ("src", "tests", "docs", "CONTRIBUTING.md", "README.md",
                      "LICENSE", "pyproject.toml", "requirements.txt", "skills")
_SKILL_EXCLUDE_SUBDIRS = ("evaluation",)  # never expose the harness/answer-key to an episode


def _copy_full_repo(dest: pathlib.Path, repo_root: pathlib.Path, mode: str) -> None:
    for name in _REPO_COPY_INCLUDE:
        src = repo_root / name
        if not src.exists():
            continue
        if name == "skills":
            if mode == "no_skill":
                continue  # bare repo: no skill content at all, not even unused on disk
            _copy_skill_tree(src, dest / "skills")
            continue
        if src.is_dir():
            shutil.copytree(src, dest / name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                          ".pytest_cache"))
        else:
            shutil.copy2(src, dest / name)


def _copy_skill_tree(skills_src: pathlib.Path, skills_dest: pathlib.Path) -> None:
    """Copy skills/ preserving structure, but excluding each skill's evaluation/ subdir
    (the harness + answer-key proxy_tasks references — never given to an episode)."""
    for item in skills_src.rglob("*"):
        rel = item.relative_to(skills_src)
        if any(part in _SKILL_EXCLUDE_SUBDIRS for part in rel.parts):
            continue
        if "__pycache__" in rel.parts or item.suffix == ".pyc":
            continue
        target = skills_dest / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


_MANIFEST_EXCLUDE_PARTS = (".claude", ".git", "__pycache__", ".pytest_cache", "skill")
_MANIFEST_EXCLUDE_NAMES = ("_prompt.txt", "_baseline_manifest.json")


def _walkable(rel: pathlib.PurePath) -> bool:
    """True if `rel` is a real repo/deliverable path worth tracking — excludes hidden
    dirs (.claude, .git, .pytest_cache), harness bookkeeping, and the solver-compat
    skill/ mirror."""
    return (not any(part.startswith(".") for part in rel.parts)
           and not any(part in _MANIFEST_EXCLUDE_PARTS for part in rel.parts)
           and rel.name not in _MANIFEST_EXCLUDE_NAMES)


def _write_baseline_manifest(ws: pathlib.Path) -> None:
    """Snapshot (relpath -> (size, mtime_ns)) for every pre-existing file, taken right
    after the repo is copied in and BEFORE the agent runs. changed_files_repo_aware()
    diffs against this to find exactly what the agent added/modified — a real diff,
    not a whole-tree glob (which would be hundreds of pre-existing repo files)."""
    manifest = {}
    for p in ws.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ws)
        if not _walkable(rel):
            continue
        st = p.stat()
        manifest[str(rel)] = [st.st_size, st.st_mtime_ns]
    (ws / "_baseline_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


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


def _install_skill(ws: pathlib.Path, skill_root: pathlib.Path) -> None:
    """Install the FULL skill (SKILL.md + references + scripts + examples) as a real
    Claude Code skill under ws/.claude/skills/<name>/, so the agent auto-discovers it and
    can run its scripts + read its worked examples — the faithful test of the whole skill,
    not a text excerpt. Excludes evaluation/ (harness) and results/ to keep it clean."""
    name = skill_root.name  # ninetoothed-operator-dev
    dest = ws / ".claude" / "skills" / name
    dest.mkdir(parents=True, exist_ok=True)
    for item in skill_root.iterdir():
        if item.name in ("evaluation", ".git", "__pycache__"):
            continue
        if item.is_dir():
            shutil.copytree(item, dest / item.name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(item, dest / item.name)


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
        # skip hidden dirs/files: .claude (installed skill), .pytest_cache, .git, etc. —
        # these are tool/cache artifacts, not the agent's deliverable edits.
        if any(part.startswith(".") for part in rel.parts):
            continue
        if any(part.startswith("_") for part in rel.parts):   # harness artifacts
            continue
        if rel.name in ("oracle_test.py",):                    # harness-supplied grader
            continue
        out.append(str(rel))
    return out


def changed_files_repo_aware(ws: pathlib.Path) -> list[str]:
    """Real diff against the pre-agent snapshot (_write_baseline_manifest): a file counts
    as changed iff it's new or its (size, mtime) differs from the baseline. Necessary once
    the sandbox is a full repo clone — "any file present" would return hundreds of
    pre-existing repo files instead of what the agent actually touched."""
    manifest_path = ws / "_baseline_manifest.json"
    if not manifest_path.exists():
        return changed_files(ws)   # no baseline recorded — fall back to legacy heuristic
    baseline = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = []
    for p in ws.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ws)
        if rel.name in ("oracle_test.py",) or rel.name.startswith("test_harness_oracle_"):
            continue     # harness-supplied grader, not the agent's own edit
        if not _walkable(rel):
            continue
        key = str(rel)
        st = p.stat()
        cur = [st.st_size, st.st_mtime_ns]
        if key not in baseline or baseline[key] != cur:
            out.append(key)
    return out


# --------------------------------------------------------------------------- solvers
def default_claude_solver(prompt: str, ws: pathlib.Path, opts: dict) -> SolverResult:
    """Invoke `claude -p --output-format json` inside the sandbox — the REAL agent harness.

    This is the faithful test: a tool-using coding agent that can read the whole skill
    (references + scripts + examples installed under ws/.claude/skills/) and run its
    workflow via Bash. Backed by whatever model ANTHROPIC_BASE_URL/AUTH_TOKEN +
    ANTHROPIC_DEFAULT_*_MODEL point at (e.g. glm-5.2 via Zhipu's Anthropic endpoint).
    Needs IS_SANDBOX=1 in the env so --dangerously-skip-permissions works headless as root.

    `compiled` here only reflects whether the CLI call itself succeeded (no crash/timeout/
    API error) — NOT whether the deliverable is correct. That finer distinction (produced
    nothing vs produced-but-wrong vs correct) is derived downstream in run_episode() from
    the rubric's oracle result, which works uniformly for flat and repo-aware sandboxes
    instead of assuming a fixed file layout inside the solver.
    """
    model = opts.get("model") or os.environ.get("NINETOOTHED_EPISODE_MODEL", "")
    timeout = int(opts.get("timeout", 1200))
    cmd = ["claude", "-p", "--output-format", "json", "--dangerously-skip-permissions"]
    if model:
        cmd += ["--model", model]
    cmd += [prompt]
    env = {**os.environ, "IS_SANDBOX": os.environ.get("IS_SANDBOX", "1")}
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              cwd=str(ws), env=env)
    except subprocess.TimeoutExpired:
        return SolverResult(wall_seconds=timeout, compiled=False, compile_attempts=0,
                            raw={"error": "timeout"})
    wall = time.time() - t0
    return _parse_claude_json(proc.stdout, wall)


def _parse_claude_json(stdout: str, wall: float) -> SolverResult:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        # streaming or plain text — still count wall time, tokens unknown; treat as a
        # solver-level failure (couldn't even confirm the CLI call completed cleanly).
        return SolverResult(wall_seconds=wall, compiled=False,
                            raw={"stdout_head": stdout[:200]})
    usage = data.get("usage", {}) or {}
    tin = usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
    tout = usage.get("output_tokens", 0)
    return SolverResult(
        tokens_in=tin, tokens_out=tout,
        cost_usd=data.get("total_cost_usd", 0.0),
        wall_seconds=data.get("duration_ms", wall * 1000) / 1000.0,
        compiled=not bool(data.get("is_error", False)),
        raw={"num_turns": data.get("num_turns"), "subtype": data.get("subtype")},
    )


def resolve_solver(name: str, model_dir: Optional[str] = None,
                   fake_files: Optional[dict] = None) -> Optional[Solver]:
    """Map a --solver name to a solver callable.
    'claude' -> None (run_episode uses default_claude_solver);
    'qwen'   -> local Qwen coder (needs --model-dir, torch+transformers on host);
    'glm'    -> hosted GLM over the Zhipu API (needs GLM_API_KEY; GLM_MODEL optional);
    'fake'   -> stub writer for offline wiring checks."""
    if name == "claude":
        return None
    if name == "qwen":
        if not model_dir:
            raise SystemExit("--solver qwen requires --model-dir")
        from qwen_solver import make_qwen_solver
        return make_qwen_solver(model_dir)
    if name == "glm":
        # default to the agentic tool-calling solver (GLM drives write_file/run_test);
        # GLM_AGENT=0 falls back to the scripted single-shot+repair solver.
        if os.environ.get("GLM_AGENT", "1") == "0":
            from glm_solver import make_glm_solver
            return make_glm_solver()
        from glm_solver import make_glm_agent_solver
        return make_glm_agent_solver()
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
                dry_run: bool = False, prompt_override: Optional[str] = None,
                repo_root: Optional[str | pathlib.Path] = None):
    """
    Returns (EpisodeRecord | dict, RubricResult). The caller logs the record.
    prompt_override lets version A inject a bandit-chosen strategy/prompt framing;
    when None the default task+skill prompt is used.

    repo_root: when given, the episode sandbox is a full clone of the repo the skill
    belongs to (src/, tests/, docs/, CONTRIBUTING.md — see prepare_sandbox/_copy_full_repo)
    instead of just the skill package. This is the faithful test of a coding agent
    actually contributing to the project, not an isolated skill-text excerpt.
    """
    skill_root = pathlib.Path(skill_root)
    out_dir = pathlib.Path(out_dir)
    repo_root = pathlib.Path(repo_root) if repo_root else None
    repo_aware = repo_root is not None
    ws = prepare_sandbox(out_dir, run_id, generation, task_meta["id"], mode, skill_root,
                         repo_root=repo_root)
    prompt = prompt_override or build_prompt(task_meta, mode, repo_aware=repo_aware)
    (ws / "_prompt.txt").write_text(prompt, encoding="utf-8")

    if dry_run:
        print(f"[dry-run] sandbox={ws}")
        print(f"[dry-run] would run claude -p in {ws} with prompt:\n{prompt}")
        return None, None

    solver = solver or default_claude_solver
    opts = {**(solver_opts or {}), "task_id": task_meta["id"],
            "kind": task_meta.get("kind", "operator"),
            "proxy_tasks_dir": str(skill_root / "evaluation" / "proxy_tasks"),
            "repo_aware": repo_aware}
    sres = solver(prompt, ws, opts)

    # For operator tasks, drop in the harness oracle test so completion is tamper-proof.
    is_operator = task_meta.get("kind", "operator") == "operator"
    if is_operator:
        try:
            if repo_aware:
                from oracle import write_oracle_test_repo_aware
                write_oracle_test_repo_aware(task_meta, ws, skill_root / "evaluation" / "proxy_tasks")
            else:
                from oracle import write_oracle_test
                write_oracle_test(task_meta, ws, skill_root / "evaluation" / "proxy_tasks")
        except Exception:  # noqa: BLE001  (oracle is best-effort; agent test is fallback)
            pass

    changed_fn = changed_files_repo_aware if repo_aware else changed_files
    changed = changed_fn(ws)

    tmeta = dict(task_meta, _mode=mode)
    rubric = score_episode(ws, tmeta, skill_root, out_dir / "rubric", changed,
                           repo_aware=repo_aware)

    # Refine "did it even run" independent of the solver's own claim: for operator tasks,
    # the agent's deliverable file must exist, or nothing was produced to grade at all.
    if is_operator:
        deliverable = (ws / "tests" / f"test_contrib_{task_meta['id']}.py") if repo_aware \
            else (ws / "wrapper.py")
        sres.compiled = sres.compiled and deliverable.exists()

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
            artifacts={"workspace": str(ws), "changed": ",".join(changed[:12])},
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
    p.add_argument("--repo-root", default=None,
                   help="whole-repo clone root (contains src/, tests/, skills/); enables "
                        "the repo-aware sandbox instead of an isolated skill-only one")
    p.add_argument("--manifest", default=None, help="proxy_tasks/manifest.json")
    p.add_argument("--out-dir", default="./evaluation/results/_adhoc")
    p.add_argument("--run-id", default="adhoc")
    p.add_argument("--generation", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--solver", default="claude", choices=["claude", "qwen", "glm", "fake"])
    p.add_argument("--model-dir", default=None, help="local model dir for --solver qwen")
    p.add_argument("--journal", default=None, help="append the record to this JSONL journal")
    args = p.parse_args(argv)

    skill_root = pathlib.Path(args.skill_root)
    manifest = pathlib.Path(args.manifest) if args.manifest else \
        skill_root / "evaluation" / "proxy_tasks" / "manifest.json"
    task_meta = _load_task(manifest, args.task_id)

    solver = resolve_solver(args.solver, args.model_dir) if not args.dry_run else None
    rec, rubric = run_episode(task_meta, args.mode, skill_root, args.out_dir,
                              run_id=args.run_id, generation=args.generation,
                              solver=solver, dry_run=args.dry_run, repo_root=args.repo_root)
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
