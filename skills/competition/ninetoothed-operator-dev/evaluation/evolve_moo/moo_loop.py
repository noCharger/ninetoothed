#!/usr/bin/env python3
"""
moo_loop.py — version B: the proposal/mid-term route recast as self-evolution.

Each generation is one deterministic step:

    1. evaluate the CURRENT skill on the train split  -> per-task rubric totals + failures
    2. attribute failures -> ranked edit proposals (guidance gaps only)
    3. for each proposal (best first):
         a. materialise a CANDIDATE skill (copy + apply the edit)
         b. re-evaluate the candidate on train
         c. pass-preservation guard: accept iff no task regresses and total not lower
       promote the first accepted candidate; if none pass, the generation is a no-op
    4. log a GenerationRecord (fitness, edit, skill-diff ref, guard verdict) + optional
       git commit/tag so every generation's diff is reproducible.

There is NO bandit / controller / memory here — that's version A. This is pure outer
evolution over the skill text, which is exactly what the proposal's Stage 3 called for,
but driven by the failure attribution instead of by hand.

Offline: pass a fake episode solver and the template drafter to exercise the whole loop
without a GPU or claude; pass --no-git to skip tagging.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Optional

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "proxy_runner"))
sys.path.insert(0, str(_HERE.parent / "journal"))

from run_episode import run_episode as _run_episode, make_fake_solver as _make_fake  # noqa: E402
from journal import Journal, GenerationRecord  # noqa: E402
from edit_ops import apply_proposal, pass_preservation_guard, EditError  # noqa: E402
from attribute import attribute, Failure, template_drafter, llm_drafter  # noqa: E402

# files/dirs to copy into a candidate skill (skip results/episodes to stay small)
_CANDIDATE_INCLUDE = ("SKILL.md", "README.md", "REFERENCE.md", "requirements.txt",
                      "requirements-optional.txt", "pytest.ini", "references", "scripts",
                      "tests", "examples")
_CANDIDATE_EVAL = ("evaluation/proxy_tasks", "evaluation/skill_eval")


def materialise_candidate(skill_root: pathlib.Path, dest: pathlib.Path) -> pathlib.Path:
    """Copy the skill-facing parts + eval infra needed to score a candidate."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in _CANDIDATE_INCLUDE:
        src = skill_root / name
        if src.is_dir():
            shutil.copytree(src, dest / name, dirs_exist_ok=True)
        elif src.exists():
            shutil.copy2(src, dest / name)
    for rel in _CANDIDATE_EVAL:
        src = skill_root / rel
        if src.exists():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest / rel, dirs_exist_ok=True)
    return dest


def evaluate_split(skill_root: pathlib.Path, tasks: list[dict], mode: str,
                   out_dir: pathlib.Path, run_id: str, generation: int,
                   solver=None, journal: Optional[Journal] = None) -> tuple[dict, list[Failure]]:
    """Run every task once; return {task_id: total} and the failure list."""
    totals: dict[str, int] = {}
    failures: list[Failure] = []
    for t in tasks:
        rec, rubric = _run_episode(t, mode, skill_root, out_dir, run_id=run_id,
                                   generation=generation, solver=solver)
        totals[t["id"]] = rubric.total
        if journal and rec is not None:
            journal.append_episode(rec)
        if rubric.subscores["completion"].value < 4:
            # feed the REAL failure detail (traceback/assertion) to attribution, so the
            # classifier can recognise the guidance gap; fall back to the summary reason.
            err = rubric.completion_error or rubric.subscores["completion"].reason
            failures.append(Failure(task_id=t["id"], family=_family(t), error_text=err))
    return totals, failures


def _family(task: dict) -> str:
    return task.get("category", "unknown")


def load_train_tasks(skill_root: pathlib.Path, family: str = None, limit: int = None) -> list[dict]:
    manifest = skill_root / "evaluation" / "proxy_tasks" / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    tasks = [t for t in data["tasks"] if t.get("split") == "train"]
    if family:
        tasks = [t for t in tasks if t.get("category") == family]
    if limit:
        tasks = tasks[:limit]
    return tasks


@dataclass
class LoopConfig:
    generations: int = 5
    max_proposals: int = 3
    use_git: bool = True
    allow_flat: bool = True
    family: str = None      # restrict train tasks to one operator family
    limit: int = None       # cap number of train tasks (cost control)


def _git(skill_repo: pathlib.Path, *args: str) -> Optional[str]:
    try:
        proc = subprocess.run(["git", "-C", str(skill_repo), *args],
                              capture_output=True, text=True, timeout=60)
        return proc.stdout.strip() if proc.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def run_loop(skill_root: str | pathlib.Path, out_dir: str | pathlib.Path,
             journal_path: str, cfg: Optional[LoopConfig] = None,
             solver: Optional[Callable] = None, drafter=None, mode: str = "b") -> dict:
    cfg = cfg or LoopConfig()
    skill_root = pathlib.Path(skill_root).resolve()
    out_dir = pathlib.Path(out_dir)
    jnl = Journal(journal_path)
    drafter = drafter or llm_drafter

    tasks = load_train_tasks(skill_root, cfg.family, cfg.limit)
    history = []
    applied: set[tuple] = set()   # (op, target_file, heading, before_heading) already promoted

    # generation 0: evaluate the incoming v0 skill as the starting fitness
    cur_totals, failures = evaluate_split(skill_root, tasks, mode, out_dir / "b-gen0",
                                          run_id=mode, generation=0, solver=solver, journal=jnl)
    jnl.append_generation(GenerationRecord(
        run_id=mode, generation=0, train_total=sum(cur_totals.values()),
        train_pass_rate=_pass_rate(cur_totals, tasks), edit_op="none",
        edit_summary="v0 starting fitness", per_task=cur_totals, accepted=True,
        skill_diff_ref=_git(skill_root, "rev-parse", "--short", "HEAD") or "",
    ))
    history.append({"generation": 0, "total": sum(cur_totals.values()), "accepted": True})

    for gen in range(1, cfg.generations + 1):
        proposals = attribute(failures, skill_root, drafter=drafter,
                              max_proposals=cfg.max_proposals)
        # drop edits already promoted in a previous generation (a reorder/prune is
        # idempotent; re-applying wastes a generation without changing fitness)
        proposals = [p for p in proposals
                     if (p.op, p.target_file, p.heading, p.before_heading) not in applied]
        if not proposals:
            jnl.append_generation(GenerationRecord(
                run_id=mode, generation=gen, train_total=sum(cur_totals.values()),
                train_pass_rate=_pass_rate(cur_totals, tasks), edit_op="none",
                edit_summary="no guidance-error proposals (all code_error/unknown)",
                per_task=cur_totals, accepted=False, guard_verdict="no-op"))
            history.append({"generation": gen, "total": sum(cur_totals.values()),
                            "accepted": False, "reason": "no proposals"})
            continue

        promoted = False
        for prop in proposals:
            cand = materialise_candidate(skill_root, out_dir / f"candidate-gen{gen}")
            try:
                apply_proposal(cand, prop, write=True)
            except EditError as e:
                continue
            cand_totals, cand_failures = evaluate_split(
                cand, tasks, mode, out_dir / f"b-gen{gen}-cand",
                run_id=mode, generation=gen, solver=solver, journal=None)
            verdict = pass_preservation_guard(cur_totals, cand_totals, allow_flat=cfg.allow_flat)
            if verdict.accepted:
                # promote: copy the edited file(s) back into the live skill, commit+tag
                _promote(cand, skill_root, prop)
                tag = f"skill-{mode}-gen{gen}"
                if cfg.use_git:
                    _git(skill_root, "add", "-A")
                    _git(skill_root, "commit", "-m",
                         f"evolve({mode}) gen{gen}: {prop.one_line()}")
                    _git(skill_root, "tag", "-f", tag)
                jnl.append_generation(GenerationRecord(
                    run_id=mode, generation=gen, train_total=verdict.train_total_after,
                    train_pass_rate=_pass_rate(cand_totals, tasks),
                    edit_op=prop.op, edit_target=prop.target_file,
                    edit_summary=prop.one_line(),
                    skill_diff_ref=(_git(skill_root, "rev-parse", "--short", "HEAD") or tag),
                    guard_verdict=f"accepted: {verdict.reason}",
                    accepted=True, per_task=cand_totals))
                cur_totals, failures = cand_totals, cand_failures
                applied.add((prop.op, prop.target_file, prop.heading, prop.before_heading))
                history.append({"generation": gen, "total": verdict.train_total_after,
                                "accepted": True, "edit": prop.one_line()})
                promoted = True
                break
            else:
                # log the rejection but keep trying the next proposal
                jnl.append_generation(GenerationRecord(
                    run_id=mode, generation=gen, train_total=verdict.train_total_before,
                    train_pass_rate=_pass_rate(cur_totals, tasks),
                    edit_op=prop.op, edit_target=prop.target_file,
                    edit_summary=prop.one_line(), accepted=False,
                    guard_verdict=f"rejected: {verdict.reason}", per_task=cur_totals))
        if not promoted:
            history.append({"generation": gen, "total": sum(cur_totals.values()),
                            "accepted": False, "reason": "all proposals rejected by guard"})

    return {"history": history, "final_total": sum(cur_totals.values()),
            "journal": journal_path}


def _promote(cand: pathlib.Path, skill_root: pathlib.Path, prop) -> None:
    src = cand / prop.target_file
    dst = skill_root / prop.target_file
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _pass_rate(totals: dict, tasks: list[dict]) -> float:
    # a task "passes" if its total >= 7 (completion>=3 out of the 10-pt rubric floor)
    if not totals:
        return 0.0
    n = sum(1 for tid, v in totals.items() if v >= 7)
    return round(n / len(totals), 3)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run the version-B light-MOO evolution loop.")
    p.add_argument("--skill-root", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--journal", required=True)
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--mode", default="b")
    p.add_argument("--no-git", action="store_true")
    p.add_argument("--fake", action="store_true", help="offline: stub solver + template drafter")
    p.add_argument("--solver", default="claude", choices=["claude", "qwen", "fake"])
    p.add_argument("--model-dir", default=None, help="local model dir for --solver qwen")
    p.add_argument("--family", default=None, help="restrict to one operator family")
    p.add_argument("--limit", type=int, default=None, help="cap number of train tasks")
    args = p.parse_args(argv)

    skill_root = pathlib.Path(args.skill_root).resolve()
    out_dir = args.out_dir or skill_root / "evaluation" / "results"
    cfg = LoopConfig(generations=args.generations, use_git=not args.no_git,
                     family=args.family, limit=args.limit)
    if args.fake:
        solver, drafter = _make_fake(_fake_files()), template_drafter
    else:
        from run_episode import resolve_solver
        solver = resolve_solver(args.solver, args.model_dir)
        if args.solver == "qwen":
            from qwen_solver import make_qwen_drafter
            drafter = make_qwen_drafter(args.model_dir)
        else:
            drafter = llm_drafter
    result = run_loop(skill_root, out_dir, args.journal, cfg, solver=solver,
                      drafter=drafter, mode=args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _fake_files() -> dict:
    from run_matrix import _FAKE_FILES
    return _FAKE_FILES


if __name__ == "__main__":
    raise SystemExit(main())
