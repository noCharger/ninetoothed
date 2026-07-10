#!/usr/bin/env python3
"""
loop.py — version A: the full self-evolving controller loop.

Realises the plan's §1 命题:
  Black-box LLM (claude -p) + learnable controller (bandit) + execution reward
  + memory (episodic/semantic/procedural) + evolutionary search (generational skill
  distillation), where procedural memory's distillation target IS the skill package.

Per episode:
  1. context   = extract_context(task, last bench verdict)
  2. arm       = bandit.select(context)          # (strategy, prompt framing)
  3. memories  = episodic.retrieve(context)       # past wins/losses on this family
  4. prompt    = strategies.build_prompt(task, arm, memories)
  5. run_episode(prompt) -> artifacts, rubric, shaped reward
  6. bandit.update(context, arm, reward);  rules.update(...);  semantic.add(...)
  7. journal.append(episode with strategy + prompt_id + reward + cost)

Per generation:
  * snapshot the bandit posterior + rules.json into results/a-genN/
  * distil high-confidence rules into a CANDIDATE skill, re-evaluate train, guard,
    and promote (commit + tag skill-a-genN) iff no negative transfer.

This is standard RL over the controller: the third-party model is never trained; the
bandit + rules are. The report shows P(best arm) rising per context (bandit.prob_best).

Offline: --fake uses a stub solver + template distiller so the whole loop runs without
GPU or claude.
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
sys.path.insert(0, str(_HERE.parent / "evolve_moo"))

from strategies import Arm, extract_context, arms_for, build_prompt  # noqa: E402
from bandit import ContextualBandit  # noqa: E402
from memory import ProceduralRules, SemanticMemory, EpisodicMemory  # noqa: E402
from distill import distill  # noqa: E402
from run_episode import run_episode as _run_episode, make_fake_solver as _make_fake  # noqa: E402
from journal import Journal, GenerationRecord  # noqa: E402
from edit_ops import pass_preservation_guard  # noqa: E402
from moo_loop import materialise_candidate, load_train_tasks, evaluate_split, _pass_rate  # noqa: E402


@dataclass
class LoopAConfig:
    generations: int = 4
    use_git: bool = True
    seed: int = 0
    min_uses: int = 3
    min_conf: float = 0.5
    allow_flat: bool = True


def _git(repo: pathlib.Path, *args: str) -> Optional[str]:
    try:
        p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                           text=True, timeout=60)
        return p.stdout.strip() if p.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _arm_from_key(key: str) -> Arm:
    strat, pid = key.split("::", 1)
    return Arm(strat, pid)


def run_loop_a(skill_root, out_dir, journal_path, cfg: Optional[LoopAConfig] = None,
               solver: Optional[Callable] = None, distiller_write: bool = True,
               mode: str = "a") -> dict:
    cfg = cfg or LoopAConfig()
    skill_root = pathlib.Path(skill_root).resolve()
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jnl = Journal(journal_path)

    tasks = load_train_tasks(skill_root)
    bandit = ContextualBandit(seed=cfg.seed)
    rules = ProceduralRules(out_dir / "rules.json")
    sem = SemanticMemory(out_dir / "semantic.json")
    epi = EpisodicMemory(journal_path)

    # gen 0: v0 starting fitness (default prompt, no strategy framing)
    cur_totals, _ = evaluate_split(skill_root, tasks, mode, out_dir / "a-gen0",
                                   run_id=mode, generation=0, solver=solver, journal=jnl)
    jnl.append_generation(GenerationRecord(
        run_id=mode, generation=0, train_total=sum(cur_totals.values()),
        train_pass_rate=_pass_rate(cur_totals, tasks), edit_op="none",
        edit_summary="v0 starting fitness", per_task=cur_totals, accepted=True,
        skill_diff_ref=_git(skill_root, "rev-parse", "--short", "HEAD") or ""))
    history = [{"generation": 0, "total": sum(cur_totals.values()), "accepted": True}]
    prob_trace: list[dict] = []

    for gen in range(1, cfg.generations + 1):
        # ---- inner: one bandit-driven pass over the train tasks ----
        for t in tasks:
            ctx = extract_context(t)
            arm_keys = [a.key() for a in arms_for(ctx)]
            chosen = bandit.select(ctx.key(), arm_keys)
            arm = _arm_from_key(chosen)
            mems = epi.retrieve(ctx.family, ctx.bottleneck)
            prompt = build_prompt(t, arm, mems, with_skill=True)

            rec, rubric = _run_episode(
                t, mode, skill_root, out_dir / f"a-gen{gen}", run_id=mode, generation=gen,
                strategy=arm.strategy, prompt_id=arm.prompt_id, solver=solver,
                prompt_override=prompt)
            reward = rec.reward if rec else 0.0
            correct = rubric.subscores["completion"].value >= 3
            bandit.update(ctx.key(), chosen, reward)
            rules.update({"family": ctx.family, "dtype": ctx.dtype, "bottleneck": ctx.bottleneck},
                         {"strategy": arm.strategy, "prompt_id": arm.prompt_id},
                         reward=reward, correct=correct,
                         speedup=0.0)
            if correct and reward > 0.3:
                sem.add(ctx.key(), f"{arm.strategy} via {arm.prompt_id} worked "
                                   f"(reward={reward:.2f})")
            if rec:
                jnl.append_episode(rec)

        # ---- snapshot posterior + rules for the report ----
        gdir = out_dir / f"a-gen{gen}"
        gdir.mkdir(parents=True, exist_ok=True)
        bandit.save(gdir / "bandit_posterior.json")
        shutil.copy2(rules.path, gdir / "rules_snapshot.json")
        # record how peaked each context's best arm has become (RL criterion evidence)
        for ck in bandit.stats:
            best = bandit.best_arm(ck)
            if best:
                ak, mean, n = best
                p = bandit.prob_best(ck, list(bandit.stats[ck].keys()), ak)
                prob_trace.append({"generation": gen, "context": ck, "best_arm": ak,
                                   "mean": round(mean, 3), "n": n, "prob_best": round(p, 3)})

        # ---- outer: distil high-confidence rules into a candidate skill, guard, promote ----
        cand = materialise_candidate(skill_root, out_dir / f"candidate-gen{gen}")
        summary = distill(rules, cand, min_uses=cfg.min_uses, min_conf=cfg.min_conf,
                          write=distiller_write)
        cand_totals, _ = evaluate_split(cand, tasks, mode, out_dir / f"a-gen{gen}-cand",
                                        run_id=mode, generation=gen, solver=solver, journal=None)
        verdict = pass_preservation_guard(cur_totals, cand_totals, allow_flat=cfg.allow_flat)
        accepted = verdict.accepted and summary["n_rules"] > 0
        if accepted:
            _promote_distilled(cand, skill_root, summary)
            tag = f"skill-{mode}-gen{gen}"
            if cfg.use_git:
                _git(skill_root, "add", "-A")
                _git(skill_root, "commit", "-m",
                     f"evolve({mode}) gen{gen}: distil {summary['n_rules']} rules")
                _git(skill_root, "tag", "-f", tag)
            cur_totals = cand_totals
        jnl.append_generation(GenerationRecord(
            run_id=mode, generation=gen,
            train_total=sum(cur_totals.values()),
            train_pass_rate=_pass_rate(cur_totals, tasks),
            edit_op="distill" if accepted else "distill-rejected",
            edit_target="references/rules.json",
            edit_summary=f"distilled {summary['n_rules']} rules -> {summary['edited_files']}",
            skill_diff_ref=(_git(skill_root, "rev-parse", "--short", "HEAD") or ""),
            guard_verdict=("accepted: " + verdict.reason) if accepted
                          else ("rejected: " + verdict.reason),
            accepted=accepted, per_task=cur_totals))
        history.append({"generation": gen, "total": sum(cur_totals.values()),
                        "accepted": accepted, "n_rules": summary["n_rules"]})

    # persist final controller state next to results
    bandit.save(out_dir / "bandit_final.json")
    (out_dir / "prob_trace.json").write_text(json.dumps(prob_trace, indent=2), encoding="utf-8")
    return {"history": history, "final_total": sum(cur_totals.values()),
            "prob_trace": prob_trace, "journal": str(journal_path)}


def _promote_distilled(cand: pathlib.Path, skill_root: pathlib.Path, summary: dict) -> None:
    # copy rules.json + any edited family files back into the live skill
    for rel in ["references/rules.json", *summary.get("edited_files", [])]:
        src, dst = cand / rel, skill_root / rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run the version-A self-evolving controller loop.")
    p.add_argument("--skill-root", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--journal", required=True)
    p.add_argument("--generations", type=int, default=4)
    p.add_argument("--mode", default="a")
    p.add_argument("--no-git", action="store_true")
    p.add_argument("--fake", action="store_true")
    p.add_argument("--solver", default="claude", choices=["claude", "qwen", "glm", "fake"])
    p.add_argument("--model-dir", default=None, help="local model dir for --solver qwen")
    args = p.parse_args(argv)

    skill_root = pathlib.Path(args.skill_root).resolve()
    out_dir = args.out_dir or skill_root / "evaluation" / "results"
    cfg = LoopAConfig(generations=args.generations, use_git=not args.no_git)
    if args.fake:
        from run_matrix import _FAKE_FILES
        solver = _make_fake(_FAKE_FILES)
    else:
        from run_episode import resolve_solver
        solver = resolve_solver(args.solver, args.model_dir)
    result = run_loop_a(skill_root, out_dir, args.journal, cfg, solver=solver, mode=args.mode)
    print(json.dumps({k: v for k, v in result.items() if k != "prob_trace"},
                     ensure_ascii=False, indent=2))
    print(f"prob_trace rows: {len(result['prob_trace'])} (see {out_dir}/prob_trace.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
