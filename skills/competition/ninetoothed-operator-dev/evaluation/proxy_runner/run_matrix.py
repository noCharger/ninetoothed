#!/usr/bin/env python3
"""
run_matrix.py — batch-run proxy tasks × modes, score each, log to the journal, and
write a summary CSV. This produces the baseline (learning-before) and every
generation's evaluation.

Examples
--------
    # baseline: 24 tasks, no_skill vs v0, log everything
    python run_matrix.py --skill-root .. --split all --modes no_skill v0 \
        --run-id baseline --journal ../results/journal.jsonl \
        --csv ../results/baseline/matrix.csv

    # one generation's train evaluation for version B
    python run_matrix.py --skill-root <candidate_skill> --split train --modes b \
        --run-id b --generation 3 --journal ../results/journal.jsonl \
        --csv ../results/b-gen-3/train.csv

Offline: --dry-run prints the plan; --fake writes stub artifacts (no claude) so the
whole pipeline can be exercised without a GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "journal"))

from run_episode import run_episode, make_fake_solver, resolve_solver  # noqa: E402
from journal import Journal, GenerationRecord, rubric_total  # noqa: E402


_FAKE_FILES = {
    "kernel.py": "# stub kernel\ndef kernel():\n    return 0\n",
    "wrapper.py": "# stub wrapper\ndef run(*a, **k):\n    return None\n",
    "test_correctness.py": "def test_stub():\n    assert True\n",
    "bench.csv": "size,ms,GB_s\n1024,0.10,50\n4096,0.20,80\n16384,0.40,120\n# verdict: memory-bound\n",
    "diagnosis.md": "Root cause: missing upcast before accumulation; fix: cast to fp32.\n",
}


def load_tasks(manifest: pathlib.Path, split: str, family: str = None) -> list[dict]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    tasks = data["tasks"]
    if split != "all":
        tasks = [t for t in tasks if t.get("split") == split]
    if family:
        tasks = [t for t in tasks if t.get("category") == family]
    return tasks


def run_matrix(skill_root, manifest, split, modes, out_dir, run_id, generation,
               journal_path=None, csv_path=None, fake=False, dry_run=False,
               limit=None, solver_name="claude", model_dir=None, family=None):
    skill_root = pathlib.Path(skill_root)
    out_dir = pathlib.Path(out_dir)
    tasks = load_tasks(pathlib.Path(manifest), split, family)
    if limit:
        tasks = tasks[:limit]
    jnl = Journal(journal_path) if journal_path else None
    solver = make_fake_solver(_FAKE_FILES) if fake else resolve_solver(solver_name, model_dir)

    rows = []
    per_mode_total: dict[str, int] = {m: 0 for m in modes}
    per_mode_pass: dict[str, int] = {m: 0 for m in modes}

    for t in tasks:
        for mode in modes:
            rec, rubric = run_episode(
                t, mode, skill_root, out_dir, run_id=run_id, generation=generation,
                solver=solver, dry_run=dry_run,
            )
            if dry_run:
                continue
            if jnl and rec is not None:
                jnl.append_episode(rec)
            total = rubric.total
            completion = rubric.subscores["completion"].value
            per_mode_total[mode] += total
            per_mode_pass[mode] += 1 if completion >= 3 else 0
            rows.append({
                "task_id": t["id"], "category": t["category"], "split": t["split"],
                "mode": mode, "total": total,
                **{f"s_{k}": v.value for k, v in rubric.subscores.items()},
                "matrix_pass_frac": round(rubric.matrix_pass_frac, 3),
            })

    if dry_run:
        print(f"[dry-run] {len(tasks)} tasks × {len(modes)} modes = "
              f"{len(tasks)*len(modes)} episodes under run_id={run_id} gen={generation}")
        return None

    if csv_path:
        csv_path = pathlib.Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["task_id"])
            w.writeheader()
            w.writerows(rows)

    # per-mode summary + optional generation record (for the evolving runs)
    n_tasks = len(tasks)
    summary = {}
    for mode in modes:
        summary[mode] = {
            "total": per_mode_total[mode],
            "mean_per_task": round(per_mode_total[mode] / max(1, n_tasks), 3),
            "pass_rate": round(per_mode_pass[mode] / max(1, n_tasks), 3),
        }
    if jnl and run_id in ("a", "b") and len(modes) == 1:
        mode = modes[0]
        jnl.append_generation(GenerationRecord(
            run_id=run_id, generation=generation,
            train_total=per_mode_total[mode],
            train_pass_rate=summary[mode]["pass_rate"],
            per_task={r["task_id"]: r["total"] for r in rows if r["mode"] == mode},
            notes=f"split={split} n={n_tasks}",
        ))

    print(f"run_matrix: run_id={run_id} gen={generation} split={split} "
          f"tasks={n_tasks} modes={modes}")
    for mode in modes:
        s = summary[mode]
        print(f"  {mode:<9} total={s['total']:<4} mean/task={s['mean_per_task']:<6} "
              f"pass_rate={s['pass_rate']}")
    if csv_path:
        print(f"  wrote {csv_path}")
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Batch-run proxy tasks × modes.")
    p.add_argument("--skill-root", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--split", default="all", choices=["all", "train", "holdout"])
    p.add_argument("--modes", nargs="+", default=["no_skill", "v0"])
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-id", default="baseline")
    p.add_argument("--generation", type=int, default=0)
    p.add_argument("--journal", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--fake", action="store_true", help="offline stub solver (no claude)")
    p.add_argument("--solver", default="claude", choices=["claude", "qwen", "fake"])
    p.add_argument("--model-dir", default=None, help="local model dir for --solver qwen")
    p.add_argument("--family", default=None, help="restrict to one operator family/category")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    skill_root = pathlib.Path(args.skill_root).resolve()
    manifest = args.manifest or skill_root / "evaluation" / "proxy_tasks" / "manifest.json"
    out_dir = args.out_dir or skill_root / "evaluation" / "results"
    run_matrix(skill_root, manifest, args.split, args.modes, out_dir, args.run_id,
               args.generation, args.journal, args.csv, args.fake, args.dry_run, args.limit,
               solver_name=args.solver, model_dir=args.model_dir, family=args.family)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
