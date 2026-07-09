#!/usr/bin/env python3
"""
curves.py — turn a journal.jsonl into the report's learning views.

Reads the append-only journal and emits:
  * per-generation fitness curve (train total + pass_rate) for each run (a, b)
  * a three-column comparison table no_skill / v0 / v1 from the baseline + final runs
  * cost accounting (tokens, gpu-seconds) per generation

Text/CSV/Markdown output only (stdlib); matplotlib is optional and used only if
--png is given, so this runs on the GPU host or the Mac without extra deps.

Usage:
    python curves.py summary  <journal.jsonl>
    python curves.py curve    <journal.jsonl> --run b [--csv out.csv]
    python curves.py compare  <journal.jsonl> --modes no_skill v0 --final-run final
    python curves.py png      <journal.jsonl> --run b --out curve_b.png   # needs matplotlib
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from collections import defaultdict

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from journal import Journal  # noqa: E402


def _gen_curve(jnl: Journal, run: str) -> list[dict]:
    """One row per generation: prefer the generation record; else aggregate episodes."""
    gens = jnl.generations(run_id=run)
    if gens:
        return [{"generation": g["generation"], "train_total": g.get("train_total", 0),
                 "pass_rate": g.get("train_pass_rate", 0), "accepted": g.get("accepted", True),
                 "edit_op": g.get("edit_op", ""), "edit_summary": g.get("edit_summary", "")}
                for g in gens]
    # fallback: aggregate episodes by generation
    by_gen: dict[int, list[dict]] = defaultdict(list)
    for e in jnl.episodes(run_id=run):
        by_gen[e.get("generation", 0)].append(e)
    rows = []
    for gen in sorted(by_gen):
        eps = by_gen[gen]
        total = sum(e.get("rubric_total", 0) for e in eps)
        npass = sum(1 for e in eps if e.get("scores", {}).get("completion", 0) >= 3)
        rows.append({"generation": gen, "train_total": total,
                     "pass_rate": round(npass / max(1, len(eps)), 3),
                     "accepted": True, "edit_op": "", "edit_summary": ""})
    return rows


def cmd_summary(path: str) -> int:
    jnl = Journal(path)
    runs = sorted({e.get("run_id") for e in jnl.iter() if e.get("kind") == "episode"})
    print(f"journal {path}: runs = {runs}")
    for run in runs:
        eps = jnl.episodes(run_id=run)
        tok = sum(e.get("cost", {}).get("tokens_in", 0) + e.get("cost", {}).get("tokens_out", 0)
                  for e in eps)
        gpu = sum(e.get("cost", {}).get("gpu_seconds", 0) for e in eps)
        gens = sorted({e.get("generation", 0) for e in eps})
        print(f"  run={run:<9} episodes={len(eps):<4} gens={gens} "
              f"tokens={int(tok):,} gpu_s={gpu:.0f}")
    return 0


def cmd_curve(path: str, run: str, csv_path: str | None) -> int:
    rows = _gen_curve(Journal(path), run)
    if not rows:
        print(f"no data for run={run}")
        return 1
    print(f"learning curve — run={run}")
    print(f"  {'gen':>3} {'train_total':>12} {'pass_rate':>10} {'accepted':>9}  edit")
    for r in rows:
        print(f"  {r['generation']:>3} {r['train_total']:>12} {r['pass_rate']:>10} "
              f"{str(r['accepted']):>9}  {r['edit_op']} {r['edit_summary'][:40]}")
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {csv_path}")
    return 0


def cmd_compare(path: str, modes: list[str], baseline_run: str, final_run: str) -> int:
    """Build the no_skill / v0 / v1 three-column table by (task_id, mode)."""
    jnl = Journal(path)
    # collect the latest episode per (task_id, mode) across baseline + final runs
    latest: dict[tuple, dict] = {}
    for e in jnl.iter():
        if e.get("kind") != "episode":
            continue
        if e.get("run_id") not in (baseline_run, final_run):
            continue
        key = (e.get("task_id"), e.get("mode"))
        if key not in latest or e.get("ts", 0) >= latest[key].get("ts", 0):
            latest[key] = e
    task_ids = sorted({k[0] for k in latest})
    print(f"three-way comparison ({' / '.join(modes)}) — baseline={baseline_run} final={final_run}")
    header = f"  {'task':<8}" + "".join(f"{m:>10}" for m in modes) + f"{'v1-v0':>8}"
    print(header)
    col_tot = {m: 0 for m in modes}
    for tid in task_ids:
        cells = []
        for m in modes:
            e = latest.get((tid, m))
            v = e.get("rubric_total", 0) if e else None
            cells.append(v)
            if v is not None:
                col_tot[m] += v
        delta = ""
        if len(modes) >= 2 and cells[-1] is not None and cells[-2] is not None:
            delta = f"{cells[-1]-cells[-2]:+d}"
        row = f"  {tid:<8}" + "".join(f"{('-' if c is None else c):>10}" for c in cells) + f"{delta:>8}"
        print(row)
    print("  " + "-" * (len(header) - 2))
    print(f"  {'TOTAL':<8}" + "".join(f"{col_tot[m]:>10}" for m in modes))
    return 0


def cmd_png(path: str, run: str, out: str) -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; use `curve --csv` and plot elsewhere", file=sys.stderr)
        return 1
    rows = _gen_curve(Journal(path), run)
    if not rows:
        print(f"no data for run={run}")
        return 1
    gens = [r["generation"] for r in rows]
    totals = [r["train_total"] for r in rows]
    passr = [r["pass_rate"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(gens, totals, "o-", color="C0", label="train total")
    ax1.set_xlabel("generation"); ax1.set_ylabel("train rubric total", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(gens, passr, "s--", color="C1", label="pass rate")
    ax2.set_ylabel("pass rate", color="C1"); ax2.set_ylim(0, 1)
    plt.title(f"Learning curve — run {run}")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary"); s.add_argument("journal")
    c = sub.add_parser("curve"); c.add_argument("journal"); c.add_argument("--run", default="b")
    c.add_argument("--csv", default=None)
    cm = sub.add_parser("compare"); cm.add_argument("journal")
    cm.add_argument("--modes", nargs="+", default=["no_skill", "v0"])
    cm.add_argument("--baseline-run", default="baseline"); cm.add_argument("--final-run", default="final")
    pg = sub.add_parser("png"); pg.add_argument("journal"); pg.add_argument("--run", default="b")
    pg.add_argument("--out", default="curve.png")
    args = p.parse_args(argv)

    if args.cmd == "summary":
        return cmd_summary(args.journal)
    if args.cmd == "curve":
        return cmd_curve(args.journal, args.run, args.csv)
    if args.cmd == "compare":
        return cmd_compare(args.journal, args.modes, args.baseline_run, args.final_run)
    if args.cmd == "png":
        return cmd_png(args.journal, args.run, args.out)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
