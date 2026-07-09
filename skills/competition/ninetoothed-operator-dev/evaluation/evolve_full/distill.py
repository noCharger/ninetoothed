#!/usr/bin/env python3
"""
distill.py — promote high-confidence procedural rules into the shipped skill.

This is version A's bridge from "external learning system" to "deliverable skill".
The bandit/memory live in evaluation/ and never ship; but the rules they distil DO:

  * references/rules.json  — a frozen, machine-readable rule bank (condition -> action
                             + confidence stats). scripts/strategy_selector.py reads it
                             at skill-run time to recommend an optimisation, as a pure,
                             offline, network-free function.
  * references/<family>.md — a human-readable bullet per rule, appended under the
                             family's Pitfalls/Strategies section so a human (or the
                             model reading the skill) sees the learned guidance.

Only rules that clear the confidence bar are distilled, and each carries its stats so
the skill is honest about how well-supported each piece of advice is.
"""
from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import asdict
from typing import Optional

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "evolve_moo"))

from memory import ProceduralRules, Rule  # noqa: E402
from edit_ops import MarkdownDoc, substitute  # noqa: E402

_FAMILY_FILE = {
    "elementwise": "references/elementwise.md",
    "reduction": "references/reduction.md",
    "layout": "references/layout.md",
    "perf_diag": "references/perf-diag.md",
}

_STRATEGY_ADVICE = {
    "upcast_accumulate": "accumulate in fp32 before writing back (fixes fp16 precision)",
    "vectorize": "widen loads/stores to vectorised access for coalescing",
    "tiling": "tune the block/tile sizes for this shape",
    "fusion": "fuse the elementwise/residual step into the kernel",
    "reduce_registers": "lower register pressure to raise occupancy",
    "pipeline": "add pipeline stages / software prefetch",
    "layout_change": "change the tile layout for coalesced access",
    "num_warps_tune": "sweep num_warps / num_stages",
}


def rule_to_bullet(r: Rule) -> str:
    c = r.condition
    strat = r.action.get("strategy", "?")
    advice = _STRATEGY_ADVICE.get(strat, strat)
    cond = f"{c.get('family','?')} / {c.get('bottleneck','any')}"
    return (f"- **Learned ({cond}):** {advice}. "
            f"_(support n={r.times_used}, success={r.success_rate():.0%}, "
            f"confidence={r.confidence():.2f})_")


def distill(rules: ProceduralRules, skill_root: str | pathlib.Path,
            min_uses: int = 3, min_conf: float = 0.6, min_reward: float = 0.2,
            write: bool = True) -> dict:
    """
    Write references/rules.json and append per-family bullets. Returns a summary dict.
    """
    skill_root = pathlib.Path(skill_root)
    hc = rules.high_confidence(min_uses=min_uses, min_conf=min_conf, min_reward=min_reward)

    # 1) machine-readable frozen bank
    bank = {
        "version": 1,
        "note": "Distilled from version-A evolution. Read by scripts/strategy_selector.py.",
        "thresholds": {"min_uses": min_uses, "min_conf": min_conf, "min_reward": min_reward},
        "rules": [
            {
                "id": r.id,
                "condition": r.condition,
                "action": r.action,
                "stats": {"times_used": r.times_used, "success_rate": round(r.success_rate(), 3),
                          "confidence": round(r.confidence(), 3), "q_reward": round(r.q_reward, 3),
                          "q_speedup": round(r.q_speedup, 3)},
            }
            for r in hc
        ],
    }
    refs = skill_root / "references"
    bank_path = refs / "rules.json"
    if write:
        refs.mkdir(parents=True, exist_ok=True)
        bank_path.write_text(json.dumps(bank, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2) human-readable bullets per family
    per_family: dict[str, list[Rule]] = {}
    for r in hc:
        per_family.setdefault(r.condition.get("family", "unknown"), []).append(r)

    edited_files = []
    for family, frules in per_family.items():
        rel = _FAMILY_FILE.get(family)
        if not rel:
            continue
        fp = skill_root / rel
        if not fp.exists():
            continue
        bullets = "\n".join(rule_to_bullet(r) for r in frules)
        block = ("Guidance below is distilled from measured runs; each item states its "
                 "support.\n\n" + bullets)
        doc = MarkdownDoc.parse(fp.read_text(encoding="utf-8"))
        heading = _learned_heading(doc)
        if heading is None:
            # append a new section
            new_text = fp.read_text(encoding="utf-8").rstrip() + \
                f"\n\n## Learned strategies\n\n{block}\n"
        else:
            new_text = substitute(doc, heading, block).render()
        if write:
            fp.write_text(new_text, encoding="utf-8")
        edited_files.append(rel)

    return {"n_rules": len(hc), "rules_json": str(bank_path),
            "edited_files": edited_files,
            "rule_ids": [r.id for r in hc]}


def _learned_heading(doc: MarkdownDoc) -> Optional[str]:
    for h in doc.headings():
        if "learned strateg" in h.lower():
            return h
    return None


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Distil procedural rules into the skill.")
    p.add_argument("--rules", required=True, help="rules.json from ProceduralRules")
    p.add_argument("--skill-root", required=True)
    p.add_argument("--min-uses", type=int, default=3)
    p.add_argument("--min-conf", type=float, default=0.6)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    pr = ProceduralRules(args.rules)
    summary = distill(pr, args.skill_root, min_uses=args.min_uses, min_conf=args.min_conf,
                      write=not args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
