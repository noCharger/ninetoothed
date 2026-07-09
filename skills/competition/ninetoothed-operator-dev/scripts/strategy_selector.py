#!/usr/bin/env python3
"""
strategy_selector.py — recommend an optimisation strategy from the frozen rule bank.

This is the ONLY learned-controller artifact that ships inside the skill. It is a pure,
deterministic, offline, network-free function: given an operator's context (family,
dtype, and a memory/compute-bound verdict from bench_compare.py), it looks up
references/rules.json — a bank distilled from measured runs — and prints the
highest-confidence matching strategy with its supporting statistics.

It performs NO inference and makes NO network calls; it just reads a JSON file that was
frozen at skill-build time. If rules.json is absent (a fresh skill), it falls back to
family defaults so the skill still gives sound advice.

Usage:
    python scripts/strategy_selector.py --family reduction --dtype float16 \
        --bottleneck memory_bound
    # or feed a bench_compare verdict line:
    python scripts/strategy_selector.py --family reduction --bench "…memory-bound…"
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Optional

# family defaults used when no distilled rule matches (mirrors references/*.md advice)
_FAMILY_DEFAULT = {
    "elementwise": ("vectorize", "widen loads/stores for coalescing; fuse residual if present"),
    "reduction":   ("upcast_accumulate", "accumulate in fp32 before writing back (fp16 precision)"),
    "layout":      ("layout_change", "choose a tile layout that keeps global access coalesced"),
    "perf_diag":   ("tiling", "sweep tile/block sizes and num_warps against the Roofline"),
}

_RULES_JSON = pathlib.Path(__file__).resolve().parent.parent / "references" / "rules.json"


def _bottleneck_from_bench(bench_text: str) -> str:
    t = (bench_text or "").lower()
    if "memory" in t:
        return "memory_bound"
    if "compute" in t:
        return "compute_bound"
    return "unknown"


def load_rules(path: Optional[pathlib.Path] = None) -> list[dict]:
    path = path or _RULES_JSON
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("rules", [])
    except (json.JSONDecodeError, OSError):
        return []


def _match_score(rule: dict, family: str, dtype: str, bottleneck: str) -> int:
    c = rule.get("condition", {})
    if c.get("family") != family:
        return -1                      # family must match
    score = 1
    if c.get("bottleneck") in (bottleneck, "unknown", None):
        score += 1 if c.get("bottleneck") == bottleneck else 0
    if c.get("dtype") in (dtype, "*", None):
        score += 1 if c.get("dtype") == dtype else 0
    return score


def recommend(family: str, dtype: str = "float16", bottleneck: str = "unknown",
              rules_path: Optional[pathlib.Path] = None) -> dict:
    """Return {'strategy', 'rationale', 'source', 'stats'} — never raises."""
    rules = load_rules(rules_path)
    candidates = []
    for r in rules:
        s = _match_score(r, family, dtype, bottleneck)
        if s < 0:
            continue
        conf = r.get("stats", {}).get("confidence", 0.0)
        candidates.append((s, conf, r))
    if candidates:
        candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
        r = candidates[0][2]
        return {
            "strategy": r["action"].get("strategy"),
            "prompt_id": r["action"].get("prompt_id"),
            "rationale": f"distilled rule (matched {family}/{bottleneck})",
            "source": "rules.json",
            "stats": r.get("stats", {}),
        }
    strat, why = _FAMILY_DEFAULT.get(family, ("tiling", "tune tile sizes against the Roofline"))
    return {"strategy": strat, "prompt_id": None, "rationale": why,
            "source": "family_default", "stats": {}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", required=True,
                   choices=["elementwise", "reduction", "layout", "perf_diag"])
    p.add_argument("--dtype", default="float16")
    p.add_argument("--bottleneck", default=None,
                   choices=[None, "memory_bound", "compute_bound", "unknown"])
    p.add_argument("--bench", default=None, help="a bench_compare verdict line to parse")
    p.add_argument("--rules", default=None, help="override path to rules.json")
    args = p.parse_args(argv)

    bottleneck = args.bottleneck or _bottleneck_from_bench(args.bench or "")
    rules_path = pathlib.Path(args.rules) if args.rules else None
    rec = recommend(args.family, args.dtype, bottleneck, rules_path)
    print(f"recommended strategy : {rec['strategy']}")
    if rec.get("prompt_id"):
        print(f"prompt framing       : {rec['prompt_id']}")
    print(f"rationale            : {rec['rationale']}")
    print(f"source               : {rec['source']}")
    if rec["stats"]:
        st = rec["stats"]
        print(f"support              : n={st.get('times_used')} "
              f"success={st.get('success_rate')} confidence={st.get('confidence')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
