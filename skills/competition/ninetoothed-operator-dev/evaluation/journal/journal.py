#!/usr/bin/env python3
"""
journal.py — append-only JSONL ledger for the self-evolving optimisation loop.

"Journal first": every episode and every generation writes one line here BEFORE
any evolution machinery runs, so the learning process is fully reconstructable
even if a loop is interrupted. The final report's three views
(learning-before / learning-during / learning-after) are all reads over this file.

Design goals
------------
* stdlib only, import-safe on any machine (no torch/ninetoothed) — same discipline
  as evaluation/proxy_tasks and evaluation/skill_eval.
* append-only, one JSON object per line, flushed + fsync'd on each write so a
  killed process never loses committed records.
* two record types share the file, discriminated by "kind":
    - "episode"     : one task solved under one mode (baseline/a/b), with scores,
                      reward, cost, and failure classification.
    - "generation"  : one evolution step summary (fitness, edit applied, skill diff
                      ref, guard verdict).
* schema is intentionally open (extra keys allowed) but the documented fields are
  guaranteed present via the dataclasses below, so curves.py can rely on them.

Usage
-----
    from journal import Journal, EpisodeRecord, GenerationRecord

    jnl = Journal("results/run-2026-07-07.jsonl")
    jnl.append_episode(EpisodeRecord(
        run_id="b", generation=1, task_id="ew01", mode="b",
        strategy="baseline-solve", prompt_id="diagnose-first",
        scores={"completion": 4, "test": 2, "perf": 1, "minimality": 1,
                "style": 1, "compliance": 1},
        reward=1.42, cost={"tokens_in": 5300, "tokens_out": 1800,
                           "gpu_seconds": 12.4, "wall_seconds": 41.0},
        classification="none", artifacts={"kernel": ".../kernel.py"},
    ))

CLI:
    python journal.py stats results/run.jsonl      # counts by kind/mode/generation
    python journal.py tail  results/run.jsonl 5     # last N records pretty-printed
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

SCHEMA_VERSION = 1

# The six rubric sub-scores (verifier_spec.md). Kept here so producers and the
# scorer agree on names and maxima.
RUBRIC_KEYS = ("completion", "test", "perf", "minimality", "style", "compliance")
RUBRIC_MAX = {"completion": 4, "test": 2, "perf": 1, "minimality": 1, "style": 1, "compliance": 1}
RUBRIC_TOTAL = sum(RUBRIC_MAX.values())  # 10


def rubric_total(scores: dict) -> int:
    """Sum the six sub-scores, ignoring unknown keys, clamping to each max."""
    return sum(min(int(scores.get(k, 0)), RUBRIC_MAX[k]) for k in RUBRIC_KEYS)


@dataclass
class EpisodeRecord:
    """One task solved once, under one mode."""
    run_id: str                       # "baseline" | "a" | "b" (which experiment)
    generation: int                   # 0 for baseline; 1..N for evolution steps
    task_id: str                      # proxy task id, e.g. "ew01"
    mode: str                         # "no_skill" | "v0" | "a" | "b" (harness condition)
    strategy: str = ""                # evolve_full action label; "" for plain solve
    prompt_id: str = ""               # which prompt template was used
    scores: dict = field(default_factory=dict)     # six rubric sub-scores
    reward: float = 0.0               # scalar reward used by the loop
    cost: dict = field(default_factory=dict)       # tokens_in/out, gpu_seconds, wall_seconds
    classification: str = "none"      # failure_classifier verdict (or "none" if passed)
    artifacts: dict = field(default_factory=dict)  # name -> path of produced files
    notes: str = ""

    kind: str = "episode"

    def total(self) -> int:
        return rubric_total(self.scores)


@dataclass
class GenerationRecord:
    """One evolution step summary."""
    run_id: str                       # "a" | "b"
    generation: int
    train_total: float                # sum/mean rubric total over train tasks
    train_pass_rate: float            # fraction of train tasks with completion>=3
    edit_op: str = ""                 # "prune" | "substitute" | "reorder" | "distill" | "none"
    edit_target: str = ""             # file the edit touched
    edit_summary: str = ""            # human-readable one-liner
    skill_diff_ref: str = ""          # git tag / commit / patch path for the skill state
    guard_verdict: str = ""           # "accepted" | "rejected:<reason>"
    accepted: bool = True
    per_task: dict = field(default_factory=dict)   # task_id -> rubric total this gen
    cost: dict = field(default_factory=dict)
    notes: str = ""

    kind: str = "generation"


class Journal:
    """Append-only JSONL writer/reader. Safe to open concurrently for append."""

    def __init__(self, path: str | os.PathLike):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---- write ----
    def _write(self, obj: dict) -> dict:
        obj = dict(obj)
        obj.setdefault("schema", SCHEMA_VERSION)
        obj.setdefault("ts", time.time())
        obj.setdefault("ts_iso", time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(obj["ts"])))
        line = json.dumps(obj, ensure_ascii=False)
        # open per-write in append mode so multiple processes can interleave safely
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        return obj

    def append(self, record: Any) -> dict:
        """Append a dataclass record or a raw dict."""
        if dataclasses.is_dataclass(record) and not isinstance(record, type):
            return self._write(dataclasses.asdict(record))
        if isinstance(record, dict):
            return self._write(record)
        raise TypeError(f"append expects a dataclass or dict, got {type(record)!r}")

    def append_episode(self, rec: EpisodeRecord) -> dict:
        d = dataclasses.asdict(rec)
        d["rubric_total"] = rec.total()   # denormalise for easy downstream reads
        return self._write(d)

    def append_generation(self, rec: GenerationRecord) -> dict:
        return self._write(dataclasses.asdict(rec))

    # ---- read ----
    def read_all(self) -> list[dict]:
        return list(self.iter())

    def iter(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:  # skip a torn final line, don't crash
                    sys.stderr.write(f"[journal] skipping malformed line {lineno}: {e}\n")

    def episodes(self, run_id: Optional[str] = None, generation: Optional[int] = None) -> list[dict]:
        out = []
        for r in self.iter():
            if r.get("kind") != "episode":
                continue
            if run_id is not None and r.get("run_id") != run_id:
                continue
            if generation is not None and r.get("generation") != generation:
                continue
            out.append(r)
        return out

    def generations(self, run_id: Optional[str] = None) -> list[dict]:
        out = [r for r in self.iter() if r.get("kind") == "generation"]
        if run_id is not None:
            out = [r for r in out if r.get("run_id") == run_id]
        out.sort(key=lambda r: (r.get("run_id", ""), r.get("generation", 0)))
        return out


# --------------------------------------------------------------------------- CLI
def _cmd_stats(path: str) -> int:
    jnl = Journal(path)
    from collections import Counter
    kinds: Counter = Counter()
    modes: Counter = Counter()
    gens: Counter = Counter()
    n = 0
    for r in jnl.iter():
        n += 1
        kinds[r.get("kind", "?")] += 1
        if r.get("kind") == "episode":
            modes[r.get("mode", "?")] += 1
            gens[(r.get("run_id", "?"), r.get("generation", 0))] += 1
    print(f"journal: {path}")
    print(f"  records      : {n}")
    print(f"  by kind      : {dict(kinds)}")
    print(f"  episode modes: {dict(modes)}")
    print("  episodes per (run, gen):")
    for k in sorted(gens):
        print(f"    {k[0]:<10} gen {k[1]:<3} : {gens[k]}")
    return 0


def _cmd_tail(path: str, n: int) -> int:
    recs = Journal(path).read_all()[-n:]
    for r in recs:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "stats" and len(argv) >= 2:
        return _cmd_stats(argv[1])
    if cmd == "tail" and len(argv) >= 2:
        n = int(argv[2]) if len(argv) >= 3 else 5
        return _cmd_tail(argv[1], n)
    print("usage: journal.py stats <file> | tail <file> [n]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
