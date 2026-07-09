#!/usr/bin/env python3
"""
memory.py — version A's three-layer memory.

  episodic   : the raw trajectory. Backed by the journal (append-only JSONL). Used to
               retrieve past attempts on similar operators to inject into prompts.
  semantic   : distilled experience strings keyed by context, e.g.
               "reduction|*|memory_bound -> upcast to fp32 before accumulation".
  procedural : machine-readable rules with running statistics (times_used,
               success_rate, mean_speedup, confidence). Updated Q-value style. These
               are what distil.py promotes into the shipped skill (references/rules.json
               + a human-readable bullet), so learning becomes a deliverable.

All three persist to JSON and are import-safe (stdlib only).
"""
from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field, asdict
from typing import Optional

# --------------------------------------------------------------------------- episodic
class EpisodicMemory:
    """Read view over the journal for prompt-time retrieval."""

    def __init__(self, journal_path: str | pathlib.Path):
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "journal"))
        from journal import Journal  # noqa
        self._jnl = Journal(journal_path)

    def retrieve(self, family: str, bottleneck: str = "unknown",
                 k: int = 3) -> dict:
        """Return {'success': [...], 'failure': [...]} short strings for the prompt."""
        eps = [e for e in self._jnl.iter()
               if e.get("kind") == "episode" and _family_of(e) == family]
        eps.sort(key=lambda e: e.get("reward", 0), reverse=True)
        success = [_episode_gist(e) for e in eps if e.get("reward", 0) > 0.3][:k]
        failure = [_episode_gist(e) for e in reversed(eps) if e.get("reward", 0) < 0][:k]
        return {"success": success, "failure": failure}


def _family_of(ep: dict) -> str:
    # task_id prefix maps to family in this task set (ew/rd/ly/pd); fall back to notes
    tid = ep.get("task_id", "")
    return {"ew": "elementwise", "rd": "reduction", "ly": "layout",
            "pd": "perf_diag"}.get(tid[:2], ep.get("family", "unknown"))


def _episode_gist(ep: dict) -> str:
    return (f"{ep.get('task_id')}: strategy={ep.get('strategy','?')} "
            f"prompt={ep.get('prompt_id','?')} reward={ep.get('reward',0):.2f}")


# --------------------------------------------------------------------------- semantic
@dataclass
class Experience:
    context: str          # "reduction|*|memory_bound"
    lesson: str           # human-readable
    support: int = 1      # how many episodes back this


class SemanticMemory:
    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        self.items: list[Experience] = []
        if self.path.exists():
            self.items = [Experience(**d) for d in json.loads(self.path.read_text())]

    def add(self, context: str, lesson: str) -> None:
        for e in self.items:
            if e.context == context and e.lesson == lesson:
                e.support += 1
                self.save()
                return
        self.items.append(Experience(context=context, lesson=lesson))
        self.save()

    def for_context(self, context_key: str) -> list[str]:
        fam = context_key.split("|")[0]
        return [e.lesson for e in self.items if e.context.split("|")[0] == fam]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(e) for e in self.items], indent=2,
                                        ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- procedural
@dataclass
class Rule:
    id: str
    condition: dict                 # {"family":..., "bottleneck":..., "dtype":...}
    action: dict                    # {"strategy":..., "prompt_id":..., "detail":...}
    times_used: int = 0
    correct_count: int = 0
    q_speedup: float = 0.0          # EMA of speedup (or reward proxy)
    q_reward: float = 0.0           # EMA of shaped reward
    last_reward: float = 0.0

    def success_rate(self) -> float:
        return self.correct_count / self.times_used if self.times_used else 0.0

    def confidence(self) -> float:
        """Wilson-ish lower bound so a 1/1 rule isn't as trusted as 8/10."""
        n, p = self.times_used, self.success_rate()
        if n == 0:
            return 0.0
        z = 1.96
        denom = 1 + z * z / n
        centre = p + z * z / (2 * n)
        margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
        return max(0.0, (centre - margin) / denom)


class ProceduralRules:
    """A rule bank with Q-value updates. Serialises to rules.json (shippable)."""

    def __init__(self, path: str | pathlib.Path, alpha: float = 0.3):
        self.path = pathlib.Path(path)
        self.alpha = alpha
        self.rules: dict[str, Rule] = {}
        if self.path.exists():
            for d in json.loads(self.path.read_text()):
                self.rules[d["id"]] = Rule(**d)

    @staticmethod
    def make_id(context_key: str, arm_key: str) -> str:
        return f"{context_key}=>{arm_key}"

    def update(self, context: dict, action: dict, reward: float, correct: bool,
               speedup: float = 0.0) -> Rule:
        ctx_key = f"{context.get('family')}|{context.get('dtype','*')}|{context.get('bottleneck','unknown')}"
        arm_key = f"{action.get('strategy')}::{action.get('prompt_id')}"
        rid = self.make_id(ctx_key, arm_key)
        r = self.rules.get(rid)
        if r is None:
            r = Rule(id=rid, condition=dict(context), action=dict(action))
            self.rules[rid] = r
        r.times_used += 1
        r.correct_count += 1 if correct else 0
        r.q_reward = (1 - self.alpha) * r.q_reward + self.alpha * reward
        if speedup:
            r.q_speedup = (1 - self.alpha) * r.q_speedup + self.alpha * speedup
        r.last_reward = reward
        self.save()
        return r

    def high_confidence(self, min_uses: int = 3, min_conf: float = 0.6,
                        min_reward: float = 0.2) -> list[Rule]:
        """Rules trustworthy enough to distil into the shipped skill."""
        return sorted(
            [r for r in self.rules.values()
             if r.times_used >= min_uses and r.confidence() >= min_conf
             and r.q_reward >= min_reward],
            key=lambda r: (r.confidence(), r.q_reward), reverse=True)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(r) for r in self.rules.values()], indent=2, ensure_ascii=False),
            encoding="utf-8")


if __name__ == "__main__":
    import tempfile
    d = pathlib.Path(tempfile.mkdtemp())
    pr = ProceduralRules(d / "rules.json")
    ctx = {"family": "reduction", "dtype": "float16", "bottleneck": "memory_bound"}
    act = {"strategy": "upcast_accumulate", "prompt_id": "roofline_guided"}
    for correct in [True, True, True, False, True]:
        pr.update(ctx, act, reward=0.6 if correct else -0.2, correct=correct, speedup=1.3)
    hc = pr.high_confidence(min_uses=3, min_conf=0.3)
    assert hc and hc[0].action["strategy"] == "upcast_accumulate", hc
    sm = SemanticMemory(d / "sem.json")
    sm.add("reduction|*|memory_bound", "upcast to fp32 before accumulation")
    sm.add("reduction|*|memory_bound", "upcast to fp32 before accumulation")
    assert sm.items[0].support == 2
    print(f"memory self-test OK — rule conf={hc[0].confidence():.2f} "
          f"success={hc[0].success_rate():.2f}, semantic support={sm.items[0].support}")
