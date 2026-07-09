#!/usr/bin/env python3
"""
bandit.py — contextual Thompson-sampling bandit over (strategy, prompt) arms.

Version A's learnable controller. Reward is continuous (the shaped reward from
reward.py), so each arm keeps a Gaussian posterior over its mean reward and we
Thompson-sample by drawing from that posterior. Selection is per-context: the same
arm has an independent posterior in each context key, so the loop learns, e.g., that
`upcast_accumulate::roofline_guided` is best for `reduction|fp16|memory_bound`.

This directly realises the RL criterion the plan promises to demonstrate: as an arm
accrues higher reward in a context, its posterior mean rises and its sampling
probability rises with it — "past reward ⇒ future action distribution shifts".

Deterministic, seedable RNG (no numpy needed) so runs are reproducible and the report
can replay the posterior evolution. State serialises to JSON for the memory layer.
"""
from __future__ import annotations

import json
import math
import pathlib
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ArmStat:
    n: int = 0
    mean: float = 0.0            # running mean reward
    m2: float = 0.0              # sum of squared deviations (Welford) for variance
    last_reward: float = 0.0

    def update(self, reward: float) -> None:
        self.n += 1
        delta = reward - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (reward - self.mean)
        self.last_reward = reward

    def variance(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 1.0

    def std(self) -> float:
        return math.sqrt(max(1e-9, self.variance()))


class ContextualBandit:
    """Per-context Gaussian Thompson sampling with an optimistic prior."""

    def __init__(self, prior_mean: float = 0.3, prior_std: float = 1.0, seed: int = 0):
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self._rng = random.Random(seed)
        # context_key -> arm_key -> ArmStat
        self.stats: dict[str, dict[str, ArmStat]] = {}

    def _arm_stat(self, ctx_key: str, arm_key: str) -> ArmStat:
        return self.stats.setdefault(ctx_key, {}).setdefault(arm_key, ArmStat())

    def _sample_arm_value(self, stat: ArmStat) -> float:
        """Thompson sample: draw the arm's plausible mean reward."""
        if stat.n == 0:
            return self._rng.gauss(self.prior_mean, self.prior_std)
        # posterior std of the mean shrinks as ~std/sqrt(n)
        post_std = stat.std() / math.sqrt(stat.n)
        # blend a little prior width so unexplored-ish arms keep some optimism
        post_std = max(post_std, self.prior_std / (stat.n + 1))
        return self._rng.gauss(stat.mean, post_std)

    def select(self, ctx_key: str, arm_keys: list[str]) -> str:
        """Pick the arm with the highest sampled value (Thompson)."""
        best, best_val = arm_keys[0], -float("inf")
        for ak in arm_keys:
            val = self._sample_arm_value(self._arm_stat(ctx_key, ak))
            if val > best_val:
                best, best_val = ak, val
        return best

    def update(self, ctx_key: str, arm_key: str, reward: float) -> None:
        self._arm_stat(ctx_key, arm_key).update(reward)

    def prob_best(self, ctx_key: str, arm_keys: list[str], target_arm: str,
                  draws: int = 2000) -> float:
        """Estimate P(target_arm is selected) under the current posteriors.
        Used by the report to show the action distribution shifting over time."""
        # use an independent RNG so this read doesn't perturb the training stream
        rng = random.Random(12345)
        wins = 0
        stats = {ak: self._arm_stat(ctx_key, ak) for ak in arm_keys}
        for _ in range(draws):
            best, best_val = None, -float("inf")
            for ak in arm_keys:
                s = stats[ak]
                if s.n == 0:
                    val = rng.gauss(self.prior_mean, self.prior_std)
                else:
                    post_std = max(s.std() / math.sqrt(s.n), self.prior_std / (s.n + 1))
                    val = rng.gauss(s.mean, post_std)
                if val > best_val:
                    best, best_val = ak, val
            if best == target_arm:
                wins += 1
        return wins / draws

    def best_arm(self, ctx_key: str) -> Optional[tuple[str, float, int]]:
        """The current empirically-best arm in a context: (arm_key, mean, n)."""
        arms = self.stats.get(ctx_key)
        if not arms:
            return None
        ak = max(arms, key=lambda k: arms[k].mean)
        return ak, arms[ak].mean, arms[ak].n

    # ---- persistence ----
    def to_dict(self) -> dict:
        return {
            "prior_mean": self.prior_mean, "prior_std": self.prior_std,
            "stats": {ck: {ak: vars(st) for ak, st in arms.items()}
                      for ck, arms in self.stats.items()},
        }

    def save(self, path: str | pathlib.Path) -> None:
        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | pathlib.Path, seed: int = 0) -> "ContextualBandit":
        d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        b = cls(prior_mean=d.get("prior_mean", 0.3), prior_std=d.get("prior_std", 1.0), seed=seed)
        for ck, arms in d.get("stats", {}).items():
            for ak, st in arms.items():
                b.stats.setdefault(ck, {})[ak] = ArmStat(**st)
        return b


if __name__ == "__main__":
    # demonstrate the RL criterion: a truly-better arm's selection prob rises over time.
    b = ContextualBandit(seed=1)
    ctx = "reduction|float16|memory_bound"
    arms = ["upcast::roofline_guided", "tiling::local_patch", "pipeline::diagnose_first"]
    # ground truth: first arm is best
    true_mean = {"upcast::roofline_guided": 0.8, "tiling::local_patch": 0.3,
                 "pipeline::diagnose_first": 0.1}
    rng = random.Random(2)
    p0 = b.prob_best(ctx, arms, "upcast::roofline_guided")
    for _ in range(200):
        a = b.select(ctx, arms)
        r = rng.gauss(true_mean[a], 0.1)
        b.update(ctx, a, r)
    p1 = b.prob_best(ctx, arms, "upcast::roofline_guided")
    print(f"P(best arm) before={p0:.2f} after={p1:.2f}")
    assert p1 > p0 and p1 > 0.8, (p0, p1)
    best = b.best_arm(ctx)
    assert best and best[0] == "upcast::roofline_guided", best
    print(f"bandit self-test OK — converged to {best[0]} (mean={best[1]:.2f}, n={best[2]})")
