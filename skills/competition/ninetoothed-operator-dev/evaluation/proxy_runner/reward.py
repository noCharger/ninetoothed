#!/usr/bin/env python3
"""
reward.py — episode-level scalar reward for the self-evolving loop.

Two distinct signals live in this project; don't conflate them:

  * rubric total (0–10)   — the *fitness* used for generation selection and for
    the A/B comparison. Same as the hidden-task rubric. Computed in rubric_scorer.

  * shaped reward (float) — the *learning signal* that drives evolve_full's bandit
    over (strategy, prompt) choices. It rewards latency wins and roofline gains but
    charges for LLM tokens, GPU seconds, and compile attempts, so the loop learns
    optimisation ROI, not raw speedup. This module computes it.

Formula (matches the plan's §1 design):

    r =  1{correct} · log(L_before / L_after)      # latency ratio, 0 if not improved
       + beta  · d_roofline                        # roofline efficiency delta
       - lam_llm     · C_llm                       # normalised token cost
       - lam_gpu     · C_gpu                        # normalised gpu-seconds
       - lam_compile · C_compile                    # normalised compile attempts

    hard floors:  incorrect      -> R_INCORRECT (default -4.0)
                  does not compile -> R_NOCOMPILE (default -2.0)

The cost terms are normalised by reference scales so the coefficients are O(1).
All knobs are dataclass fields so the loop can sweep them and the report can state
exactly what objective was optimised.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RewardConfig:
    beta: float = 0.5           # weight on roofline-efficiency delta
    lam_llm: float = 0.10       # weight on LLM token cost
    lam_gpu: float = 0.05       # weight on GPU seconds
    lam_compile: float = 0.05   # weight on compile attempts
    # normalisation scales (so cost terms are ~O(1) at typical magnitudes)
    tok_scale: float = 10_000.0     # tokens
    gpu_scale: float = 60.0         # seconds
    compile_scale: float = 4.0      # attempts
    # hard floors
    r_incorrect: float = -4.0
    r_nocompile: float = -2.0


@dataclass
class EpisodeOutcome:
    compiled: bool
    correct: bool
    latency_before_ms: float = 0.0   # best prior latency (0 = unknown/baseline)
    latency_after_ms: float = 0.0    # this episode's latency (0 = not measured)
    roofline_before: float = 0.0     # efficiency in [0,1]
    roofline_after: float = 0.0
    tokens: float = 0.0              # tokens_in + tokens_out
    gpu_seconds: float = 0.0
    compile_attempts: int = 1


def compute_reward(outcome: EpisodeOutcome, cfg: RewardConfig | None = None) -> float:
    cfg = cfg or RewardConfig()
    if not outcome.compiled:
        return cfg.r_nocompile
    if not outcome.correct:
        return cfg.r_incorrect

    # latency term: log ratio, only credited when improved and both measured
    lat = 0.0
    if outcome.latency_before_ms > 0 and outcome.latency_after_ms > 0:
        ratio = outcome.latency_before_ms / outcome.latency_after_ms
        lat = math.log(ratio) if ratio > 1.0 else 0.0

    roof = cfg.beta * (outcome.roofline_after - outcome.roofline_before)

    c_llm = cfg.lam_llm * (outcome.tokens / cfg.tok_scale)
    c_gpu = cfg.lam_gpu * (outcome.gpu_seconds / cfg.gpu_scale)
    c_cmp = cfg.lam_compile * (outcome.compile_attempts / cfg.compile_scale)

    return lat + roof - c_llm - c_gpu - c_cmp


def reward_from_rubric(total: int, tokens: float, gpu_seconds: float,
                       compile_attempts: int = 1, cfg: RewardConfig | None = None) -> float:
    """
    Fallback shaped reward when latency/roofline aren't available (e.g. correctness-
    only tasks or diagnosis tasks): use the rubric total (0..10 -> 0..1) as the value
    term, minus the same cost charges. Keeps the bandit signal aligned with fitness.
    """
    cfg = cfg or RewardConfig()
    value = total / 10.0
    c_llm = cfg.lam_llm * (tokens / cfg.tok_scale)
    c_gpu = cfg.lam_gpu * (gpu_seconds / cfg.gpu_scale)
    c_cmp = cfg.lam_compile * (compile_attempts / cfg.compile_scale)
    return value - c_llm - c_gpu - c_cmp


if __name__ == "__main__":
    # tiny self-demo
    demo = EpisodeOutcome(compiled=True, correct=True, latency_before_ms=2.0,
                          latency_after_ms=1.0, roofline_before=0.4, roofline_after=0.6,
                          tokens=8000, gpu_seconds=30, compile_attempts=2)
    print("reward:", round(compute_reward(demo), 4))
    print("nocompile:", compute_reward(EpisodeOutcome(False, False)))
    print("incorrect:", compute_reward(EpisodeOutcome(True, False)))
    print("from_rubric(8):", round(reward_from_rubric(8, 8000, 30, 2), 4))
