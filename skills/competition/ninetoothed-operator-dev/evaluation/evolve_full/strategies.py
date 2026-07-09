#!/usr/bin/env python3
"""
strategies.py — version A's discrete action space and prompt-template pool, plus the
context extractor that keys the bandit.

Version A treats the third-party coding model as a proposal generator and learns, per
context, WHICH optimisation strategy to pursue and WHICH prompt framing elicits the
best kernel. This module defines those two discrete sets and the context that
conditions the choice; bandit.py learns the mapping; loop.py wires them to claude -p.

Nothing here calls a model — it's pure data + prompt assembly, so it unit-tests offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# --------------------------------------------------------------------------- actions
# Optimisation strategies the agent can be steered toward. Kept operator-family aware:
# not every strategy makes sense for every family, so STRATEGIES_BY_FAMILY gates them.
STRATEGIES = (
    "tiling",              # tune block/tile sizes
    "fusion",              # fuse elementwise/residual into the kernel
    "vectorize",           # widen loads/stores (vectorised access)
    "reduce_registers",    # lower register pressure to raise occupancy
    "pipeline",            # add pipeline stages / software prefetch
    "layout_change",       # change tile layout for coalescing
    "upcast_accumulate",   # accumulate in fp32 to fix fp16 precision
    "num_warps_tune",      # sweep num_warps / num_stages
)

STRATEGIES_BY_FAMILY = {
    "elementwise": ("vectorize", "fusion", "tiling", "num_warps_tune"),
    "reduction":   ("upcast_accumulate", "tiling", "reduce_registers", "num_warps_tune", "pipeline"),
    "layout":      ("layout_change", "vectorize", "tiling", "num_warps_tune"),
    "perf_diag":   ("tiling", "num_warps_tune", "reduce_registers", "pipeline", "vectorize"),
}

# --------------------------------------------------------------------------- prompts
# Prompt framings. Each is a way of asking the SAME model to produce/repair a kernel;
# the bandit learns which framing pays off for which context.
PROMPT_TEMPLATES = {
    "diagnose_first": (
        "Before writing code, diagnose the operator: identify its family, the memory "
        "vs compute bound regime, and the one optimisation most likely to help "
        "({strategy}). Then implement and verify."
    ),
    "local_patch": (
        "Make the smallest possible change to achieve {strategy}. Do not refactor "
        "unrelated code; touch only the kernel body."
    ),
    "roofline_guided": (
        "Use the Roofline model: estimate arithmetic intensity, place the kernel "
        "relative to the ridge point, and justify why {strategy} moves it toward the "
        "roof. Then implement."
    ),
    "with_success_examples": (
        "Here are past successful optimisations for similar operators:\n{success_memories}\n"
        "Apply the same idea ({strategy}) to this task, then verify."
    ),
    "with_failure_examples": (
        "Avoid these past mistakes on similar operators:\n{failure_memories}\n"
        "Implement using {strategy}, steering clear of the above."
    ),
}
PROMPT_IDS = tuple(PROMPT_TEMPLATES.keys())


@dataclass
class Context:
    """The bandit key. Coarse on purpose so arms get enough samples to learn."""
    family: str
    dtype: str = "float16"
    bottleneck: str = "unknown"   # "memory_bound" | "compute_bound" | "unknown"

    def key(self) -> str:
        return f"{self.family}|{self.dtype}|{self.bottleneck}"


@dataclass
class Arm:
    strategy: str
    prompt_id: str

    def key(self) -> str:
        return f"{self.strategy}::{self.prompt_id}"


def arms_for(context: Context) -> list[Arm]:
    """Legal (strategy, prompt) arms for a context."""
    strategies = STRATEGIES_BY_FAMILY.get(context.family, STRATEGIES)
    return [Arm(s, p) for s in strategies for p in PROMPT_IDS]


def extract_context(task_meta: dict, bench_verdict: Optional[str] = None,
                    dtype: Optional[str] = None) -> Context:
    """Build the bandit context from a task and (optionally) a prior bench verdict."""
    family = task_meta.get("category", "unknown")
    dt = dtype or (task_meta.get("dtypes", ["float16"])[0] if task_meta.get("dtypes") else "float16")
    bottleneck = "unknown"
    if bench_verdict:
        v = bench_verdict.lower()
        if "memory" in v:
            bottleneck = "memory_bound"
        elif "compute" in v:
            bottleneck = "compute_bound"
    return Context(family=family, dtype=dt, bottleneck=bottleneck)


def build_prompt(task_meta: dict, arm: Arm, memories: Optional[dict] = None,
                 with_skill: bool = True) -> str:
    """Assemble the episode prompt: task + skill pointer + strategy/prompt framing."""
    memories = memories or {}
    framing = PROMPT_TEMPLATES[arm.prompt_id].format(
        strategy=arm.strategy,
        success_memories=_fmt_mem(memories.get("success", [])),
        failure_memories=_fmt_mem(memories.get("failure", [])),
    )
    parts = []
    if with_skill:
        parts.append("A NineToothed operator-dev skill is available at ./skill/. Read "
                     "./skill/SKILL.md and the relevant ./skill/references/*.md first.")
    parts.append(f"Task ({task_meta.get('category','?')}): {task_meta.get('prompt','').strip()}")
    parts.append(f"Optimisation guidance: {framing}")
    parts.append("Produce kernel.py, wrapper.py, test_correctness.py, and (if perf "
                 "matters) bench.csv with a bound verdict. No network access.")
    return "\n\n".join(parts)


def _fmt_mem(items: list) -> str:
    if not items:
        return "  (none on record)"
    return "\n".join(f"  - {s}" for s in items[:3])


if __name__ == "__main__":
    ctx = extract_context({"category": "reduction", "dtypes": ["float16"]}, "memory-bound")
    assert ctx.key() == "reduction|float16|memory_bound", ctx.key()
    arms = arms_for(ctx)
    assert all(a.strategy in STRATEGIES_BY_FAMILY["reduction"] for a in arms)
    assert len(arms) == len(STRATEGIES_BY_FAMILY["reduction"]) * len(PROMPT_IDS)
    p = build_prompt({"category": "reduction", "prompt": "impl rms_norm"}, arms[0],
                     {"success": ["upcast to fp32 fixed NaN on rms_norm"]})
    assert "rms_norm" in p and "skill" in p
    print(f"strategies self-test OK ({len(arms)} arms for {ctx.key()})")
