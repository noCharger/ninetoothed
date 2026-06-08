"""
failure_classifier.py — classify a NineToothed kernel failure as
code_error or guidance_error, following the Ascend slide's insight:

  Code Repair   → fix kernel.py        (agent made a known mistake)
  Prompt Repair → fix references/*.md  (skill text has a gap or ambiguity)

The distinction matters for Stage 3 MOO edit targeting:
  code_error    → prune/substitute target: kernel.py, wrapper.py
  guidance_error → prune/substitute target: references/<family>.md or SKILL.md

Classification is heuristic — deliberately simple and auditable.
It is NOT an LLM call; it is a deterministic rule-based classifier so
the Stage 3 loop can attribute failures without spending inference budget.

Heuristics (applied in order, first match wins):
  1. Error text matches a known pattern in common-errors.md → code_error
  2. debug_arrangement reports OOB → code_error (arrangement mistake)
  3. Same wrong symptom appears ≥ REPEAT_THRESHOLD times across
     different shapes/dtypes → guidance_error (skill didn't prevent it)
  4. Error text contains an API signature mismatch (unexpected keyword,
     wrong ndim, wrong number of args) → guidance_error (reference
     example may show wrong call signature)
  5. Fallback → unknown (escalate to human)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FailureType(str, Enum):
    CODE_ERROR     = "code_error"      # fix kernel.py / wrapper.py
    GUIDANCE_ERROR = "guidance_error"  # fix references/<family>.md or SKILL.md
    UNKNOWN        = "unknown"         # escalate


# Minimum times a symptom must repeat across distinct (shape, dtype) pairs
# before we consider it a guidance gap rather than a one-off code mistake.
REPEAT_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Known-error pattern bank  (mirrors common-errors.md)
# If a failure message matches one of these, it is a CODE error — the agent
# failed to follow existing, clearly documented guidance.
# ---------------------------------------------------------------------------
_CODE_ERROR_PATTERNS: list[tuple[str, str]] = [
    # (regex_pattern, description)
    (r"TypeError.*not iterable",          "arrangement returned bare Tensor, not tuple (error #1)"),
    (r"F841.*output.*assigned.*never used", "missing # noqa: F841 on output assignment (error #2)"),
    (r"RecursionError",                    "output over-arranged to mirror input blocks (error #6)"),
    (r"unsqueeze.*eval",                   "unsqueeze in arrangement eval failure (error #7)"),
    (r"rank.*mismatch|ndim.*mismatch|shape.*mismatch.*Tensor\(\d\)",
                                           "multi-dim tensor passed to wrong-rank kernel (error #8)"),
    (r"BLOCK_SIZE.*constexpr.*block_size", "constexpr vs block_size confusion (error #9)"),
    (r"NaN.*output|nan.*values",           "fp16 accumulation without upcast (error #5)"),
    (r"allclose.*failed|assert.*close.*fail",
                                           "correctness failure — check dtype tolerance and upcast"),
    (r"other.*float.*inf|mask.*fill",      "missing other=float('-inf') on softmax/max input (error #4)"),
]

# ---------------------------------------------------------------------------
# Guidance-error API-mismatch patterns  (skill reference shows wrong idiom)
# ---------------------------------------------------------------------------
_GUIDANCE_ERROR_PATTERNS: list[tuple[str, str]] = [
    (r"offsets.*takes \d+ positional argument|takes \d+ positional argument.*offsets",
     "references use offsets(dim) — but 0.25.0 offsets() takes no args; call offsets() with no arguments"),
    (r"takes \d+ positional argument.*\d+ given",
     "reference example has wrong argument count — fix references/"),
    (r"has no attribute.*tile|has no attribute.*expand|has no attribute.*squeeze",
     "meta-op not available on this Tensor subtype — reference example may be wrong version"),
    (r"TypeError.*block_size\(\).*constexpr",
     "block_size() vs Symbol(constexpr=True) confused in reference example"),
]


@dataclass
class FailureClassification:
    failure_type: FailureType
    evidence: str                          # what triggered the rule
    repair_target: str                     # "kernel.py" | "references/<X>.md" | "SKILL.md"
    repair_hint: str                       # one-line suggestion for what to change
    repeat_count: int = 0                  # how many times this symptom was seen
    matched_rule: str = ""                 # which rule fired

    def __str__(self) -> str:
        return (
            f"[{self.failure_type.value.upper()}] {self.evidence}\n"
            f"  repair_target : {self.repair_target}\n"
            f"  hint          : {self.repair_hint}\n"
            f"  rule          : {self.matched_rule}"
        )


def classify(
    error_text: str,
    oob_count: int = 0,
    symptom_history: Optional[list[str]] = None,
    family: str = "unknown",
) -> FailureClassification:
    """
    Classify a kernel failure.

    Args:
        error_text       : Full error / assertion message from pytest or kernel run.
        oob_count        : Output of debug_arrangement (0 = no OOB).
        symptom_history  : List of error messages from PREVIOUS iterations of the
                           SAME operator (same session). Used for repeat detection.
        family           : Operator family ("elementwise" | "reduction" | "layout" | "perf-diag").

    Returns:
        FailureClassification
    """
    history = symptom_history or []

    # ---- Rule 1: OOB from debug_arrangement → always code error ----
    if oob_count > 0:
        return FailureClassification(
            failure_type=FailureType.CODE_ERROR,
            evidence=f"debug_arrangement reported {oob_count} OOB accesses",
            repair_target="kernel.py (arrangement)",
            repair_hint=(
                "Fix the tile hierarchy: check squeeze/expand dims, "
                "ensure output arrangement is independent of input block structure."
            ),
            matched_rule="oob_from_debug_arrangement",
        )

    # ---- Rule 2: known common-errors.md pattern → code error ----
    for pattern, description in _CODE_ERROR_PATTERNS:
        if re.search(pattern, error_text, re.IGNORECASE):
            return FailureClassification(
                failure_type=FailureType.CODE_ERROR,
                evidence=f"matched known error pattern: {description}",
                repair_target="kernel.py",
                repair_hint=(
                    f"Apply the fix documented in references/common-errors.md: {description}"
                ),
                matched_rule=f"known_pattern:{pattern[:40]}",
            )

    # ---- Rule 3: guidance/API mismatch → guidance error ----
    for pattern, description in _GUIDANCE_ERROR_PATTERNS:
        if re.search(pattern, error_text, re.IGNORECASE):
            return FailureClassification(
                failure_type=FailureType.GUIDANCE_ERROR,
                evidence=f"API mismatch pattern: {description}",
                repair_target=f"references/{family}.md",
                repair_hint=(
                    f"The reference example in references/{family}.md likely shows "
                    f"an incorrect API call. Fix: {description}"
                ),
                matched_rule=f"api_mismatch:{pattern[:40]}",
            )

    # ---- Rule 4: repeat symptom → guidance error ----
    # If the SAME class of error (first 80 chars) has appeared ≥ REPEAT_THRESHOLD
    # times across different iterations, the skill text didn't prevent it.
    symptom_key = error_text.strip()[:80]
    repeat_count = sum(1 for h in history if h.strip()[:80] == symptom_key)
    if repeat_count >= REPEAT_THRESHOLD:
        return FailureClassification(
            failure_type=FailureType.GUIDANCE_ERROR,
            evidence=(
                f"Same symptom appeared {repeat_count + 1} times across iterations "
                f"(threshold={REPEAT_THRESHOLD}) — skill text did not prevent it"
            ),
            repair_target=f"references/{family}.md",
            repair_hint=(
                "Add a more prominent warning or concrete example to "
                f"references/{family}.md covering this failure pattern. "
                "Consider making it the first pitfall in the 'Pitfalls' section."
            ),
            repeat_count=repeat_count + 1,
            matched_rule="repeat_symptom",
        )

    # ---- Rule 5: fallback ----
    return FailureClassification(
        failure_type=FailureType.UNKNOWN,
        evidence=f"no rule matched; error: {error_text[:120]}",
        repair_target="human review",
        repair_hint=(
            "Neither a known code pattern nor a repeated symptom. "
            "Add to common-errors.md after root-cause analysis."
        ),
        matched_rule="fallback",
    )


def classify_batch(
    failures: list[dict],
    family: str = "unknown",
) -> list[FailureClassification]:
    """
    Classify a list of failure dicts from run_correctness_matrix output.

    Each dict should have keys: 'error_text', 'oob_count' (optional).
    Builds the symptom_history incrementally so repeat detection works.
    """
    results = []
    history: list[str] = []
    for f in failures:
        clf = classify(
            error_text=f.get("error_text", ""),
            oob_count=f.get("oob_count", 0),
            symptom_history=history,
            family=family,
        )
        results.append(clf)
        history.append(f.get("error_text", ""))
    return results


def summary(classifications: list[FailureClassification]) -> dict:
    """Return counts by failure type — useful for the A/B comparison table."""
    from collections import Counter
    counts = Counter(c.failure_type.value for c in classifications)
    repair_targets = Counter(c.repair_target for c in classifications)
    return {
        "total": len(classifications),
        "code_errors": counts.get("code_error", 0),
        "guidance_errors": counts.get("guidance_error", 0),
        "unknown": counts.get("unknown", 0),
        "top_repair_targets": repair_targets.most_common(3),
    }
