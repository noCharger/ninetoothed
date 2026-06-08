"""
skill_eval — KernelSwift + Ascend-slide inspired evaluation utilities for NineToothed.

Modules:
    robust_bench         - outlier-removed timing + reward-hacking bandwidth check
    reward_hacking_guard - AST static + dynamic + ncu stub
    failure_classifier   - classify failures as code_error | guidance_error | unknown
                           (from Ascend slide: Code Repair vs Prompt Repair distinction)
"""
from .robust_bench import robust_benchmark, compare_robust, BenchResult, RewardHackingError
from .reward_hacking_guard import full_guard, static_analysis, dynamic_analysis, HackingReport
from .failure_classifier import (
    classify, classify_batch, summary as failure_summary,
    FailureClassification, FailureType,
)
