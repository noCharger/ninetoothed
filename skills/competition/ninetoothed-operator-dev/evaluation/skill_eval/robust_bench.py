"""
robust_bench.py — KernelSwift-inspired robust benchmark evaluator.

Borrows three ideas from KernelSwift Agent Infra (Shanghai AI Lab):
  1. Fixed computation graph during measurement.
  2. Multiple measurements with outlier removal.
  3. Reward-hacking detection (flag kernels that look anomalously fast).

Reward hacking in the context of NineToothed skill evaluation: a kernel that
appears faster than the theoretical roofline bound, or that passes correctness
by being so fast it skips actual computation (e.g. returns a zero tensor).

Usage:
    from skill_eval.robust_bench import robust_benchmark, RewardHackingError

    result = robust_benchmark(
        kernel_fn=lambda: my_op(x),
        reference_fn=lambda: torch.softmax(x, dim=-1),
        bytes_moved=x.numel() * 2 * 2,   # read + write, fp16
        gpu="H100",
    )
    print(result)
    # {'mean_ms': ..., 'std_ms': ..., 'outliers_removed': 2,
    #  'GB_s': ..., 'verdict': 'memory-bound', 'hacking_check': 'clean'}
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, Optional

# Ridge points (FLOP/byte) for common GPUs.
RIDGE_POINTS = {
    "H100": 295.0,
    "A100": 156.0,
    "L4": 121.0,
    "V100": 140.0,
}

# If a kernel is faster than PEAK_GBPS * this fraction, flag as suspicious.
# Theoretical peak bandwidths (GB/s):
PEAK_BW_GBPS = {
    "H100": 3350.0,
    "A100": 2000.0,
    "L4": 300.0,
    "V100": 900.0,
}

# Fraction of theoretical peak beyond which a result is flagged as suspicious.
# A kernel claiming > 95% of HBM bandwidth is almost certainly incorrect.
SUSPICIOUS_BW_FRACTION = 0.95


class RewardHackingError(Exception):
    """Raised when a kernel's measured performance exceeds physical limits."""


@dataclass
class BenchResult:
    mean_ms: float
    std_ms: float
    median_ms: float
    outliers_removed: int
    raw_samples: int
    GB_s: Optional[float]
    TFLOPS: Optional[float]
    verdict: Optional[str]          # 'compute-bound' | 'memory-bound' | None
    hacking_check: str              # 'clean' | 'suspicious:<reason>' | 'unchecked'
    device: str

    def __str__(self) -> str:
        parts = [
            f"mean={self.mean_ms:.3f}ms",
            f"std={self.std_ms:.3f}ms",
            f"outliers={self.outliers_removed}/{self.raw_samples}",
        ]
        if self.GB_s:
            parts.append(f"BW={self.GB_s:.1f}GB/s")
        if self.TFLOPS:
            parts.append(f"{self.TFLOPS:.2f}TFLOPS")
        if self.verdict:
            parts.append(self.verdict)
        parts.append(f"hacking={self.hacking_check}")
        return "  ".join(parts)


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _cuda_time_ms(fn: Callable, iters: int) -> list[float]:
    """Time fn using CUDA events. Returns list of per-iter ms."""
    import torch
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _cpu_time_ms(fn: Callable, iters: int) -> list[float]:
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def _remove_outliers_iqr(samples: list[float], k: float = 1.5) -> tuple[list[float], int]:
    """Remove IQR-based outliers (KernelSwift technique).

    Uses the Tukey fence: remove values outside [Q1 - k*IQR, Q3 + k*IQR].
    Returns (clean_samples, n_removed).
    """
    if len(samples) < 4:
        return samples, 0
    sorted_s = sorted(samples)
    n = len(sorted_s)
    q1 = sorted_s[n // 4]
    q3 = sorted_s[(3 * n) // 4]
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    clean = [s for s in samples if lo <= s <= hi]
    return clean, n - len(clean)


def _check_reward_hacking(
    mean_ms: float,
    bytes_moved: int,
    gpu: str,
    reference_fn: Optional[Callable],
) -> str:
    """
    KernelSwift reward-hacking guard: detect 'anomalously fast' kernels.

    Checks:
    1. Bandwidth sanity: claimed GB/s must be < SUSPICIOUS_BW_FRACTION of peak.
    2. (If reference_fn given) output sanity: we don't re-run here — that's
       handled by the separate correctness matrix. This check is about perf.

    Returns 'clean' or 'suspicious:<reason>'.
    """
    if bytes_moved <= 0 or mean_ms <= 0:
        return "unchecked"

    peak_bw = PEAK_BW_GBPS.get(gpu)
    if peak_bw is None:
        return "unchecked"

    claimed_bw = bytes_moved / (mean_ms * 1e-3) / 1e9
    limit = peak_bw * SUSPICIOUS_BW_FRACTION
    if claimed_bw > limit:
        return (
            f"suspicious:bandwidth {claimed_bw:.1f}GB/s > "
            f"{SUSPICIOUS_BW_FRACTION*100:.0f}% of peak {peak_bw:.0f}GB/s — "
            f"likely skipping computation or timing error"
        )
    return "clean"


def robust_benchmark(
    kernel_fn: Callable[[], object],
    bytes_moved: int = 0,
    flops: int = 0,
    gpu: str = "H100",
    warmup: int = 25,
    iters: int = 100,
    outlier_k: float = 1.5,
    reference_fn: Optional[Callable[[], object]] = None,
    raise_on_hacking: bool = False,
) -> BenchResult:
    """
    Robust benchmark with KernelSwift-inspired noise suppression.

    Args:
        kernel_fn       : callable that runs the NineToothed kernel (no args).
        bytes_moved     : total bytes read+written (for bandwidth calculation).
        flops           : total FLOPs (for TFLOPS calculation).
        gpu             : GPU model string — must match RIDGE_POINTS keys.
        warmup          : warm-up iterations (not timed).
        iters           : timed iterations.
        outlier_k       : IQR fence multiplier for outlier removal.
        reference_fn    : optional PyTorch reference (used for hacking check
                          metadata; actual correctness tested separately).
        raise_on_hacking: if True, raise RewardHackingError on suspicious result.

    Returns:
        BenchResult dataclass.
    """
    use_cuda = _has_cuda()

    # ---- fixed computation graph (warm-up locks in the kernel) ----
    for _ in range(warmup):
        kernel_fn()
    if use_cuda:
        import torch
        torch.cuda.synchronize()

    # ---- repeated measurement ----
    if use_cuda:
        raw = _cuda_time_ms(kernel_fn, iters)
    else:
        print("[robust_bench] WARNING: CUDA unavailable, using wall-clock timing.")
        raw = _cpu_time_ms(kernel_fn, iters)

    # ---- outlier removal ----
    clean, n_removed = _remove_outliers_iqr(raw, k=outlier_k)
    if not clean:
        clean = raw  # fallback: don't remove everything

    mean_ms = statistics.fmean(clean)
    std_ms = statistics.pstdev(clean) if len(clean) > 1 else 0.0
    median_ms = statistics.median(clean)

    # ---- Throughput ----
    sec = mean_ms * 1e-3
    gb_s = (bytes_moved / sec / 1e9) if bytes_moved > 0 else None
    tflops = (flops / sec / 1e12) if flops > 0 else None

    # ---- Roofline verdict ----
    verdict = None
    if bytes_moved > 0 and flops > 0:
        ridge = RIDGE_POINTS.get(gpu)
        if ridge is not None:
            ai = flops / bytes_moved
            verdict = "compute-bound" if ai >= ridge else "memory-bound"

    # ---- Reward Hacking check ----
    hacking = _check_reward_hacking(mean_ms, bytes_moved, gpu, reference_fn)
    if raise_on_hacking and hacking.startswith("suspicious"):
        raise RewardHackingError(hacking)

    device = f"cuda:{0}" if use_cuda else "cpu"

    return BenchResult(
        mean_ms=mean_ms,
        std_ms=std_ms,
        median_ms=median_ms,
        outliers_removed=n_removed,
        raw_samples=len(raw),
        GB_s=gb_s,
        TFLOPS=tflops,
        verdict=verdict,
        hacking_check=hacking,
        device=device,
    )


def compare_robust(
    candidates: dict[str, Callable[[], object]],
    bytes_moved: int = 0,
    flops: int = 0,
    gpu: str = "H100",
    **bench_kw,
) -> dict[str, BenchResult]:
    """Benchmark several callables with full outlier-removal and hacking checks."""
    results = {}
    for name, fn in candidates.items():
        r = robust_benchmark(fn, bytes_moved=bytes_moved, flops=flops, gpu=gpu, **bench_kw)
        results[name] = r
    # Add relative speedup vs slowest.
    slowest = max(r.mean_ms for r in results.values())
    for name, r in results.items():
        r.__dict__["speedup_vs_slowest"] = slowest / r.mean_ms if r.mean_ms else float("inf")
    return results
