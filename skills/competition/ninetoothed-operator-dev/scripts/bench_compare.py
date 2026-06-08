#!/usr/bin/env python3
"""Benchmarking + Roofline helpers for NineToothed operators.

Import these from your own bench file (keeps this script free of any dynamic
import of your kernel):

    from bench_compare import benchmark, roofline, compare
    ms = benchmark(lambda: my_op(x))
    print(roofline(flops=2*M*N*K, bytes_moved=(M*K+K*N+M*N)*2, gpu="H100"))

CUDA timing uses torch.cuda.Event; on CPU it falls back to perf_counter with a
warning so the script still runs in a CUDA-less environment.
"""
from __future__ import annotations

import statistics
import time
from typing import Callable

# Ridge points: arithmetic intensity (FLOP/byte). below ridge => memory-bound.
# Derived from peak FLOPs / peak bandwidth; update for your exact device.
RIDGE_POINTS = {
    "H100": 295.0,   # SXM fp16 ~989 TFLOP/s / ~3.35 TB/s
    "A100": 156.0,   # fp16 ~312 TFLOP/s / ~2.0 TB/s
    "L4": 121.0,     # fp16 ~242 TFLOP/s / ~2.0 TB/s (approx)
}


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def benchmark(fn: Callable[[], object], warmup: int = 25, iters: int = 100) -> dict:
    """Time `fn` and return {'mean_ms','std_ms','min_ms','iters','timer'}."""
    if _has_cuda():
        import torch

        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms
        timer = "cuda.Event"
    else:
        print("[bench_compare] WARNING: CUDA unavailable; using perf_counter (wall).")
        for _ in range(min(warmup, 3)):
            fn()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)
        timer = "perf_counter"

    return {
        "mean_ms": statistics.fmean(times),
        "std_ms": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "iters": len(times),
        "timer": timer,
    }


def throughput(mean_ms: float, *, bytes_moved: int = 0, flops: int = 0) -> dict:
    """Return GB/s and TFLOPS from a mean latency."""
    sec = mean_ms * 1e-3
    out = {}
    if bytes_moved:
        out["GB_s"] = bytes_moved / sec / 1e9
    if flops:
        out["TFLOPS"] = flops / sec / 1e12
    return out


def roofline(*, flops: int, bytes_moved: int, gpu: str = "H100") -> dict:
    """Classify compute- vs memory-bound by arithmetic intensity vs ridge."""
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be > 0")
    ai = flops / bytes_moved
    ridge = RIDGE_POINTS.get(gpu)
    if ridge is None:
        raise ValueError(f"unknown gpu {gpu!r}; known: {sorted(RIDGE_POINTS)}")
    verdict = "compute-bound" if ai >= ridge else "memory-bound"
    return {"arithmetic_intensity": ai, "ridge_point": ridge, "gpu": gpu, "verdict": verdict}


def compare(candidates: dict[str, Callable[[], object]], **bench_kw) -> dict:
    """Benchmark several callables; return name -> result, with speedup vs the
    slowest as a convenience field."""
    results = {name: benchmark(fn, **bench_kw) for name, fn in candidates.items()}
    slowest = max(r["mean_ms"] for r in results.values())
    for r in results.values():
        r["speedup_vs_slowest"] = slowest / r["mean_ms"] if r["mean_ms"] else float("inf")
    return results


if __name__ == "__main__":
    # self-test that runs anywhere (no CUDA / no torch required for roofline)
    print("roofline demo (fp16 GEMM 4096^3 on H100):")
    M = N = K = 4096
    print(roofline(flops=2 * M * N * K, bytes_moved=(M * K + K * N + M * N) * 2, gpu="H100"))
    print("benchmark demo (CPU sleep 1ms):")
    print(benchmark(lambda: time.sleep(0.001), warmup=2, iters=5))
