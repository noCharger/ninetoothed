"""
Correctness + perf/diag tests for the softmax regression example.

Demonstrates:
* The bad kernel fails on large-value fp16 inputs (detects the regression).
* The fixed kernel passes the full correctness matrix.
* Benchmark comparison: fixed vs PyTorch baseline.
* Generated source inspection output captured in the trace log.

This is the 'after' picture of a performance/diagnosis task.
"""

import itertools
import subprocess
import sys

import pytest
import torch

from .wrapper import softmax_bad, softmax_fixed

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE = "cuda"
TOL_FIXED = {
    torch.float32: dict(atol=1e-5, rtol=1e-5),
    torch.float16: dict(atol=1e-3, rtol=1e-3),
}
SHAPES = [(8, 128), (16, 511), (32, 1024)]
DTYPES = [torch.float32, torch.float16]


def reference(x):
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


# ---------------------------------------------------------------------------
# Regression: document the failure of the bad kernel
# ---------------------------------------------------------------------------
def test_bad_kernel_fails_on_large_fp16():
    """
    The bad kernel should produce NaN or large error on fp16 with large values.
    This test DOCUMENTS the known regression — it is expected to fail or produce
    wrong results, which is the starting point of the diagnostic task.
    """
    x = torch.randn(16, 512, dtype=torch.float16, device=DEVICE) * 10
    got = softmax_bad(x)
    expected = reference(x)
    # We expect this NOT to match; if it matches, the 'bad' kernel is not broken.
    try:
        torch.testing.assert_close(got, expected, atol=1e-2, rtol=1e-2)
        # If we get here, the kernel happened to be OK — not what we expect.
        pytest.xfail("bad kernel unexpectedly passed — regenerate with a more extreme input")
    except AssertionError:
        pass  # expected: bad kernel is wrong, regression confirmed


# ---------------------------------------------------------------------------
# Fixed kernel: must pass the full correctness matrix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape,dtype",
    list(itertools.product(SHAPES, DTYPES)),
)
def test_softmax_fixed(shape, dtype):
    x = torch.randn(*shape, dtype=dtype, device=DEVICE)
    expected = reference(x)
    got = softmax_fixed(x)
    torch.testing.assert_close(got, expected, **TOL_FIXED[dtype])


def test_softmax_large_values_fp16():
    """Large-magnitude input: numerically stable path must not produce NaN."""
    x = torch.randn(16, 512, dtype=torch.float16, device=DEVICE) * 10
    got = softmax_fixed(x)
    assert not torch.isnan(got).any(), "NaN in fixed softmax output"
    assert not torch.isinf(got).any(), "Inf in fixed softmax output"
    # Sums should be close to 1.
    row_sums = got.float().sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Generated-source inspection (diagnostic evidence)
# ---------------------------------------------------------------------------
def test_inspect_generated_source():
    """
    Run inspect_generated_source.py and assert it finds a cached kernel.
    The output is the 'evidence' you would capture in the trace log.
    """
    # Ensure the fixed kernel has been built (trigger compilation).
    x = torch.randn(4, 32, dtype=torch.float16, device=DEVICE)
    softmax_fixed(x)
    torch.cuda.synchronize()

    scripts_dir = (
        __file__.split("examples")[0] + "scripts"
    )
    result = subprocess.run(
        [sys.executable, "inspect_generated_source.py"],
        capture_output=True,
        text=True,
        cwd=scripts_dir,
        shell=False,
    )
    print("\n--- generated source inspection ---")
    print(result.stdout)
    assert "loads=" in result.stdout, "expected inspect output to contain 'loads='"


# ---------------------------------------------------------------------------
# Benchmark: fixed vs PyTorch (required by self_test_tasks.md)
# ---------------------------------------------------------------------------
@pytest.mark.benchmark
def test_benchmark_comparison():
    """
    Benchmark fixed softmax vs torch.softmax on fp16 (M=1024, N=512).
    Records the evidence that the fix does not regress performance.
    """
    import time

    M, N = 1024, 512
    x = torch.randn(M, N, dtype=torch.float16, device=DEVICE)

    def run_nt():
        return softmax_fixed(x)

    def run_pt():
        return torch.softmax(x, dim=-1)

    for _ in range(20):
        run_nt(); run_pt()
    torch.cuda.synchronize()

    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        run_nt()
    torch.cuda.synchronize()
    nt_ms = (time.perf_counter() - t0) / iters * 1e3

    t0 = time.perf_counter()
    for _ in range(iters):
        run_pt()
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) / iters * 1e3

    bytes_moved = (M * N * 2) * 2  # read + write, fp16=2 bytes
    nt_gbps = bytes_moved / (nt_ms * 1e-3) / 1e9
    pt_gbps = bytes_moved / (pt_ms * 1e-3) / 1e9

    print(f"\n[benchmark] NineToothed fixed: {nt_ms:.3f} ms  ({nt_gbps:.1f} GB/s)")
    print(f"[benchmark] PyTorch          : {pt_ms:.3f} ms  ({pt_gbps:.1f} GB/s)")
    print(f"[benchmark] speedup          : {pt_ms/nt_ms:.2f}x")

    # Both must produce correct output (not just fast).
    torch.testing.assert_close(
        softmax_fixed(x), reference(x), atol=1e-2, rtol=1e-2
    )
