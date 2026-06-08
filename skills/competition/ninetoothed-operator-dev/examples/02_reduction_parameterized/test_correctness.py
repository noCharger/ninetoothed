"""
Correctness + benchmark tests for parameterized reduction.

At least 2 self-test tasks require a benchmark; this is one of them.
"""

import itertools

import pytest
import torch

from .wrapper import reduce_last_dim

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE = "cuda"
TOL = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),  # fp32 sum: slight accumulation variance
    torch.float16: dict(atol=1e-2, rtol=1e-2),  # fp16 input, fp32 internal
}
REDUCTIONS = ["none", "sum", "mean"]
SHAPES = [(32, 128), (64, 511), (128, 1024)]  # incl. non-pow2
DTYPES = [torch.float32, torch.float16]


def reference(x, reduction):
    if reduction == "none":
        return x.clone()
    if reduction == "sum":
        return x.sum(dim=-1)
    return x.mean(dim=-1)


@pytest.mark.parametrize(
    "shape,dtype,reduction",
    list(itertools.product(SHAPES, DTYPES, REDUCTIONS)),
)
def test_reduce_last_dim(shape, dtype, reduction):
    x = torch.randn(*shape, dtype=dtype, device=DEVICE)
    expected = reference(x, reduction)
    got = reduce_last_dim(x, reduction)
    torch.testing.assert_close(got, expected, **TOL[dtype])


def test_nan_inf_boundary():
    """Sum of a row containing NaN propagates NaN (expected behaviour)."""
    x = torch.ones(4, 16, dtype=torch.float32, device=DEVICE)
    x[0, 0] = float("nan")
    out = reduce_last_dim(x, "sum")
    assert torch.isnan(out[0]), "NaN row should produce NaN sum"
    assert not torch.isnan(out[1]), "clean row should not produce NaN"


def test_3d_input():
    """Batch dim (B, M, N) is supported via view in the wrapper."""
    x = torch.randn(4, 32, 64, dtype=torch.float32, device=DEVICE)
    torch.testing.assert_close(
        reduce_last_dim(x, "mean"),
        x.mean(dim=-1),
        atol=1e-4, rtol=1e-4,
    )


# ---------------------------------------------------------------------------
# Benchmark (required by self_test_tasks.md for task 2)
# ---------------------------------------------------------------------------
@pytest.mark.benchmark
def test_benchmark(benchmark):
    """Compare NineToothed mean-reduction vs PyTorch on a realistic shape."""
    M, N = 1024, 2048
    x = torch.randn(M, N, dtype=torch.float16, device=DEVICE)

    def nt_fn():
        return reduce_last_dim(x, "mean")

    def pt_fn():
        return x.mean(dim=-1)

    # warm-up
    for _ in range(10):
        nt_fn()
        pt_fn()
    torch.cuda.synchronize()

    # if pytest-benchmark is installed it will handle timing;
    # otherwise run a manual loop so the test always passes.
    try:
        nt_result = benchmark(nt_fn)
    except TypeError:
        import time
        start = time.perf_counter()
        for _ in range(100):
            nt_fn()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 100 * 1e3
        print(f"\nNineToothed mean ({M}x{N} fp16): {elapsed:.3f} ms/iter")

    # correctness check after benchmark
    torch.testing.assert_close(
        reduce_last_dim(x, "mean"),
        x.float().mean(dim=-1).half(),
        atol=1e-2, rtol=1e-2,
    )
