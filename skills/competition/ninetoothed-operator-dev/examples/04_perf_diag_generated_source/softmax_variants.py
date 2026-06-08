"""Example 4 — softmax performance-regression diagnosis.

Family: performance / diagnosis.
Confidence: HIGH — kernel is the verified `softmax` pattern (ops/ninetoothed/
kernels/softmax.py). Two wrappers differ only in the block size:

  softmax_fast  -> BLOCK_SIZE = row length N            (no wasted lanes)
  softmax_slow  -> BLOCK_SIZE = SLOW_BLOCK (>> N)        (wastes masked lanes)

Both are numerically correct: out-of-range lanes read `other=-inf`, so
exp(-inf)=0 and the max is unchanged. `softmax_slow` just does more work per
row -> a real, reproducible regression to diagnose with
inspect_generated_source.py + bench_compare.py.
"""
import ninetoothed
import ninetoothed.language as ntl
import torch
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)
SLOW_BLOCK = 8192  # deliberately oversized to waste masked lanes


def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((1, BLOCK_SIZE)), output.tile((1, BLOCK_SIZE))


def application(input, output):
    row_minus_max = input - ntl.max(input)
    numerator = ntl.exp(row_minus_max)
    output = numerator / ntl.sum(numerator)  # noqa: F841


_kernel = ninetoothed.make(
    arrangement, application, (Tensor(2, other=float("-inf")), Tensor(2))
)


def softmax_fast(x):
    out = torch.empty_like(x)
    _kernel(x, out, BLOCK_SIZE=x.shape[-1])      # exact row length
    return out


def softmax_slow(x):
    out = torch.empty_like(x)
    _kernel(x, out, BLOCK_SIZE=SLOW_BLOCK)        # oversized -> wasted lanes
    return out


def reference(x):
    return torch.softmax(x, dim=-1)
