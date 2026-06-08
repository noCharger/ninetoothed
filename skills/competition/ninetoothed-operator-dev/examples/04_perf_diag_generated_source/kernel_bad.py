"""
DELIBERATELY SUBOPTIMAL softmax kernel — used as the 'before' in the perf/diag
self-test task.

Two intentional problems:
1. Missing numerical stability: no subtract-max step → exp(large value) overflows.
2. No fp16→fp32 upcast: accumulation in fp16 → catastrophic cancellation on
   large rows.

These cause correctness failures on fp16 inputs with large row values AND are
detectable in the generated source (no intermediate fp32 cast visible in the
Triton code, smaller tile → more passes over HBM).
"""

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(x, out, BLOCK_SIZE=BLOCK_SIZE):
    return x.tile((1, BLOCK_SIZE)), out.tile((1, BLOCK_SIZE))


def application(x, out):
    # BUG 1: no subtract-max (numerical instability)
    # BUG 2: no upcast to fp32 (precision loss on fp16)
    numerator = ntl.exp(x)
    out = numerator / ntl.sum(numerator)  # noqa: F841


_TENSORS = (Tensor(2, other=float("-inf")), Tensor(2))

kernel_bad = ninetoothed.make(arrangement, application, _TENSORS)
