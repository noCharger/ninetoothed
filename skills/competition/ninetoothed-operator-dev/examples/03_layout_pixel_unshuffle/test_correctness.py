"""
Correctness tests for pixel_unshuffle.

Key coverage:
* Multiple downscale factors (2, 3, 4).
* Contiguous AND non-contiguous (transposed) inputs — the defining feature of
  the layout-sensitive family.
* fp16 and fp32.
* Non-power-of-two spatial dimensions.
"""

import itertools

import pytest
import torch
import torch.nn.functional as F

from .wrapper import pixel_unshuffle

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE = "cuda"
# (B, C, H, W) — H and W must be divisible by the downscale factor used.
# Using multiples of 12 to cover factors 2, 3, 4 with a single shape set.
SHAPES = [(1, 2, 12, 12), (2, 4, 24, 24), (1, 1, 12, 36)]
FACTORS = [2, 3, 4]
DTYPES = [torch.float32, torch.float16]
TOL = {
    torch.float32: dict(atol=0, rtol=0),   # pure copy — must be exact
    torch.float16: dict(atol=0, rtol=0),
}


@pytest.mark.parametrize(
    "shape,factor,dtype,contig",
    list(itertools.product(SHAPES, FACTORS, DTYPES, [True, False])),
)
def test_pixel_unshuffle(shape, factor, dtype, contig):
    B, C, H, W = shape
    if H % factor != 0 or W % factor != 0:
        pytest.skip(f"shape {shape} not divisible by factor {factor}")

    x = torch.randn(B, C, H, W, dtype=dtype, device=DEVICE)

    if not contig:
        # Make non-contiguous by transposing the last two dims, then pass as-is.
        x = x.transpose(-1, -2)   # (B, C, W, H) — non-contiguous
        # swap H/W in shape expectations
        x = x.transpose(-1, -2)   # back to (B, C, H, W) but non-contiguous storage
        assert not x.is_contiguous(), "expected non-contiguous tensor for this branch"

    expected = F.pixel_unshuffle(x.contiguous(), factor)  # torch reference
    got = pixel_unshuffle(x, factor)

    torch.testing.assert_close(got, expected, **TOL[dtype])
    assert got.is_contiguous(), "output must be contiguous"
    assert got.shape == (B, C * factor * factor, H // factor, W // factor)


def test_not_supported_non_divisible():
    """H not divisible by factor → raises ValueError (documented limitation)."""
    x = torch.randn(1, 1, 5, 4, device=DEVICE)
    with pytest.raises(ValueError, match="divisible"):
        pixel_unshuffle(x, downscale_factor=2)
