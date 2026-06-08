# Reduction / Blocking family

Output folds many inputs along an axis: sum, mean, max, argmax, softmax, norms.

## Row-wise softmax (the canonical pattern)

Tile per row with `tile((1, BLOCK_SIZE))`; the loaded tile is one row, so
`ntl.max` / `ntl.sum` over the whole tile is the row reduction. Use
`other=float("-inf")` so padded lanes do not affect the max.

```python
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((1, BLOCK_SIZE)), output.tile((1, BLOCK_SIZE))

def application(input, output):
    row_minus_max = input - ntl.max(input)      # subtract max for stability
    numerator = ntl.exp(row_minus_max)
    output = numerator / ntl.sum(numerator)     # noqa: F841

kernel = ninetoothed.make(
    arrangement, application, (Tensor(2, other=float("-inf")), Tensor(2)))
```

Wrapper passes the row length as the block size:

```python
def softmax(x):
    out = torch.empty_like(x)
    kernel(x, out, BLOCK_SIZE=x.shape[-1])
    return out
```

## RMS norm / mean over last dim

Accumulate in fp32 even for fp16/bf16 inputs — this is the main correctness
trap in this family.

```python
def application(input, weight, output, eps):
    x = ntl.cast(input, ntl.float32)
    var = ntl.sum(x * x) / input.shape[-1]
    output = input * ntl.rsqrt(var + eps) * weight  # noqa: F841
```

For an op that must accept any ndim and reduce only the last dim, build the
arrangement shape from `len(input.shape)`: tile all leading dims as 1 and the
last as `BLOCK_SIZE`, then `squeeze` the leading singleton levels off `.dtype`.

## reduction='none' | 'mean' | 'sum' (loss family)

Split into two stages:

1. `reduction='none'` → pure elementwise; emit the per-element result (e.g.
   `(a-b)**2` for MSE). Use `elementwise.md`.
2. `'mean'` / `'sum'` → fold the elementwise result. `'mean'` divides by
   `numel` in the wrapper after the sum-reduction kernel.

Keep the elementwise kernel and the reduction kernel separate; compose in the
wrapper. Do not try to express both stages in one arrangement unless the fold
axis matches the tile.

## Blocked reduction (axis reduction with large N)

For `(B, N) -> (B,)` with large N, tile as `(1, BLOCK)` then a second
`tile((1, -1))` to iterate K blocks; loop in the application and combine. Output
is `tile((1,))` per row — do **not** force the output arrangement to mirror the
input's K blocks (causes `RecursionError`; see common-errors.md).

`ntl.max(input, axis=1)` reduces along a specific tile axis (used by pooling).

## Pitfalls

- fp16 accumulation overflow / precision loss → upcast to fp32 before sum.
- Missing `other=float("-inf")` on a softmax/max input → NaN from padded lanes.
- Over-arranging the output to match input blocks → `RecursionError`.
