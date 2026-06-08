# Elementwise / Broadcast family

Output element depends only on same-index inputs. The arrangement just tiles
every tensor the same way; the application writes one expression.

## 1D flat (add, mul, relu, gelu, silu)

```python
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

def arrangement(input, other, output, BLOCK_SIZE=BLOCK_SIZE):
    return (input.tile((BLOCK_SIZE,)),
            other.tile((BLOCK_SIZE,)),
            output.tile((BLOCK_SIZE,)))

def application(input, other, output):
    output = input + other  # noqa: F841

kernel = ninetoothed.make(arrangement, application, tuple(Tensor(1) for _ in range(3)))
```

Torch wrapper (the standard invocation pattern):

```python
import torch
def add(input, other):
    output = torch.empty_like(input)
    kernel(input, other, output, BLOCK_SIZE=1024)   # constexpr passed at call
    return output
```

For an N-D input that is contiguous, flatten in the wrapper and reshape back
(this is how `silu`/`swiglu` work in the repo):

```python
def silu(x):
    x_flat = x.flatten()
    out_flat = torch.empty_like(x_flat)
    kernel(x_flat, out_flat, BLOCK_SIZE=1024)
    return out_flat.view_as(x)
```

## Unary via libdevice / ntl math

`application` can call `ntl.<fn>`; common: `ntl.exp`, `ntl.sigmoid`, `ntl.tanh`,
`ntl.where`, `ntl.maximum`. Example ReLU and GELU-ish:

```python
def application(input, output):
    output = ntl.where(input > 0, input, 0.0)  # noqa: F841
```

## Broadcast (A[M,1] op B[1,N])

Tile both operands and the output on the 2-D grid; rely on Triton scalar/row
broadcast inside the tile. Keep the broadcasted axis size 1 in the source
`Tensor` and let the expression broadcast:

```python
def arrangement(a, b, output, BM=Symbol("BM", constexpr=True),
                BN=Symbol("BN", constexpr=True)):
    return (a.tile((BM, 1)), b.tile((1, BN)), output.tile((BM, BN)))
```

If the broadcast shapes do not divide evenly, pass `other=` so out-of-range
loads are well-defined (see mask below).

## Mask / padding (`other=`)

When a tile can read past the tensor edge (non-divisible shape, or a masked op),
declare the fill value on the **source** `Tensor`:

```python
tensors = (Tensor(2, other=0.0), Tensor(2))   # OOB loads read 0.0
```

Use `float("-inf")` for max/softmax inputs, `0.0` for sum/add. To apply an
explicit boolean mask inside the application use `ntl.where(cond, x, fill)`.

## fp16 / bf16 numeric care

Pure elementwise add/mul in fp16 is fine. Only upcast when the op is
numerically sensitive (involves exp, division by small numbers, or long sums) —
see `reduction.md`. Do not blanket-upcast; it wastes registers.

## Pitfalls (see common-errors.md for fixes)

- Forgetting `# noqa: F841` on the `output = ...` line.
- Using `tile((BLOCK_SIZE,))` on a 2-D tensor when you meant row-wise — that
  tiles dim 0. For per-row elementwise use `tile((1, BLOCK_SIZE))`.
- Passing a multi-dim tensor to a `Tensor(1)` kernel without flattening.
