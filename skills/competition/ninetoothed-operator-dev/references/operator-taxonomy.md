# Operator Taxonomy — classify before you write

Pick the family by the **computation shape**, not the op name. Many torch ops
decompose: `mse_loss` = elementwise `(a-b)**2` then reduction `mean`; `fliplr`
= layout `flip(dim=-1)`. Classify each stage.

## Decision rule

```
Does the output element depend only on the same-index input element(s)?
├── yes, 1:1 (maybe with broadcast / mask)        → ELEMENTWISE
└── no
    ├── output folds many inputs along an axis
    │   (sum/mean/max/softmax/norm/argmax)         → REDUCTION
    ├── output is a re-indexing / re-view of input
    │   (flip/narrow/permute/space-to-depth,
    │    non-contiguous / stride / offset inputs)  → LAYOUT
    └── task is "benchmark / inspect / AOT /
        find the regression / fix failing test"    → PERF-DIAG
```

Composite ops: handle the elementwise stage with `elementwise.md`, the fold
with `reduction.md`. A reduction with `reduction='none'` is pure elementwise.

## Family → reference

| Family | Reference | Core idiom |
|--------|-----------|------------|
| Elementwise / broadcast | `elementwise.md` | `tile((BLOCK_SIZE,))` flat; `tile((1, BLOCK_SIZE))` row-wise; `other=` for mask |
| Reduction / blocking | `reduction.md` | row tile + `ntl.sum/max`; fp32 accumulate; `reduction=` modes |
| Layout-sensitive | `layout.md` | `ravel`/`flatten`/`permute`; `offsets()`; contiguous-fallback decision |
| Perf / diagnosis | `perf-diag.md` | `simulate_arrangement`, generated source, benchmark, Roofline, AOT |

When a failure appears at any step, jump to `common-errors.md`.

## Before writing: find prior art

Run `scripts/find_similar_ops.py <keyword>` to locate the nearest existing
arrangement in the repo (`ops/ninetoothed/kernels/`, `src/ninetoothed/`,
`ntops/`). Reusing an existing arrangement skeleton is faster and matches repo
style — a scoring dimension in the rubric.
