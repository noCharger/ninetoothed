# proxy_tasks — offline evaluation task set

**English** | [中文](README.zh-CN.md)

The 8 hidden evaluation tasks are not visible. The proxy task set is a stand-in
that mirrors the hidden distribution, giving a reproducible offline benchmark to
measure the skill's before/after gain (A/B) and to provide a reward signal for
the Stage 3 MOO loop.

## Size and split

24 tasks across the four families, 6 per family; 4 train + 2 holdout each
(16 train, 8 holdout total).

| Family | train | holdout |
|---|---|---|
| elementwise / broadcast | add, mul_broadcast, relu, gelu | silu, masked_add |
| reduction / blocking | sum_last, mean_last, softmax, rms_norm | max_last, l2_norm |
| layout-sensitive | contig_transpose, flip_last, narrow_half, strided_gather | pixel_unshuffle, pixel_shuffle |
| perf / diagnosis | softmax_no_maxsub, mean_no_upcast, add_bench_memorybound, inspect_tile_config | noncontig_regression, aot_numwarps_mismatch |

Holdout deliberately uses operators absent from train (e.g. pixel_unshuffle /
pixel_shuffle). Passing holdout therefore shows the skill's gain generalises
beyond the tuning set rather than overfitting public samples.

## Two task kinds

- **operator** (18 tasks): implement a NineToothed operator. Carries a PyTorch
  reference and an input generator; correctness is judged by MERE/MARE against
  the reference (thresholds in `../../scripts/run_correctness_matrix.py`:
  fp32 1.22e-4, fp16 9.77e-4, bf16 7.81e-3).
- **diagnosis** (6 tasks): given a failing or slow kernel, locate the root cause
  and give a minimal fix. Carries a `scenario` and an `expected_findings` list;
  scored by findings hit, with correctness judged separately when a fix is required.

Task `prompt` / `scenario` / `expected_findings` strings are kept in Chinese on
purpose: they are task content for a Chinese-language competition, not code.

## Task sources

- operator-development scenarios from merged PRs and issues in InfiniTensor/ninetoothed
- existing operators in the repo `examples/` and `ntops`
- hand-authored cases for the layout-sensitive and scatter coverage gaps
  (public work is weak in these two families)

## Files

```
proxy_tasks/
  schema.py       # TaskSpec definition + validation; randn_inputs factory
  elementwise.py  # 6 tasks
  reduction.py    # 6 tasks
  layout.py       # 6 tasks
  perf_diag.py    # 6 tasks (diagnosis)
  loader.py       # load, validate, regenerate manifest, optional torch CPU self-check
  manifest.json   # generated from the modules so it never drifts from the code
  README.md
```

## Usage

```bash
# validate schema + structural invariants (24 tasks, 4+2 per family), rebuild manifest
python loader.py

# additionally run each operator reference on CPU as a numeric self-check (needs torch)
python loader.py --check
```

Modules import without torch installed (torch is imported lazily inside
reference / make_inputs), so structural validation and manifest generation do
not need a GPU; the numeric self-check runs where torch is present.

## Link to the evaluation loop

The evaluator (`../skill_eval/`) consumes this set: each task is solved under
no-skill and v0; operator tasks are judged by MERE/MARE with `robust_bench` for
performance and reward-hacking checks; diagnosis tasks are scored by
`expected_findings` hit rate; failures are attributed by `failure_classifier`.
The train split drives Stage 3 optimization; holdout is used only for final
generalisation evaluation.
