# Verifier spec

How each self-test run is scored. Mirrors the official single-task rubric
(rules §4.4): 0–10 per task across six sub-scores. Used identically for
no-skill and with-skill runs so the delta is meaningful.

## Sub-scores (per task, total 10)

| Sub-score | Max | Pass condition | How to check |
|-----------|-----|----------------|--------------|
| Task completion | 4 | operator semantics / shape / dtype / boundary or diagnosis goal met | `run_correctness_matrix.py` full pass = 4; partial = 1–3; none = 0 |
| Test & verification | 2 | specified tests pass and the loop is closed | all matrix cases pass + commands+results recorded = 2; partial = 1; not run/failed = 0 |
| Performance awareness | 1 | benchmark / generated-source analysis / sound optimization rationale present | benchmark.csv + Roofline verdict, or `inspect_generated_source` evidence |
| Patch minimality | 1 | no unrelated refactor, no mass reformat, no breaking change | `git diff` review |
| Repo-style consistency | 1 | naming / structure / error handling / test & doc style match repo | ruff/black clean + matches `ops/ninetoothed/kernels/` style |
| Process & compliance | 1 | reproducible trace, no secrets, no network, no test bypass, no hidden answers | inspect the run log |

## Deterministic gates (auto-checkable)

- **Correctness**: `python scripts/run_correctness_matrix.py <test_file>` exits 0
  (no FAILED/ERROR; SKIPPED allowed only when CUDA absent).
- **Benchmark present** (tasks 2, 4): a CSV with ≥3 input sizes and a stated
  compute/memory-bound conclusion exists.
- **Generated-source evidence** (task 4): `inspect_generated_source.py` output
  captured for the before and after kernels.
- **Compliance**: no `eval(`/`exec(`/network calls in produced scripts; no test
  deleted vs the provided scaffold.

## Award-threshold mapping (rules §4.5)

The skill is "on track" when, across the 4 self-tests:

- with-skill total ≥ no-skill total on every task, and
- ≥ 2 tasks demonstrate a valid performance verification (tasks 2 & 4), and
- 0 negative-transfer regressions (no case that passed no-skill fails
  with-skill).

These mirror the hidden-task thresholds: ≥48/80 pre-scaling, ≥5/8 tasks with
completion ≥3/4, ≥2 tasks with valid performance work.
