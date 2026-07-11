# ninetoothed-operator-dev

**English** | [中文](README.zh-CN.md)

A reusable `.skill` that helps a coding agent write, validate, optimize, and
debug **NineToothed** DSL operators (arrangement + application), grounded in the
real `ninetoothed>=0.25.0` API.

Built for the NineToothed `.skill` challenge (T3-1-1). Targets two agents:
**Claude Code** and **GPT-5.5 Codex**, via the standard Anthropic Agent Skills
layout.

## What's inside

```
ninetoothed-operator-dev/
  SKILL.md                  # entry point: trigger / workflow / stop / checklist / constraints
  references/               # load-on-demand family playbooks
    operator-taxonomy.md    # classify before writing
    elementwise.md  reduction.md  layout.md  perf-diag.md  common-errors.md
  scripts/                  # runnable, no eval/exec/dynamic-import/shell-interp
    gen_pytorch_oracle.py       # emit a PyTorch reference + pytest scaffold (pure templating)
    run_correctness_matrix.py   # run pytest, summarize shape x dtype x layout into CSV (MERE/MARE)
    inspect_generated_source.py # read ~/.ninetoothed/<sha256>.py, report tiles/ops (read-only)
    bench_compare.py            # CUDA-event timing + Roofline classifier (importable)
    debug_arrangement.py        # verify an arrangement before compiling the kernel
    failure_classifier.py       # classify a failure as code_error / guidance_error / unknown
    strategy_selector.py        # read references/rules.json (if distilled), recommend an action
    aot_build_smoke.sh          # check AOT build produced .py + .h
  examples/                 # 4 worked self-test tasks (one per family)
  tests/
    self_test_tasks.md      # the 4 self-test tasks (mirror the hidden-task families)
    verifier_spec.md        # pass/fail rules aligned to the official 6 sub-scores
  REFERENCE.md              # citations & provenance
```

This package is self-contained — no network, no external repo dependency at run
time. A/B measurement and self-evolution tooling (the 24-task proxy set + answer
key, `claude -p`/local-model/GLM-API automation, and the two evolution routes)
live in a separate companion repo,
[`ninetoothed-skill-eval-harness`](https://github.com/noCharger/ninetoothed-skill-eval-harness);
it is not required to install or run this skill.

## Install / activate

**Claude Code** — copy or symlink the folder into a skills directory; it is
discovered automatically:

```bash
mkdir -p .claude/skills
cp -r ninetoothed-operator-dev .claude/skills/
# or user-wide: ~/.claude/skills/
```

**GPT-5.5 Codex** — place the same folder where your Codex config loads skills
(the package layout is the portable Anthropic Agent Skills format). The agent
reads `SKILL.md` first and pulls `references/*` on demand.

No installation step runs code; activation is just file discovery.

## Dependencies

| Dependency | Version | Used for |
|------------|---------|----------|
| ninetoothed | >= 0.25.0 | the DSL under development |
| triton | >= 3.0.0 | NineToothed backend |
| torch | >= 2.4.0 | oracle / correctness / benchmark baseline |
| numpy, sympy | per ninetoothed | transitive |
| pytest | >= 7 | correctness runner |
| CUDA Toolkit | 12.x | compile + run on NVIDIA GPU |

```bash
pip install "ninetoothed>=0.25.0" "torch>=2.4.0" "triton>=3.0.0" pytest
```

CUDA-free environments: `bench_compare.py` falls back to wall-clock timing with
a warning; correctness tests `skipif(not torch.cuda.is_available())`.

## Quick start (one operator, end to end)

```bash
# 1. emit a PyTorch oracle + test scaffold for your wrapper module `my_ops`
python scripts/gen_pytorch_oracle.py --op softmax \
    --wrapper-module my_ops --wrapper-fn softmax \
    --shapes 1024 4095 64,128 --dtypes float16 float32 \
    --out test_softmax_correctness.py

# 2. run the shape x dtype x layout matrix
python scripts/run_correctness_matrix.py test_softmax_correctness.py --csv matrix.csv

# 3. inspect what NineToothed generated
python scripts/inspect_generated_source.py

# 4. benchmark + Roofline (from your own bench file importing bench_compare)
```

## Safety / compliance

No API keys, no network calls, no hidden answers, no test deletion. Scripts are
read-only or subprocess-only (list-form argv, `shell=False`). See `SKILL.md`
section 5 and `REFERENCE.md`.

## Status

v0 (curated baseline). Self-test tasks and an A/B vs no-skill baseline are the
next milestone (see `tests/self_test_tasks.md`).
