# REFERENCE.md — citations & provenance

All external material this `.skill` relies on, disclosed per rules §11 / §6.

## Primary source: NineToothed itself

- **NineToothed** — `github.com/InfiniTensor/ninetoothed`, package version
  `0.25.0` (`triton>=3.0.0`, `torch>=2.4.0`, `python>=3.10`). Every API used in
  `references/*` and `scripts/*` was verified against this version's source:
  - `make(arrangement, application, tensors, caller=..., num_warps=, num_stages=)`
    — `src/ninetoothed/make.py`
  - `Tensor` meta-ops `tile / expand / squeeze / unsqueeze / permute / flatten /
    ravel / pad / offsets` — `src/ninetoothed/tensor.py`
  - `Symbol(..., constexpr=True | meta=True)`, `block_size()` — `src/ninetoothed/symbol.py`
  - generated-source cache `~/.ninetoothed/<sha256>.py`
    (`CACHE_DIR`, `cache_source`) — `src/ninetoothed/generation.py`
  - `simulate_arrangement` — `src/ninetoothed/debugging.py`
  - `visualize(..., save_path=)` / `visualize_arrangement` — `src/ninetoothed/visualization.py`
- **Example kernels** (patterns adapted, repo style followed):
  `ops/ninetoothed/kernels/{add,softmax,mm,max_pool2d,...}.py` and
  `tests/test_ops.py` from the NineToothed examples repo.

## Design rationale (methodology, not code)

- **SkillsBench** (arxiv 2602.12670) — curated skills outperform self-generated;
  focused modules beat comprehensive docs. → motivates the curated, family-split
  `references/` design and the "do not self-generate at inference" stance.
- **SkillMOO** (arxiv 2604.09297) — prune/substitute editing helps, expansion
  does not. → motivates the planned trace-driven optimization (out of scope for
  v0; experimental only).
- **Anthropic Agent Skills** — `docs.claude.com/en/docs/agents-and-tools/agent-skills`
  — package layout, progressive disclosure, scripts-for-determinism.
- **KernelSwift** (Shanghai AI Lab) — three-technique reward-hacking detection
  (static AST analysis / dynamic runtime analysis / NCU roofline sanity check).
  → `evaluation/skill_eval/reward_hacking_guard.py` implements technique 1 and 2
  (no root/profiling access for NCU); `evaluation/skill_eval/robust_bench.py`'s
  outlier-removal + bandwidth-sanity check borrows the same paper's fixed-graph
  and IQR-outlier-removal measurement protocol.
- **KernelBench** (arXiv 2502.10517, Stanford Scaling Intelligence Lab) — the
  `Model`/`ModelNew`/`get_inputs()` task contract and its three-gate grading
  (compiles → `torch.allclose` correctness on random inputs → legality); v0.1's
  speed-of-light / excessive-speedup anti-cheat additions. → motivates the
  compile/correctness/legality gate mapping documented in `tests/verifier_spec.md`
  ("Three-gate mapping"); this skill's task interface (`wrapper.py:solve()`)
  differs from `ModelNew`, so only the grading *convention* is borrowed, not code.
- **MusaCoder** (arXiv 2606.04847, Moore Threads AI) — bans `aten::*`/cuBLAS
  high-level fallback (matmul/conv/reduce family) in generated kernels, detected
  via static + runtime analysis in its MooreEval sandbox, "命中即零奖励" (a hit
  zeroes the reward). → motivates `banned_fallback_analysis()` in
  `evaluation/skill_eval/reward_hacking_guard.py` (static/AST half only — no
  runtime/profiler confirmation) and its wiring into `rubric_scorer.py`'s
  completion cap and compliance sub-score.

## Tooling

- PyTorch, Triton, pytest, NumPy, SymPy — open-source, versions in `README.md`.

## AI-assistance disclosure

This `.skill`'s prose and scripts were drafted with AI assistance and then
verified against the NineToothed `0.25.0` source as listed above. No third-party
code was copied verbatim; example-kernel patterns are adapted and cited.

## Honor code

No API keys, credentials, unauthorized data, hidden evaluation answers, or
network dependencies are included. See `HONOR_CODE.md` (added at final
submission).
