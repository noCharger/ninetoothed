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
