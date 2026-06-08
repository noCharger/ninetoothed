# Performance / Diagnosis / Integration

Covers: verify arrangement, read generated source, benchmark + Roofline, AOT
build, and locate a regression. These map to the performance-awareness and
diagnostic rubric items.

## A. Verify the arrangement (deterministic, no kernel run)

**Step 5 of the SKILL.md workflow.** Call this before compiling the kernel.

### Via the skill's debug script (recommended for agents)

```bash
# From the skill root, pass module_dotpath:fn_name
python scripts/debug_arrangement.py examples.02_reduction_parameterized.kernel:arrangement
```

Or import as a library from your own driver script:

```python
from scripts.debug_arrangement import check_no_oob, summarise
from my_kernel import arrangement, _TENSORS

ok = check_no_oob(arrangement, _TENSORS)   # prints summary + returns bool
# Output:
#   tensor[0]  source=(64, 512)  target=(64, 1, 512)  programs=64  tile=(1, 512)
#              oob=0  unique_read=32768/32768
#   tensor[1]  source=(64,)      target=(64, 1)        programs=64  tile=(1,)
#              oob=0  unique_read=64/64
#   status: OK  all_covered=True
```

Interpret the output:

| Field | Meaning | Alarm condition |
|-------|---------|-----------------|
| `oob_count` | tiles that read a -1 sentinel (OOB access) | any value > 0 |
| `unique_read` / `total` | how many distinct source elements are accessed | < total may mean elements are missed |
| `tile_shape` | shape seen inside each GPU program | should match your intended block dims |

### Underlying API (for reference)

```python
from ninetoothed.debugging import simulate_arrangement
src, tgt = simulate_arrangement(arrangement, tensors)   # device defaults to cuda
# src[i]: source index grid; tgt[i]: arranged grid with source indices
```

### Visualization for human review (optional, requires matplotlib)

**Boundary rules — read before calling:**

| API | Headless-safe | Who should use it |
|-----|:-:|---|
| `debug_arrangement.visualize_and_save(arrangement, tensors, save_dir=".")` | ✅ yes | Agent generates PNG → human inspects |
| `ninetoothed.visualization.visualize(tensor, save_path="x.png")` | ✅ yes | Same — headless PNG |
| `ninetoothed.visualization.visualize_arrangement(arrangement, tensors)` | ❌ no | Local development only (tkinter GUI) |

Install optional deps first: `bash setup.sh --with-viz`

```python
from scripts.debug_arrangement import visualize_and_save
paths = visualize_and_save(arrangement, tensors, save_dir="report/")
# gracefully prints a warning and returns [] if matplotlib not installed
```

The agent should call `visualize_and_save` ONLY if the `--with-viz` flag is
confirmed available in the environment; otherwise skip and note "visualization
not generated (matplotlib not installed)" in the trace log.

## B. Read the generated Triton source

NineToothed caches generated source at `~/.ninetoothed/<sha256>.py`
(`ninetoothed.generation.CACHE_DIR`). After a kernel has been built once, run:

```
python scripts/inspect_generated_source.py            # newest cached kernel
python scripts/inspect_generated_source.py --digest <sha256>
```

It reports the `tl.*` ops used, any `num_warps` / `num_stages` constants, tile
sizes, and load/store counts — the evidence you cite when judging whether a
kernel is reasonable or has regressed. The script is read-only and parses with
`ast` (no execution).

## C. Benchmark + Roofline

Use `scripts/bench_compare.py` as a library from your own bench file:

```python
from scripts.bench_compare import benchmark, roofline
ms = benchmark(lambda: my_op(x), warmup=25, iters=100)   # CUDA-event timed
gbps = bytes_moved / (ms * 1e-3) / 1e9
verdict = roofline(flops=flops, bytes_moved=bytes_moved, gpu="H100")
# verdict -> "compute-bound" | "memory-bound" with the ridge point used
```

Benchmark requirements (rubric): a PyTorch (or repo Triton) baseline, ≥3 input
sizes incl. one non-power-of-two, mean±std over ≥30 iters, GB/s or TFLOPS, and
a compute/memory-bound conclusion.

**Ridge points** (arithmetic intensity FLOP/byte; below ⇒ memory-bound):
H100 SXM fp16 ≈ 295, A100 fp16 ≈ 156. Update from the actual device's peak
FLOPs ÷ peak bandwidth if different.

## D. AOT build

`make` with a non-`torch` caller emits ahead-of-time artifacts instead of a
JIT handle:

```python
ninetoothed.make(arrangement, application, tensors,
                 caller="cuda", kernel_name="my_op", output_dir="build/")
# writes a .py launcher and a .h header into build/
```

Smoke-test with `scripts/aot_build_smoke.sh build/` (checks the expected
`.py` + `.h` exist and are non-empty). If AOT fails where JIT worked, suspect
`num_warps` / `num_stages` defaults vs the target — pass them explicitly to
`make(...)`.

## E. Locate a performance regression

1. Reproduce: benchmark current vs the known-good revision on identical shapes.
2. Diff generated source: `inspect_generated_source.py` on both digests; compare
   tile sizes, num_warps/num_stages, and load/store counts.
3. Attribute: a regression usually shows up as a changed tile/auto-tune config
   or extra load/store. State the evidence (which field changed) in the report.
4. Minimal fix: pin the better config via `make(..., num_warps=, num_stages=)`
   or `block_size()` bounds — do not refactor unrelated code.

## F. fp16/bf16 tolerance reference (for correctness during perf work)

| dtype | atol | rtol |
|-------|------|------|
| float32 | 1e-5 | 1e-5 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1e-2 | 1e-2 |

matmul/attention in fp16 typically need rtol≈1e-2; elementwise stays tight.
Always compare against PyTorch with `torch.testing.assert_close(got, expected,
atol=, rtol=)`.
