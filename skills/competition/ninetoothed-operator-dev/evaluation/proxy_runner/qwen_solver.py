#!/usr/bin/env python3
"""
qwen_solver.py — a local open-weight coding model as the episode solver.

This is the "black-box LLM = proposal generator" from the plan, realised with a local
Qwen2.5-Coder model instead of a hosted API. The model is fixed (never trained); the
skill text is what evolves. Because the model has essentially never seen NineToothed,
its kernels lean heavily on the skill's references — so the skill's A/B gain is large
and honest.

The model is loaded ONCE (module-level cache) and reused across every episode in the
orchestration process; reloading a 7B model per episode would dominate wall time.

The model is not tool-using, so we wrap it in a minimal generate→write→(optional
import-repair) loop:
  1. system+user chat prompt instructs it to emit each deliverable in a
     `=== FILE: <name> ===` block.
  2. parse those blocks into files in the sandbox.
  3. one optional repair round: if wrapper.py fails to import, feed the error back.

Token counts come from the tokenizer, so cost accounting in the journal is real.
torch/transformers are imported lazily so this file stays importable on the Mac.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import time
from typing import Optional

_MODEL_CACHE: dict = {}

_SYSTEM = (
    "You are a GPU kernel engineer writing NineToothed operators. NineToothed is a "
    "tile-based Python DSL: you define an `arrangement(*tensors)` returning tiled views "
    "and an `application(*tiles)` computing on them, then `ninetoothed.make(arrangement, "
    "application, tensors)` compiles a kernel. Follow the provided skill references "
    "exactly for API and pitfalls. Output ONLY the requested files, each in its own block:\n"
    "=== FILE: wrapper.py ===\n```python\n<code>\n```\n"
    "=== FILE: kernel.py ===\n```python\n<code>\n```\n"
    "=== FILE: test_correctness.py ===\n```python\n<code>\n```\n"
    "Use ABSOLUTE imports (the files sit flat in one dir): `from wrapper import solve`. "
    "wrapper.py MUST expose `def solve(*inputs)` returning the output tensor — the grader "
    "calls solve(). The test must use pytest and skip if CUDA is unavailable."
)

_FILE_RE = re.compile(r"===\s*FILE:\s*([A-Za-z0-9_./-]+)\s*===\s*\n```(?:python)?\s*\n(.*?)```",
                      re.DOTALL)


def _load(model_dir: str):
    if model_dir in _MODEL_CACHE:
        return _MODEL_CACHE[model_dir]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    _MODEL_CACHE[model_dir] = (tok, model)
    return tok, model


def _generate(tok, model, messages: list[dict], max_new_tokens: int = 2048) -> tuple[str, int, int]:
    import torch
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(model.device)
    n_in = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
    gen = out[0][n_in:]
    n_out = gen.shape[0]
    return tok.decode(gen, skip_special_tokens=True), n_in, int(n_out)


def _write_files(text: str, ws: pathlib.Path) -> list[str]:
    written = []
    for name, body in _FILE_RE.findall(text):
        name = name.strip().split("/")[-1]            # flatten any path
        if not name.endswith(".py") and not name.endswith(".csv"):
            continue
        (ws / name).write_text(body.strip() + "\n", encoding="utf-8")
        written.append(name)
    # fallback: if the model emitted a single unlabelled python block, treat it as wrapper
    if not written:
        m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if m:
            (ws / "wrapper.py").write_text(m.group(1).strip() + "\n", encoding="utf-8")
            written.append("wrapper.py")
    return written


def _import_ok(ws: pathlib.Path) -> tuple[bool, str]:
    """Try importing wrapper.py in a subprocess (isolates CUDA init + crashes)."""
    if not (ws / "wrapper.py").exists():
        return False, "no wrapper.py produced"
    proc = subprocess.run([sys.executable, "-c", "import wrapper"], cwd=str(ws),
                          capture_output=True, text=True, timeout=180)
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or proc.stdout)[-800:]


def make_qwen_solver(model_dir: str, max_new_tokens: int = 2048, repair_rounds: int = 1):
    """Return a Solver closure: (prompt, ws, opts) -> SolverResult. Loads model once."""
    from run_episode import SolverResult   # local import to avoid cycle at module load

    def _solver(prompt: str, ws: pathlib.Path, opts: dict) -> SolverResult:
        tok, model = _load(model_dir)
        t0 = time.time()
        messages = [{"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt}]
        tin = tout = 0
        text, a, b = _generate(tok, model, messages, max_new_tokens)
        tin += a; tout += b
        _write_files(text, ws)

        ok, err = _import_ok(ws)
        rounds = 0
        while not ok and rounds < repair_rounds:
            rounds += 1
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content":
                             f"That failed to import with:\n{err}\n"
                             "Fix it. Re-emit ALL files in the same === FILE: === format."})
            text, a, b = _generate(tok, model, messages, max_new_tokens)
            tin += a; tout += b
            _write_files(text, ws)
            ok, err = _import_ok(ws)

        return SolverResult(tokens_in=tin, tokens_out=tout, wall_seconds=time.time() - t0,
                            gpu_seconds=time.time() - t0, compiled=ok,
                            compile_attempts=rounds + 1,
                            raw={"import_ok": ok, "err": err[:200] if err else ""})
    return _solver


if __name__ == "__main__":
    # offline: verify the file-block parser without loading any model
    demo = ("blah\n=== FILE: wrapper.py ===\n```python\nprint('w')\n```\n"
            "=== FILE: test_correctness.py ===\n```python\ndef test(): assert True\n```\n")
    import tempfile
    ws = pathlib.Path(tempfile.mkdtemp())
    got = _write_files(demo, ws)
    assert set(got) == {"wrapper.py", "test_correctness.py"}, got
    assert (ws / "wrapper.py").read_text().strip() == "print('w')"
    print("qwen_solver parser self-test OK:", got)
