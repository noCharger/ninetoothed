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
    "tile-based Python DSL: define `arrangement(*tensors)` returning tiled views (via "
    "`tensor.tile((BLOCK_SIZE,))`) and `application(*tiles)` computing on them, then "
    "`kernel = ninetoothed.make(arrangement, application, tuple(Tensor(ndim) for ...))` "
    "compiles it; call `kernel(*tensors, BLOCK_SIZE=...)` writing into a preallocated "
    "output. Follow the provided skill guidance exactly for API and pitfalls.\n\n"
    "Output EXACTLY ONE self-contained file — no separate kernel module, no test:\n"
    "=== FILE: wrapper.py ===\n```python\n<all code here>\n```\n\n"
    "wrapper.py MUST import ninetoothed, define the kernel inline, and expose "
    "`def solve(*inputs)` that allocates the output, launches the kernel, and RETURNS the "
    "output tensor. The grader imports and calls solve(). Do not read files; everything "
    "you need is in this message."
)

_FILE_RE = re.compile(r"===\s*FILE:\s*([A-Za-z0-9_./-]+)\s*===\s*\n```(?:python)?\s*\n(.*?)```",
                      re.DOTALL)

# task family -> the single most relevant reference to inline (keeps context tight)
_FAMILY_REF = {
    "elementwise": "elementwise.md", "reduction": "reduction.md",
    "layout": "layout.md", "perf_diag": "perf-diag.md", "perf-diag": "perf-diag.md",
}


def _read_skill_context(ws: pathlib.Path, prompt: str, budget_chars: int = 12000) -> str:
    """Inline the skill text into the prompt. A plain LM cannot read files, so with-skill
    mode must hand it SKILL.md + the family-relevant reference verbatim. Returns "" when no
    skill/ was seeded (no_skill mode) — that is exactly the A/B contrast we measure."""
    skill = ws / "skill"
    if not skill.is_dir():
        return ""
    parts = []
    sk = skill / "SKILL.md"
    if sk.exists():
        parts.append("=== SKILL.md ===\n" + sk.read_text(encoding="utf-8", errors="ignore"))
    # pick the reference matching the family named in the prompt
    fam = ""
    m = re.search(r"Task \(([a-z_\-]+)", prompt)
    if m:
        fam = m.group(1)
    ref_name = _FAMILY_REF.get(fam)
    refs_dir = skill / "references"
    chosen = []
    if ref_name and (refs_dir / ref_name).exists():
        chosen.append(refs_dir / ref_name)
    # always include the common-errors + taxonomy if room allows
    for extra in ("common-errors.md", "operator-taxonomy.md"):
        p = refs_dir / extra
        if p.exists():
            chosen.append(p)
    for p in chosen:
        parts.append(f"=== references/{p.name} ===\n" +
                     p.read_text(encoding="utf-8", errors="ignore"))
    text = "\n\n".join(parts)
    return text[:budget_chars]


def _load(model_dir: str):
    if model_dir in _MODEL_CACHE:
        return _MODEL_CACHE[model_dir]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    # transformers 5.x renamed torch_dtype -> dtype; try new kw first, fall back.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map="cuda")
    except TypeError:
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


def _solve_runs(ws: pathlib.Path, task_id: str, proxy_tasks_dir: str) -> tuple[bool, str]:
    """Actually call solve() on the task's real inputs in a subprocess. Catches missing
    imports, wrong signature, and shape/runtime errors that a bare import misses — this
    is the repair signal that makes a weak model converge."""
    if not (ws / "wrapper.py").exists():
        return False, "no wrapper.py produced"
    check = (
        "import sys; sys.path.insert(0, %r); sys.path.insert(0, '.')\n"
        "import torch\n"
        "from loader import load_all\n"
        "t=[x for x in load_all() if x.id==%r][0]\n"
        "from wrapper import solve\n"
        "ins=t.make_inputs(device='cuda', dtype='float32')\n"
        "out=solve(*[x.clone() for x in ins])\n"
        "assert torch.is_tensor(out), 'solve did not return a tensor'\n"
        "print('SOLVE_OK', tuple(out.shape))\n"
    ) % (proxy_tasks_dir, task_id)
    proc = subprocess.run([sys.executable, "-c", check], cwd=str(ws),
                          capture_output=True, text=True, timeout=300)
    if proc.returncode == 0 and "SOLVE_OK" in proc.stdout:
        return True, ""
    return False, (proc.stderr or proc.stdout)[-900:]


def make_qwen_solver(model_dir: str, max_new_tokens: int = 3072, repair_rounds: int = 2):
    """Return a Solver closure: (prompt, ws, opts) -> SolverResult. Loads model once."""
    from run_episode import SolverResult   # local import to avoid cycle at module load

    def _solver(prompt: str, ws: pathlib.Path, opts: dict) -> SolverResult:
        tok, model = _load(model_dir)
        t0 = time.time()
        # inline skill text (with-skill modes) — a plain LM cannot open files itself
        skill_ctx = _read_skill_context(ws, prompt)
        user = prompt if not skill_ctx else (
            "Use the following skill guidance verbatim (API, workflow, pitfalls):\n\n"
            + skill_ctx + "\n\n---\n\n" + prompt)
        messages = [{"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user}]
        tin = tout = 0
        text, a, b = _generate(tok, model, messages, max_new_tokens)
        tin += a; tout += b
        _write_files(text, ws)

        # verification signal for the repair loop: prefer actually running solve() on the
        # task's real inputs (catches missing imports/shape/runtime bugs); fall back to import.
        task_id = opts.get("task_id")
        ptd = opts.get("proxy_tasks_dir")
        is_operator = opts.get("kind", "operator") == "operator"

        def _check():
            if is_operator and task_id and ptd:
                return _solve_runs(ws, task_id, ptd)
            return _import_ok(ws)

        ok, err = _check()
        rounds = 0
        while not ok and rounds < repair_rounds:
            rounds += 1
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content":
                             f"Your wrapper.py failed when the grader ran solve():\n{err}\n"
                             "Fix it. Re-emit the COMPLETE wrapper.py in the "
                             "=== FILE: wrapper.py === format. Remember `import torch` and "
                             "any other imports you use."})
            text, a, b = _generate(tok, model, messages, max_new_tokens)
            tin += a; tout += b
            _write_files(text, ws)
            ok, err = _check()

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
