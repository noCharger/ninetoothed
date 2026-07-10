#!/usr/bin/env python3
"""
glm_solver.py — a hosted GLM model (Zhipu / bigmodel.cn) as the episode solver.

Same role as qwen_solver but the "black-box LLM = proposal generator" is a strong
hosted model reached over an OpenAI-compatible API, instead of a local weight file.
Generation happens over the network; the kernel compile/correctness still runs on the
GPU host (the oracle). This sidesteps the local-model capability ceiling.

Reuses qwen_solver's stable helpers (skill inlining, file parsing, correctness-feedback
repair, diagnosis path) — only the token-generation call differs. So the two solvers
score identically and the A/B stays fair.

Config via env / args:
    GLM_API_KEY   : bearer key (id.secret form)
    GLM_MODEL     : model id (default glm-4.5-flash; glm-4.6 / glm-5.2 need account balance)
    GLM_ENDPOINT  : override the chat-completions URL

GLM-4.5+ are reasoning models: the final answer is in choices[0].message.content;
reasoning_content is separate and ignored. max_tokens is set generously so reasoning
does not starve the answer.
"""
from __future__ import annotations

import json
import os
import pathlib
import time
import urllib.error
import urllib.request

# reuse the stable solve-loop helpers from the local-model solver
from qwen_solver import (                                   # noqa: E402
    _read_skill_context, _write_files, _strip_fences,
    _solve_runs, _import_ok, _SYSTEM, _SYSTEM_DIAG,
)
from run_episode import SolverResult                        # noqa: E402

_DEFAULT_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


def _glm_chat(messages: list[dict], model: str, api_key: str, endpoint: str,
              max_tokens: int, temperature: float = 0.2, timeout: int = 180
              ) -> tuple[str, int, int]:
    """One chat completion. Returns (content, prompt_tokens, completion_tokens)."""
    body = json.dumps({
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    last_err = ""
    for attempt in range(3):                                # simple retry on transient errors
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            msg = d["choices"][0]["message"]
            content = msg.get("content") or ""              # ignore reasoning_content
            usage = d.get("usage", {}) or {}
            return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read()[:200].decode(errors='ignore')}"
            if e.code in (400, 401, 403):                   # not transient — stop
                break
        except Exception as e:                              # noqa: BLE001
            last_err = str(e)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GLM chat failed: {last_err}")


def make_glm_solver(model: str | None = None, api_key: str | None = None,
                    endpoint: str | None = None, max_new_tokens: int = 4096,
                    repair_rounds: int = 3):
    api_key = api_key or os.environ.get("GLM_API_KEY")
    if not api_key:
        raise SystemExit("--solver glm needs GLM_API_KEY (or api_key=...)")
    model = model or os.environ.get("GLM_MODEL", "glm-4.5-flash")
    endpoint = endpoint or os.environ.get("GLM_ENDPOINT", _DEFAULT_ENDPOINT)

    def _gen(messages):
        return _glm_chat(messages, model, api_key, endpoint, max_new_tokens)

    def _solver(prompt: str, ws: pathlib.Path, opts: dict) -> SolverResult:
        t0 = time.time()
        is_operator = opts.get("kind", "operator") == "operator"
        system = _SYSTEM if is_operator else _SYSTEM_DIAG
        skill_ctx = _read_skill_context(ws, prompt)
        user = prompt if not skill_ctx else (
            "Use the following skill guidance verbatim (API, workflow, pitfalls):\n\n"
            + skill_ctx + "\n\n---\n\n" + prompt)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        tin = tout = 0
        text, a, b = _gen(messages); tin += a; tout += b
        _write_files(text, ws)

        if not is_operator:
            dm = ws / "diagnosis.md"
            if not dm.exists() or not dm.read_text(encoding="utf-8").strip():
                body = _strip_fences(text) if ("```" in text or "=== FILE" in text) else text
                dm.write_text(body.strip() + "\n", encoding="utf-8")
            return SolverResult(tokens_in=tin, tokens_out=tout,
                                wall_seconds=time.time() - t0, compiled=True, compile_attempts=1)

        task_id = opts.get("task_id"); ptd = opts.get("proxy_tasks_dir")

        def _check():
            if task_id and ptd:
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
                             "=== FILE: wrapper.py === format. Remember `import torch`."})
            text, a, b = _gen(messages); tin += a; tout += b
            _write_files(text, ws)
            ok, err = _check()

        return SolverResult(tokens_in=tin, tokens_out=tout, wall_seconds=time.time() - t0,
                            gpu_seconds=0.0, compiled=ok, compile_attempts=rounds + 1,
                            raw={"import_ok": ok, "model": model})
    return _solver


if __name__ == "__main__":
    # connectivity smoke test (no GPU): one chat round-trip
    key = os.environ.get("GLM_API_KEY")
    if not key:
        print("set GLM_API_KEY to run the smoke test"); raise SystemExit(0)
    txt, ni, no = _glm_chat([{"role": "user", "content": "reply with the single word ok"}],
                            os.environ.get("GLM_MODEL", "glm-4.5-flash"), key,
                            _DEFAULT_ENDPOINT, max_tokens=50)
    print(f"glm ok: content={txt[:60]!r} tokens_in={ni} tokens_out={no}")
