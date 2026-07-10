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
import ssl
import time
import urllib.error
import urllib.request

# a verified SSL context that works across platforms (macOS python often lacks a system
# CA bundle); fall back to certifi's bundle, then to the default context.
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:  # noqa: BLE001
    _SSL_CTX = ssl.create_default_context()

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
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
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


# ===========================================================================
# Agentic (tool-calling) GLM solver — GLM drives its own loop via function calls.
# This is the "agent harness" form: instead of a scripted generate->parse->repair,
# the model decides to write_file / run_test / read_file and iterates on real tool
# results, like a coding agent. Much stronger for a capable model (GLM-4.6 / glm-5.2).
# ===========================================================================

_TOOLS_OPERATOR = [
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write (overwrite) a file in the working directory.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "e.g. wrapper.py"},
            "content": {"type": "string"}}, "required": ["filename", "content"]}}},
    {"type": "function", "function": {
        "name": "run_test",
        "description": "Run the grader: import wrapper.solve and check its output against "
                       "the reference on real inputs. Returns PASS or the error/mismatch.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file you previously wrote.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string"}}, "required": ["filename"]}}},
]

_TOOLS_DIAG = [
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write your analysis to diagnosis.md.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string"}, "content": {"type": "string"}},
            "required": ["filename", "content"]}}},
]

_AGENT_SYSTEM_OP = (
    "You are a GPU kernel engineer using NineToothed (a tile-based Python DSL: define "
    "arrangement(*tensors) returning tiled views and application(*tiles); "
    "kernel=ninetoothed.make(arrangement, application, tuple(Tensor(ndim) ...)); call "
    "kernel(*tensors, BLOCK_SIZE=...) into a preallocated output). Follow the provided "
    "skill guidance for API and pitfalls.\n\n"
    "Work agentically with the tools: write a SINGLE self-contained wrapper.py that "
    "imports ninetoothed + torch, defines the kernel inline, and exposes "
    "`def solve(*inputs)` returning the output tensor. Then call run_test. If it fails, "
    "read the error, fix wrapper.py, and run_test again. Stop when run_test returns PASS "
    "(or after a few honest attempts). Do NOT fake correctness — the grader is independent."
)
_AGENT_SYSTEM_DIAG = (
    "You are diagnosing a NineToothed GPU-kernel problem. Use the provided skill guidance "
    "(Roofline, fp16/fp32 accumulation, generated-source inspection, num_warps/num_stages, "
    "layout/contiguity). Call write_file once to write diagnosis.md listing every root "
    "cause AND its concrete fix as explicit bullets, using the exact NineToothed terms."
)


def _glm_chat_tools(messages, model, api_key, endpoint, tools, max_tokens, timeout=180):
    """One tool-enabled completion. Returns (assistant_message_dict, p_tok, c_tok)."""
    body = json.dumps({"model": model, "messages": messages, "tools": tools,
                       "temperature": 0.2, "max_tokens": max_tokens}).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, method="POST",
                                 headers={"Authorization": f"Bearer {api_key}",
                                          "Content-Type": "application/json"})
    last_err = ""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
                d = json.load(r)
            msg = d["choices"][0]["message"]
            usage = d.get("usage", {}) or {}
            return msg, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read()[:200].decode(errors='ignore')}"
            if e.code in (400, 401, 403):
                break
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GLM tool chat failed: {last_err}")


def _exec_tool(name, args, ws: pathlib.Path, task_id, ptd) -> str:
    if name == "write_file":
        fn = pathlib.Path(str(args.get("filename", "wrapper.py"))).name
        (ws / fn).write_text(str(args.get("content", "")), encoding="utf-8")
        return f"wrote {fn} ({len(args.get('content',''))} chars)"
    if name == "read_file":
        fn = pathlib.Path(str(args.get("filename", ""))).name
        p = ws / fn
        return p.read_text(encoding="utf-8")[:4000] if p.exists() else f"{fn} not found"
    if name == "run_test":
        if not task_id or not ptd:
            return "run_test unavailable"
        ok, err = _solve_runs(ws, task_id, ptd)
        return "PASS" if ok else f"FAIL: {err}"
    return f"unknown tool {name}"


def make_glm_agent_solver(model=None, api_key=None, endpoint=None,
                          max_new_tokens=4096, max_steps=8):
    api_key = api_key or os.environ.get("GLM_API_KEY")
    if not api_key:
        raise SystemExit("--solver glm needs GLM_API_KEY")
    model = model or os.environ.get("GLM_MODEL", "glm-4.6")
    endpoint = endpoint or os.environ.get("GLM_ENDPOINT", _DEFAULT_ENDPOINT)

    def _solver(prompt: str, ws: pathlib.Path, opts: dict) -> SolverResult:
        t0 = time.time()
        is_operator = opts.get("kind", "operator") == "operator"
        task_id = opts.get("task_id"); ptd = opts.get("proxy_tasks_dir")
        tools = _TOOLS_OPERATOR if is_operator else _TOOLS_DIAG
        system = _AGENT_SYSTEM_OP if is_operator else _AGENT_SYSTEM_DIAG
        skill_ctx = _read_skill_context(ws, prompt)
        user = prompt if not skill_ctx else (
            "Use the following skill guidance verbatim (API, workflow, pitfalls):\n\n"
            + skill_ctx + "\n\n---\n\n" + prompt)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        tin = tout = 0
        passed = False
        for _ in range(max_steps):
            msg, a, b = _glm_chat_tools(messages, model, api_key, endpoint, tools, max_new_tokens)
            tin += a; tout += b
            tool_calls = msg.get("tool_calls") or []
            # record the assistant turn (content may be null when only tool calls)
            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             "tool_calls": tool_calls} if tool_calls
                            else {"role": "assistant", "content": msg.get("content") or ""})
            if not tool_calls:
                break                          # model produced a final answer, no tools
            for tc in tool_calls:
                fn = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = _exec_tool(fn, args, ws, task_id, ptd)
                if fn == "run_test" and result == "PASS":
                    passed = True
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                                 "content": result[:2000]})
            if passed:
                break

        # diagnosis: ensure diagnosis.md exists even if the model answered in plain text
        if not is_operator:
            dm = ws / "diagnosis.md"
            if not dm.exists() or not dm.read_text(encoding="utf-8").strip():
                last = next((m["content"] for m in reversed(messages)
                             if m["role"] == "assistant" and m.get("content")), "")
                dm.write_text((_strip_fences(last) if "```" in last else last).strip() + "\n",
                              encoding="utf-8")
            return SolverResult(tokens_in=tin, tokens_out=tout, wall_seconds=time.time() - t0,
                                compiled=True, compile_attempts=1)

        # operator: final correctness verdict (independent of the model's own claims)
        if not passed:
            ok, _ = _solve_runs(ws, task_id, ptd) if (task_id and ptd) else _import_ok(ws)
            passed = ok
        return SolverResult(tokens_in=tin, tokens_out=tout, wall_seconds=time.time() - t0,
                            gpu_seconds=0.0, compiled=passed, compile_attempts=1,
                            raw={"agentic": True, "model": model})
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
