#!/usr/bin/env python3
"""
oracle.py — harness-supplied, tamper-proof correctness test for operator tasks.

Trusting the agent's own test_correctness.py for the completion score is a
reward-hacking hole: a weak model can emit `assert True`. So for operator tasks the
harness writes its OWN oracle test, generated from the proxy task's PyTorch reference
and input generator, that calls a FIXED entry point in the agent's wrapper (`solve`).
This test — not the agent's — drives the completion sub-score.

The oracle compares agent output vs the reference across the task's dtypes using the
same MERE/MARE thresholds as scripts/run_correctness_matrix.py, so numbers are
consistent with the skill's own tooling.

The agent still writes its own test (scored under the separate 'test & verification'
sub-score), but it cannot inflate 'completion'.
"""
from __future__ import annotations

import pathlib
import textwrap

# canonical entry point the agent must expose in wrapper.py
ENTRY = "solve"

_ORACLE_TEMPLATE = '''\
# AUTO-GENERATED oracle test — do not edit. Correctness of the agent's `{entry}`
# against the proxy task reference. Independent of the agent's own test.
import sys, itertools, pathlib
import pytest, torch

_PROXY = pathlib.Path({proxy_tasks_dir!r})
sys.path.insert(0, str(_PROXY))
sys.path.insert(0, str(pathlib.Path(__file__).parent))   # for wrapper.py

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")

_MERE_THRESH = {{"float32": 1.22e-4, "float16": 9.77e-4, "bfloat16": 7.81e-3}}

def _load_task():
    from loader import load_all
    for t in load_all():
        if t.id == {task_id!r}:
            return t
    raise RuntimeError("task {task_id} not found")

def _mere_mare(got, ref):
    got = got.float(); ref = ref.float()
    eps = 1e-8
    rel = (got - ref).abs() / (ref.abs() + eps)
    return rel.mean().item(), rel.max().item()

@pytest.mark.parametrize("dtype", {dtypes!r})
def test_oracle(dtype):
    task = _load_task()
    from wrapper import {entry} as _solve
    inputs = task.make_inputs(device="cuda", dtype=dtype)
    ref = task.reference(*[x.clone() for x in inputs])
    got = _solve(*[x.clone() for x in inputs])
    assert torch.is_tensor(got), "solve did not return a tensor"
    assert got.shape == ref.shape, f"shape {{got.shape}} != {{ref.shape}}"
    thr = _MERE_THRESH.get(dtype, 1e-3)
    mere, mare = _mere_mare(got, ref)
    assert mere < thr and mare < 10 * thr, f"MERE={{mere:.2e}} MARE={{mare:.2e}} thr={{thr:.2e}}"
'''


def write_oracle_test(task_meta: dict, workspace: str | pathlib.Path,
                      proxy_tasks_dir: str | pathlib.Path) -> pathlib.Path:
    """Write oracle_test.py into the workspace. Returns its path. Operator tasks only."""
    ws = pathlib.Path(workspace)
    dtypes = list(task_meta.get("dtypes", ["float32", "float16"]))
    src = _ORACLE_TEMPLATE.format(
        entry=ENTRY, task_id=task_meta["id"], dtypes=dtypes,
        proxy_tasks_dir=str(proxy_tasks_dir),
    )
    path = ws / "oracle_test.py"
    path.write_text(src, encoding="utf-8")
    return path


if __name__ == "__main__":
    import tempfile
    ws = pathlib.Path(tempfile.mkdtemp())
    p = write_oracle_test({"id": "ew06", "dtypes": ["float32", "float16"]}, ws, "/tmp/proxy")
    txt = p.read_text()
    assert "test_oracle" in txt and "ew06" in txt and "solve" in txt
    compile(txt, str(p), "exec")   # must be valid python
    print("oracle self-test OK:", p)
