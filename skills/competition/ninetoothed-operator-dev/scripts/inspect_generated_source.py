#!/usr/bin/env python3
"""Inspect the Triton source NineToothed caches at ~/.ninetoothed/<sha256>.py.

Read-only. Parses with `ast` (no execution of the cached code). Reports the
Triton ops used, tile / num_warps / num_stages hints, and load/store counts —
the evidence you cite when judging whether a kernel is reasonable or regressed.

Usage:
    python inspect_generated_source.py                 # newest cached kernel
    python inspect_generated_source.py --digest <hex>  # a specific kernel
    python inspect_generated_source.py --list          # list cached kernels
    python inspect_generated_source.py --cache-dir DIR  # override cache dir
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import sys
from collections import Counter


def default_cache_dir() -> pathlib.Path:
    return pathlib.Path.home() / ".ninetoothed"


def list_cached(cache_dir: pathlib.Path) -> list[pathlib.Path]:
    if not cache_dir.is_dir():
        return []
    return sorted(cache_dir.glob("*.py"), key=lambda p: p.stat().st_mtime, reverse=True)


def pick_file(cache_dir: pathlib.Path, digest: str | None) -> pathlib.Path | None:
    files = list_cached(cache_dir)
    if not files:
        return None
    if digest is None:
        return files[0]
    for f in files:
        if f.stem == digest or f.stem.startswith(digest):
            return f
    return None


class _Analyzer(ast.NodeVisitor):
    """Collect call names, decorator kwargs, and numeric constants."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.kw_consts: dict[str, list] = {}
        self.int_consts: Counter[int] = Counter()

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted_name(node.func)
        if name:
            self.calls[name] += 1
        # capture num_warps=/num_stages=/BLOCK_*=... style kwargs
        for kw in node.keywords:
            if kw.arg and isinstance(kw.value, ast.Constant):
                self.kw_consts.setdefault(kw.arg, []).append(kw.value.value)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            self.int_consts[node.value] += 1
        self.generic_visit(node)


def _dotted_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def analyze(path: pathlib.Path) -> dict:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    a = _Analyzer()
    a.visit(tree)

    tl_ops = {k: v for k, v in a.calls.items() if k.startswith(("tl.", "ntl.", "triton"))}
    loads = sum(v for k, v in a.calls.items() if k.endswith(".load"))
    stores = sum(v for k, v in a.calls.items() if k.endswith(".store"))
    dots = sum(v for k, v in a.calls.items() if k.endswith(".dot"))

    interesting = {}
    for key in ("num_warps", "num_stages"):
        if key in a.kw_consts:
            interesting[key] = a.kw_consts[key]

    return {
        "path": str(path),
        "lines": src.count("\n") + 1,
        "tl_ops": dict(sorted(tl_ops.items(), key=lambda kv: -kv[1])),
        "loads": loads,
        "stores": stores,
        "dots": dots,
        "config_kwargs": interesting,
        "top_int_consts": a.int_consts.most_common(8),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--digest", default=None, help="sha256 (or prefix) of a cached kernel")
    p.add_argument("--cache-dir", default=None, help="override ~/.ninetoothed")
    p.add_argument("--list", action="store_true", help="list cached kernels and exit")
    args = p.parse_args(argv)

    cache_dir = pathlib.Path(args.cache_dir) if args.cache_dir else default_cache_dir()

    if args.list:
        files = list_cached(cache_dir)
        if not files:
            print(f"no cached kernels under {cache_dir}")
            return 1
        for f in files:
            print(f"{f.stem}\t{f.stat().st_size:>8} B\t{f.name}")
        return 0

    target = pick_file(cache_dir, args.digest)
    if target is None:
        print(
            f"no matching cached kernel under {cache_dir} "
            f"(build/run a kernel first, then re-run)",
            file=sys.stderr,
        )
        return 1

    info = analyze(target)
    print(f"# generated source: {info['path']}  ({info['lines']} lines)")
    print(f"loads={info['loads']}  stores={info['stores']}  dots={info['dots']}")
    if info["config_kwargs"]:
        print(f"config kwargs: {info['config_kwargs']}")
    print("triton ops:")
    for op, n in info["tl_ops"].items():
        print(f"  {op:<28} x{n}")
    print(f"top int constants (tile/size hints): {info['top_int_consts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
