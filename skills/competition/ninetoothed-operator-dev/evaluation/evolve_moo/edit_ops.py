#!/usr/bin/env python3
"""
edit_ops.py — the three skill-text edit operators for the light-MOO loop (version B),
plus the markdown section model they operate on and the pass-preservation guard.

The skill is a text policy. Version B evolves it with a deliberately small, auditable
operator set (SkillMOO-style), so every generation's diff is reviewable:

    prune       : delete a section (redundant / misleading guidance).
    substitute  : replace a section's body with revised text.
    reorder     : move a section to just before another (e.g. promote a
                  frequently-hit pitfall to the top of a Pitfalls list).

All operators work on a heading-delimited block model of a markdown file and are
pure (return new text; never write in place unless you ask them to via apply_to_file).

Pass-preservation guard: after an edit, the loop re-evaluates train tasks; the guard
REJECTS the edit if it causes any negative transfer (a task that regresses) or lowers
the aggregate train total. This is the proposal's safety rule against skill edits that
help one task while breaking another.
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field
from typing import Optional

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Block:
    level: int              # 0 = preamble (before first heading)
    heading: str            # "" for preamble
    body: list[str] = field(default_factory=list)   # lines AFTER the heading line

    def render(self) -> str:
        out = []
        if self.level > 0:
            out.append(f"{'#' * self.level} {self.heading}")
        out.extend(self.body)
        return "\n".join(out)


@dataclass
class MarkdownDoc:
    blocks: list[Block]

    @classmethod
    def parse(cls, text: str) -> "MarkdownDoc":
        blocks: list[Block] = [Block(level=0, heading="", body=[])]
        for line in text.splitlines():
            m = _HEADING_RE.match(line)
            if m:
                blocks.append(Block(level=len(m.group(1)), heading=m.group(2).strip(), body=[]))
            else:
                blocks[-1].body.append(line)
        # drop an empty preamble
        if blocks and blocks[0].level == 0 and not any(l.strip() for l in blocks[0].body):
            blocks = blocks[1:]
        return cls(blocks)

    def render(self) -> str:
        return "\n".join(b.render() for b in self.blocks).rstrip() + "\n"

    def find(self, heading_substr: str) -> Optional[int]:
        """Index of the first block whose heading contains the substring (case-insensitive)."""
        key = heading_substr.lower()
        for i, b in enumerate(self.blocks):
            if b.level > 0 and key in b.heading.lower():
                return i
        return None

    def headings(self) -> list[str]:
        return [b.heading for b in self.blocks if b.level > 0]


# --------------------------------------------------------------------------- operators
@dataclass
class EditProposal:
    op: str                       # "prune" | "substitute" | "reorder"
    target_file: str              # path relative to skill root, e.g. "references/reduction.md"
    heading: str                  # heading (substring) the op targets
    new_body: Optional[str] = None      # for substitute
    before_heading: Optional[str] = None  # for reorder
    rationale: str = ""           # why (from failure attribution)
    source: str = ""              # "attribution" | "llm" | "manual"

    def one_line(self) -> str:
        tgt = f"{self.target_file}#{self.heading}"
        extra = ""
        if self.op == "reorder" and self.before_heading:
            extra = f" -> before '{self.before_heading}'"
        return f"{self.op} {tgt}{extra} :: {self.rationale[:60]}"


class EditError(ValueError):
    pass


def prune(doc: MarkdownDoc, heading: str) -> MarkdownDoc:
    i = doc.find(heading)
    if i is None:
        raise EditError(f"prune: heading '{heading}' not found")
    return MarkdownDoc(doc.blocks[:i] + doc.blocks[i + 1:])


def substitute(doc: MarkdownDoc, heading: str, new_body: str) -> MarkdownDoc:
    i = doc.find(heading)
    if i is None:
        raise EditError(f"substitute: heading '{heading}' not found")
    blocks = list(doc.blocks)
    old = blocks[i]
    blocks[i] = Block(level=old.level, heading=old.heading,
                      body=[""] + new_body.rstrip().splitlines() + [""])
    return MarkdownDoc(blocks)


def reorder(doc: MarkdownDoc, heading: str, before_heading: str) -> MarkdownDoc:
    i = doc.find(heading)
    if i is None:
        raise EditError(f"reorder: heading '{heading}' not found")
    blocks = list(doc.blocks)
    moving = blocks.pop(i)
    j = None
    key = before_heading.lower()
    for k, b in enumerate(blocks):
        if b.level > 0 and key in b.heading.lower():
            j = k
            break
    if j is None:
        raise EditError(f"reorder: anchor '{before_heading}' not found")
    blocks.insert(j, moving)
    return MarkdownDoc(blocks)


def apply_proposal(skill_root: str | pathlib.Path, prop: EditProposal,
                   write: bool = False) -> str:
    """Apply one proposal to its target file. Returns the new file text.
    If write=True, the file is overwritten (used inside the loop on a candidate copy)."""
    skill_root = pathlib.Path(skill_root)
    fpath = skill_root / prop.target_file
    if not fpath.exists():
        raise EditError(f"target file not found: {fpath}")
    doc = MarkdownDoc.parse(fpath.read_text(encoding="utf-8"))
    if prop.op == "prune":
        new = prune(doc, prop.heading)
    elif prop.op == "substitute":
        if prop.new_body is None:
            raise EditError("substitute requires new_body")
        new = substitute(doc, prop.heading, prop.new_body)
    elif prop.op == "reorder":
        if not prop.before_heading:
            raise EditError("reorder requires before_heading")
        new = reorder(doc, prop.heading, prop.before_heading)
    else:
        raise EditError(f"unknown op {prop.op!r}")
    text = new.render()
    if write:
        fpath.write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------- guard
@dataclass
class GuardVerdict:
    accepted: bool
    reason: str
    train_total_before: int
    train_total_after: int
    regressions: list[str] = field(default_factory=list)   # task_ids that got worse


def pass_preservation_guard(before: dict, after: dict,
                            allow_flat: bool = True) -> GuardVerdict:
    """
    before/after: {task_id: rubric_total} for the train set.
    Reject if any task regresses (negative transfer) or the aggregate total drops.
    allow_flat=True accepts an edit that keeps the total equal (lets neutral
    restructurings through so later edits can build on them).
    """
    tb = sum(before.values())
    ta = sum(after.values())
    regressions = [tid for tid in before if after.get(tid, before[tid]) < before[tid]]
    if regressions:
        return GuardVerdict(False, f"negative transfer on {regressions}", tb, ta, regressions)
    if ta < tb:
        return GuardVerdict(False, f"aggregate train total dropped {tb}->{ta}", tb, ta, [])
    if ta == tb and not allow_flat:
        return GuardVerdict(False, f"no improvement ({tb})", tb, ta, [])
    return GuardVerdict(True, f"train total {tb}->{ta}, no regression", tb, ta, [])


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    sample = """# Guide

Intro line.

## Setup

do setup.

## Pitfalls

- watch out for X.

## Advanced

deep stuff.
"""
    doc = MarkdownDoc.parse(sample)
    assert doc.headings() == ["Guide", "Setup", "Pitfalls", "Advanced"], doc.headings()
    assert MarkdownDoc.parse(prune(doc, "Advanced").render()).headings() == ["Guide", "Setup", "Pitfalls"]
    sub = substitute(doc, "Pitfalls", "- watch out for X.\n- and also Y (fp16 upcast).")
    assert "also Y" in sub.render()
    reo = reorder(doc, "Pitfalls", "Setup")
    assert reo.headings() == ["Guide", "Pitfalls", "Setup", "Advanced"], reo.headings()
    v = pass_preservation_guard({"t1": 8, "t2": 7}, {"t1": 9, "t2": 7})
    assert v.accepted, v
    v2 = pass_preservation_guard({"t1": 8, "t2": 7}, {"t1": 9, "t2": 5})
    assert not v2.accepted and v2.regressions == ["t2"], v2
    print("edit_ops self-test OK")
