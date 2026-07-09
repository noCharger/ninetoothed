#!/usr/bin/env python3
"""
attribute.py — turn a generation's failures into ranked, auditable edit proposals.

Version B's mutation step. It reuses the existing deterministic failure_classifier
(Code Repair vs Prompt Repair, from the Ascend slide) to decide WHERE a skill edit
should land, then drafts WHAT the edit says. The draft text is produced by a pluggable
"drafter" — by default an LLM (claude -p) acting as the mutation operator, but any
callable can be injected (offline tests pass a template drafter).

Only guidance_error failures produce skill edits — code_error failures mean the agent
ignored existing correct guidance, so editing the skill wouldn't help (they're logged
for the report but not acted on here). This keeps version B honest: it edits the skill
only when the skill text is the actual gap.

Ranking: proposals are ordered by how many distinct tasks share the symptom (a gap that
breaks many tasks is fixed first), then by classifier confidence.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))          # evaluation/  (for skill_eval)

from edit_ops import EditProposal              # noqa: E402
from skill_eval.failure_classifier import classify, FailureType  # noqa: E402

# family -> reference file the skill uses for that operator family
_FAMILY_FILE = {
    "elementwise": "references/elementwise.md",
    "reduction": "references/reduction.md",
    "layout": "references/layout.md",
    "perf_diag": "references/perf-diag.md",
    "perf-diag": "references/perf-diag.md",
}

# a drafter takes (skill_root, target_file, heading, context) -> new markdown body
Drafter = Callable[[pathlib.Path, str, str, dict], str]


@dataclass
class Failure:
    task_id: str
    family: str
    error_text: str
    oob_count: int = 0


def _target_file(repair_target: str, family: str) -> str:
    if repair_target.startswith("references/"):
        # classifier says references/<family>.md — normalise family spelling
        return _FAMILY_FILE.get(family, repair_target)
    if "SKILL.md" in repair_target:
        return "SKILL.md"
    return _FAMILY_FILE.get(family, "SKILL.md")


def attribute(failures: list[Failure], skill_root: str | pathlib.Path,
              drafter: Optional[Drafter] = None, max_proposals: int = 3) -> list[EditProposal]:
    """
    Classify failures and emit up to max_proposals ranked EditProposals.
    Only guidance_error failures yield proposals.
    """
    skill_root = pathlib.Path(skill_root)
    drafter = drafter or llm_drafter

    # classify, keeping symptom history per family so repeat-detection works
    history: dict[str, list[str]] = defaultdict(list)
    guidance: list[tuple[Failure, object]] = []
    for f in failures:
        clf = classify(error_text=f.error_text, oob_count=f.oob_count,
                       symptom_history=history[f.family], family=f.family)
        history[f.family].append(f.error_text)
        if clf.failure_type == FailureType.GUIDANCE_ERROR:
            guidance.append((f, clf))

    # group by (target_file, repair intent) and rank by count of distinct tasks
    groups: dict[str, list[tuple[Failure, object]]] = defaultdict(list)
    for f, clf in guidance:
        tf = _target_file(clf.repair_target, f.family)
        groups[tf].append((f, clf))

    ranked = sorted(groups.items(), key=lambda kv: -len({fc[0].task_id for fc in kv[1]}))

    proposals: list[EditProposal] = []
    for target_file, items in ranked[:max_proposals]:
        f0, clf0 = items[0]
        n_tasks = len({it[0].task_id for it in items})
        # decide op: a "repeat symptom" gap → reorder the pitfall up AND substitute a
        # sharper warning; a one-off API/idiom gap → substitute the offending example.
        heading = _pick_heading(skill_root, target_file, clf0)
        is_repeat = "repeat" in clf0.matched_rule
        context = {
            "family": f0.family,
            "error_samples": [it[0].error_text[:200] for it in items[:3]],
            "repair_hint": clf0.repair_hint,
            "n_tasks": n_tasks,
            "matched_rule": clf0.matched_rule,
        }
        if is_repeat and heading:
            # promote the pitfall to the top of its section list, then sharpen it
            proposals.append(EditProposal(
                op="reorder", target_file=target_file, heading=heading,
                before_heading=_first_section(skill_root, target_file, heading),
                rationale=f"repeated symptom across {n_tasks} tasks: {clf0.repair_hint}",
                source="attribution",
            ))
        new_body = drafter(skill_root, target_file, heading or "", context)
        proposals.append(EditProposal(
            op="substitute", target_file=target_file, heading=heading or _fallback_heading(skill_root, target_file),
            new_body=new_body,
            rationale=f"{n_tasks} task(s) hit: {clf0.repair_hint}",
            source="attribution+" + ("llm" if drafter is llm_drafter else "template"),
        ))
    return proposals


def _pick_heading(skill_root: pathlib.Path, target_file: str, clf) -> Optional[str]:
    """Find the most relevant heading to edit: prefer a 'Pitfall'/'Common' section."""
    from edit_ops import MarkdownDoc
    fp = skill_root / target_file
    if not fp.exists():
        return None
    doc = MarkdownDoc.parse(fp.read_text(encoding="utf-8"))
    for want in ("pitfall", "common error", "gotcha", "caveat", "note"):
        for h in doc.headings():
            if want in h.lower():
                return h
    hs = doc.headings()
    return hs[-1] if hs else None


def _first_section(skill_root: pathlib.Path, target_file: str, exclude: str) -> Optional[str]:
    from edit_ops import MarkdownDoc
    fp = skill_root / target_file
    doc = MarkdownDoc.parse(fp.read_text(encoding="utf-8"))
    for h in doc.headings():
        if h != exclude:
            return h
    return None


def _fallback_heading(skill_root: pathlib.Path, target_file: str) -> str:
    from edit_ops import MarkdownDoc
    fp = skill_root / target_file
    if fp.exists():
        hs = MarkdownDoc.parse(fp.read_text(encoding="utf-8")).headings()
        if hs:
            return hs[-1]
    return "Pitfalls"


# --------------------------------------------------------------------------- drafters
def template_drafter(skill_root: pathlib.Path, target_file: str, heading: str,
                     context: dict) -> str:
    """Deterministic offline drafter — emits a structured pitfall from the attribution."""
    samples = context.get("error_samples", [])
    hint = context.get("repair_hint", "")
    lines = [f"- **Pitfall ({context.get('family','?')}):** {hint}"]
    if samples:
        lines.append(f"  - Symptom seen: `{samples[0][:120]}`")
    lines.append("  - Follow the workflow above and verify with run_correctness_matrix before benchmarking.")
    return "\n".join(lines)


def llm_drafter(skill_root: pathlib.Path, target_file: str, heading: str,
                context: dict) -> str:
    """LLM mutation operator: ask claude -p to draft a sharper section body.
    Falls back to template_drafter if claude is unavailable."""
    fp = skill_root / target_file
    current = ""
    if fp.exists():
        from edit_ops import MarkdownDoc
        doc = MarkdownDoc.parse(fp.read_text(encoding="utf-8"))
        i = doc.find(heading) if heading else None
        if i is not None:
            current = "\n".join(doc.blocks[i].body)
    prompt = (
        "You are improving a GPU-operator development skill's reference text. "
        f"Operator family: {context.get('family')}. "
        f"The following failures recurred because the guidance was insufficient:\n"
        + "\n".join(f"- {s}" for s in context.get("error_samples", []))
        + f"\n\nRepair hint from the failure classifier: {context.get('repair_hint')}\n\n"
        f"Current section body under heading '{heading}':\n{current}\n\n"
        "Rewrite ONLY this section's body (markdown, no heading line) to prevent the "
        "failures. Be concrete: give the exact correct idiom and a one-line 'why'. "
        "Keep it under 12 lines. Output only the markdown body."
    )
    try:
        proc = subprocess.run(
            ["claude", "-p", "--output-format", "json", prompt],
            capture_output=True, text=True, timeout=300,
        )
        data = json.loads(proc.stdout)
        body = data.get("result", "").strip()
        if body:
            return body
    except Exception:  # noqa: BLE001
        pass
    return template_drafter(skill_root, target_file, heading, context)


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    # offline test with the template drafter against the real skill references
    root = _HERE.parent.parent  # ninetoothed-operator-dev
    fails = [
        Failure("rd03", "reduction", "TypeError: offsets() takes 1 positional argument but 2 given"),
        Failure("rd05", "reduction", "TypeError: offsets() takes 1 positional argument but 2 given"),
        Failure("rd06", "reduction", "AssertionError: allclose failed: NaN in output"),
    ]
    props = attribute(fails, root, drafter=template_drafter)
    print(f"{len(props)} proposal(s):")
    for p in props:
        print("  -", p.one_line())
    assert props, "expected at least one guidance-error proposal"
    print("attribute self-test OK")
