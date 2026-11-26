"""Who judges the judge? Compare the LLM judge against human labels.

A calibration file is JSONL, one example per line:
  {"id": "...", "rubric": "...", "prompt": "...", "response": "...", "label": "pass" | "fail", "note": "..."}

Reports accuracy, how many real failures the judge catches (recall), how often its
"fail" is right (precision), Cohen's kappa (agreement beyond chance), and - with repeats -
how often the judge changes its mind about the same input.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from llm_arena.judge import Judge


@dataclass
class Example:
    id: str
    rubric: str
    prompt: str
    response: str
    label: str
    note: str = ""


@dataclass
class Calibration:
    examples: int
    accuracy: float
    fail_recall: float  # of the truly bad responses, how many did the judge catch?
    fail_precision: float  # when the judge says fail, how often is it right?
    kappa: float
    errors: int
    unstable: int  # examples where repeated grading disagreed with itself
    confusion: dict = field(default_factory=dict)
    disagreements: list[dict] = field(default_factory=list)


def load_examples(path: str | Path) -> list[Example]:
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("label") not in ("pass", "fail"):
            raise ValueError(f"{path}:{n}: label must be 'pass' or 'fail'")
        out.append(Example(**{k: d.get(k, "") for k in Example.__dataclass_fields__}))
    return out


def _kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    if not n:
        return 0.0
    observed = sum(a == b for a, b in pairs) / n
    human, judge = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(human[k] * judge[k] for k in ("pass", "fail")) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def calibrate(judge: Judge, examples: list[Example], repeat: int = 1) -> Calibration:
    pairs, errors, unstable, disagreements = [], 0, 0, []
    confusion: Counter[str] = Counter()
    for ex in examples:
        verdicts = [judge.grade(ex.rubric, ex.prompt, ex.response) for _ in range(repeat)]
        labels = [v.verdict for v in verdicts]
        if len(set(labels)) > 1:
            unstable += 1
        graded = [v for v in verdicts if v.verdict != "error"]
        if not graded:
            errors += 1
            disagreements.append({"id": ex.id, "label": ex.label, "judge": "error", "note": ex.note})
            continue
        # Majority vote across repeats; ties go to "fail" (the cautious choice).
        counts = Counter(v.verdict for v in graded)
        final = "fail" if counts["fail"] >= counts["pass"] else "pass"
        pairs.append((ex.label, final))
        confusion[f"human_{ex.label}/judge_{final}"] += 1
        if final != ex.label:
            reason = next(v.reason for v in graded if v.verdict == final)
            disagreements.append({"id": ex.id, "label": ex.label, "judge": final, "reason": reason, "note": ex.note})

    tp = sum(a == b == "fail" for a, b in pairs)
    real_fail = sum(a == "fail" for a, _ in pairs)
    said_fail = sum(b == "fail" for _, b in pairs)
    return Calibration(
        examples=len(examples),
        accuracy=round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else 0.0,
        fail_recall=round(tp / real_fail, 4) if real_fail else 1.0,
        fail_precision=round(tp / said_fail, 4) if said_fail else 1.0,
        kappa=round(_kappa(pairs), 4),
        errors=errors,
        unstable=unstable,
        confusion=dict(confusion),
        disagreements=disagreements,
    )
