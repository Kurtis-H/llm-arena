"""Markdown reports for runs, calibrations and model comparisons."""

from __future__ import annotations

from collections import defaultdict

from llm_arena.calibrate import Calibration
from llm_arena.runner import Results


def md(text: str, limit: int = 160) -> str:
    """Escape untrusted model output for markdown tables and cap its length."""
    text = " ".join(str(text).split())
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    for a, b in (("\\", "\\\\"), ("|", "\\|"), ("<", "&lt;"), (">", "&gt;"), ("`", "'"), ("[", "\\["), ("]", "\\]")):
        text = text.replace(a, b)
    return text


def summary(r: Results) -> dict:
    attacks = [c for c in r.cases if c.attack and c.verdict != "error"]
    quality = [c for c in r.cases if not c.attack and c.verdict != "error"]
    all_runs = [run for c in r.cases for run in c.runs]
    return {
        "cases": len(r.cases),
        "attack_success_rate": round(sum(c.verdict == "fail" for c in attacks) / len(attacks), 4) if attacks else None,
        "quality_pass_rate": round(sum(c.verdict == "pass" for c in quality) / len(quality), 4) if quality else None,
        "unstable_cases": sum(not c.stable for c in r.cases),
        "errors": sum(c.verdict == "error" for c in r.cases),
        "judge_missed": sum(run.judge_missed for run in all_runs),
        "decided_by_judge": sum(run.decided_by == "judge" for run in all_runs),
        "decided_by_code": sum(run.decided_by == "deterministic" for run in all_runs),
    }


def run_markdown(r: Results) -> str:
    s = summary(r)
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"  # noqa: E731
    lines = [
        f"## 🥊 LLM Arena: {md(r.suite)}",
        "",
        f"Target `{md(r.target)}` · judge `{r.judge_model or 'none'}` · attacker `{r.attacker_model or 'none'}` · "
        f"{r.repeat} run(s) per case · {r.started}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Attack success rate (lower is better) | **{pct(s['attack_success_rate'])}** |",
        f"| Quality pass rate | **{pct(s['quality_pass_rate'])}** |",
        f"| Unstable cases (verdict changed between repeats) | {s['unstable_cases']} |",
        f"| Verdicts decided by code / by judge | {s['decided_by_code']} / {s['decided_by_judge']} |",
        f"| Judge said pass but a deterministic check failed | {s['judge_missed']} |",
        f"| Errors | {s['errors']} |",
        "",
    ]
    by_cat: dict[str, list] = defaultdict(list)
    for c in r.cases:
        by_cat[c.category].append(c)
    lines += ["### By category", "", "| Category | Cases | Failed | Unstable |", "|---|--:|--:|--:|"]
    for cat, cs in sorted(by_cat.items()):
        lines.append(
            f"| {md(cat)} | {len(cs)} | {sum(c.verdict == 'fail' for c in cs)} | {sum(not c.stable for c in cs)} |"
        )
    lines.append("")

    failed = [c for c in r.cases if c.verdict == "fail"]
    if failed:
        lines += ["### Failures", "", "| Case | Origin | Fail rate | Decided by | Prompt | Response (first failing run) |",
                  "|---|---|--:|---|---|---|"]  # fmt: skip
        for c in sorted(failed, key=lambda c: -c.fail_rate):
            bad = next(run for run in c.runs if run.verdict == "fail")
            lines.append(
                f"| `{md(c.id)}` | {c.origin} | {c.fail_rate:.0%} | {bad.decided_by} | {md(c.prompt, 80)} | {md(bad.response)} |"
            )
        lines.append("")
    unstable = [c for c in r.cases if not c.stable]
    if unstable:
        lines += ["### Unstable (same input, different verdicts)", ""]
        lines += [
            f"- `{md(c.id)}`: " + " ".join("❌" if run.verdict == "fail" else "✅" for run in c.runs) for c in unstable
        ]
        lines.append("")
    return "\n".join(lines) + "\n"


def calibration_markdown(c: Calibration, judge_model: str) -> str:
    lines = [
        f"## ⚖️ Judge calibration: `{judge_model}`",
        "",
        "| Metric | Value | Meaning |",
        "|---|---|---|",
        f"| Accuracy | **{c.accuracy:.0%}** | agrees with the human label |",
        f"| Fail recall | **{c.fail_recall:.0%}** | of the bad responses, how many it caught |",
        f"| Fail precision | {c.fail_precision:.0%} | when it says fail, how often it's right |",
        f"| Cohen's kappa | {c.kappa:.2f} | agreement beyond chance (>0.8 strong, <0.4 weak) |",
        f"| Unstable | {c.unstable} of {c.examples} | changed its verdict on the same input |",
        f"| Errors | {c.errors} | no usable verdict |",
        "",
    ]
    if c.disagreements:
        lines += ["### Where the judge disagreed with the human label", "", "| Example | Human | Judge | Judge's reason | Note |",
                  "|---|---|---|---|---|"]  # fmt: skip
        for d in c.disagreements:
            lines.append(
                f"| `{md(d['id'])}` | {d['label']} | {d['judge']} | {md(d.get('reason', ''), 100)} | {md(d.get('note', ''), 100)} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def compare(a: Results, b: Results) -> dict:
    """Regression check between two runs (e.g. model A vs model B) on the same cases."""
    av = {c.id: c for c in a.cases if c.origin != "attacker"}
    bv = {c.id: c for c in b.cases if c.origin != "attacker"}
    shared = sorted(av.keys() & bv.keys())
    return {
        "a": a.target,
        "b": b.target,
        "shared": len(shared),
        "regressions": [i for i in shared if av[i].verdict == "pass" and bv[i].verdict == "fail"],
        "fixes": [i for i in shared if av[i].verdict == "fail" and bv[i].verdict == "pass"],
        "still_failing": [i for i in shared if av[i].verdict == bv[i].verdict == "fail"],
        "only_in_a": sorted(av.keys() - bv.keys()),
        "only_in_b": sorted(bv.keys() - av.keys()),
    }


def compare_markdown(diff: dict) -> str:
    lines = [
        "## 🔀 Model comparison",
        "",
        f"A = `{md(diff['a'])}` → B = `{md(diff['b'])}` · {diff['shared']} shared cases",
        "",
        f"- **Regressions** (passed in A, fail in B): {len(diff['regressions'])}",
        f"- **Fixes** (failed in A, pass in B): {len(diff['fixes'])}",
        f"- **Still failing in both:** {len(diff['still_failing'])}",
        "",
    ]
    for key, title in (("regressions", "Regressions"), ("fixes", "Fixes"), ("still_failing", "Still failing")):
        if diff[key]:
            lines += [f"### {title}", ""] + [f"- `{md(i)}`" for i in diff[key]] + [""]
    return "\n".join(lines) + "\n"
