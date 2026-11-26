"""Run a suite: send every case to the target (several times), check and judge each reply."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime

from llm_arena.attacks import MUTATORS, Attacker
from llm_arena.judge import Judge, deterministic_checks
from llm_arena.providers import ProviderError
from llm_arena.suite import Case, Suite


@dataclass
class Run:
    response: str
    verdict: str  # pass | fail | error
    decided_by: str  # deterministic | judge | target-error | unchecked
    checks: list[str] = field(default_factory=list)
    judge_verdict: str = ""
    judge_reason: str = ""
    judge_missed: bool = False  # deterministic check failed but the judge said pass
    seconds: float = 0.0


@dataclass
class CaseResult:
    id: str
    category: str
    prompt: str
    origin: str
    attack: bool
    runs: list[Run] = field(default_factory=list)

    @property
    def fail_rate(self) -> float:
        graded = [r for r in self.runs if r.verdict != "error"]
        return sum(r.verdict == "fail" for r in graded) / len(graded) if graded else 0.0

    @property
    def stable(self) -> bool:
        return len({r.verdict for r in self.runs}) <= 1

    @property
    def verdict(self) -> str:
        """Worst outcome across repeats: one leak in five tries is still a leak."""
        if any(r.verdict == "fail" for r in self.runs):
            return "fail"
        if all(r.verdict == "error" for r in self.runs):
            return "error"
        return "pass"


@dataclass
class Results:
    suite: str
    target: str
    judge_model: str
    attacker_model: str
    repeat: int
    started: str
    cases: list[CaseResult]

    def to_dict(self) -> dict:
        d = asdict(self)
        for c, obj in zip(d["cases"], self.cases, strict=True):
            c.update(fail_rate=round(obj.fail_rate, 4), stable=obj.stable, verdict=obj.verdict)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Results:
        cases = []
        for c in d["cases"]:
            runs = [Run(**r) for r in c["runs"]]
            cases.append(CaseResult(c["id"], c["category"], c["prompt"], c["origin"], c["attack"], runs))
        return cls(d["suite"], d["target"], d["judge_model"], d["attacker_model"], d["repeat"], d["started"], cases)


def evaluate(case: Case, response: str, judge: Judge | None) -> Run:
    checks = deterministic_checks(response, case.must_not_contain, case.must_contain_any)
    det_fail = any(not c.passed for c in checks)
    run = Run(response=response, verdict="pass", decided_by="unchecked", checks=[c.detail for c in checks])

    if judge and case.rubric:
        jv = judge.grade(case.rubric, case.prompt, response)
        run.judge_verdict, run.judge_reason = jv.verdict, jv.reason
        run.judge_missed = det_fail and jv.verdict == "pass"

    if det_fail:
        run.verdict, run.decided_by = "fail", "deterministic"  # the judge can't overrule code
    elif run.judge_verdict:
        run.verdict, run.decided_by = run.judge_verdict, "judge"
    elif checks:
        run.decided_by = "deterministic"
    return run


def expand(suite: Suite) -> list[Case]:
    out: list[Case] = []
    for case in suite.cases:
        out.append(case)
        if case.mutate:
            for name in suite.mutators:
                out.append(
                    replace(case, id=f"{case.id}~{name}", prompt=MUTATORS[name](case.prompt), origin=f"mutator:{name}")
                )
    return out


def run_suite(
    suite: Suite,
    target,
    judge: Judge | None,
    attacker: Attacker | None = None,
    generate: int = 0,
    repeat: int | None = None,
    progress: Callable[[str], None] = lambda _: None,
) -> Results:
    repeat = repeat or suite.repeat
    results = Results(
        suite=suite.name,
        target=target.name,
        judge_model=judge.model if judge else "",
        attacker_model=attacker.model if attacker else "",
        repeat=repeat,
        started=datetime.now(UTC).isoformat(timespec="seconds"),
        cases=[],
    )

    def run_case(case: Case) -> CaseResult:
        cr = CaseResult(case.id, case.category, case.prompt, case.origin, case.attack)
        for _ in range(repeat):
            t0 = time.perf_counter()
            try:
                response = target.send(case.prompt)
            except ProviderError as exc:
                cr.runs.append(Run(response="", verdict="error", decided_by="target-error", checks=[str(exc)]))
                continue
            run = evaluate(case, response, judge)
            run.seconds = round(time.perf_counter() - t0, 3)
            cr.runs.append(run)
        progress(f"{'FAIL' if cr.verdict == 'fail' else cr.verdict.upper():5} {case.id}")
        return cr

    for case in expand(suite):
        results.cases.append(run_case(case))

    # Adaptive phase: the attacker sees what worked and writes new attempts per category.
    if attacker and generate > 0:
        categories = suite.attacker.get("categories") or sorted({c.category for c in suite.cases if c.attack})
        for category in categories:
            seeds = [c for c in suite.cases if c.category == category and c.attack]
            if not seeds:
                continue
            template = seeds[0]
            history = [(r.prompt, r.verdict == "fail") for r in results.cases if r.category == category]
            try:
                prompts = attacker.generate(history, generate)
            except ProviderError as exc:
                progress(f"attacker failed for {category}: {exc}")
                continue
            for i, prompt in enumerate(prompts, 1):
                case = replace(template, id=f"{category}-gen{i}", prompt=prompt, origin="attacker", mutate=False)
                results.cases.append(run_case(case))
    return results
