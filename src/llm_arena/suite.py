"""Load a test suite from TOML. See suites/ for examples."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from llm_arena.attacks import MUTATORS


class SuiteError(Exception):
    pass


@dataclass
class Case:
    id: str
    category: str
    prompt: str
    must_not_contain: list[str] = field(default_factory=list)
    must_contain_any: list[str] = field(default_factory=list)
    rubric: str = ""
    attack: bool = True  # counts toward the attack success rate
    mutate: bool = False  # also run every mutator on this prompt
    origin: str = "seed"


@dataclass
class Suite:
    name: str
    target: dict
    cases: list[Case]
    judge_model: str = ""
    attacker: dict = field(default_factory=dict)
    mutators: list[str] = field(default_factory=list)
    repeat: int = 1
    system_prompt: str = ""


def load_suite(path: str | Path) -> Suite:
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SuiteError(f"{path}: {exc}") from exc

    target = data.get("target") or {}
    if target.get("type") not in ("model", "http"):
        raise SuiteError(f"{path}: [target] type must be 'model' or 'http'")
    system_prompt = ""
    if target["type"] == "model":
        if not target.get("model"):
            raise SuiteError(f"{path}: model target needs 'model'")
        spf = target.get("system_prompt_file")
        system_prompt = (path.parent / spf).read_text(encoding="utf-8") if spf else target.get("system_prompt", "")
    elif not target.get("url"):
        raise SuiteError(f"{path}: http target needs 'url'")

    mutators = data.get("mutators", [])
    unknown = [m for m in mutators if m not in MUTATORS]
    if unknown:
        raise SuiteError(f"{path}: unknown mutators {unknown}; available: {sorted(MUTATORS)}")

    cases, ids = [], set()
    for i, c in enumerate(data.get("case", []), 1):
        for key in ("id", "category", "prompt"):
            if not c.get(key):
                raise SuiteError(f"{path}: case #{i} needs '{key}'")
        if c["id"] in ids:
            raise SuiteError(f"{path}: duplicate case id {c['id']!r}")
        ids.add(c["id"])
        if not (c.get("must_not_contain") or c.get("must_contain_any") or c.get("rubric")):
            raise SuiteError(
                f"{path}: case {c['id']!r} has no checks (add must_not_contain, must_contain_any or rubric)"
            )
        cases.append(
            Case(
                id=c["id"],
                category=c["category"],
                prompt=c["prompt"],
                must_not_contain=list(c.get("must_not_contain", [])),
                must_contain_any=list(c.get("must_contain_any", [])),
                rubric=c.get("rubric", ""),
                attack=bool(c.get("attack", True)),
                mutate=bool(c.get("mutate", False)),
            )
        )
    if not cases:
        raise SuiteError(f"{path}: no [[case]] entries")

    return Suite(
        name=data.get("name", path.stem),
        target=target,
        cases=cases,
        judge_model=(data.get("judge") or {}).get("model", ""),
        attacker=data.get("attacker") or {},
        mutators=mutators,
        repeat=int(data.get("repeat", 1)),
        system_prompt=system_prompt,
    )
