"""`arena run`, `arena calibrate`, `arena compare`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from llm_arena import __version__
from llm_arena.attacks import Attacker
from llm_arena.calibrate import calibrate, load_examples
from llm_arena.judge import Judge
from llm_arena.providers import OllamaProvider, ProviderError, RecordingProvider
from llm_arena.report import calibration_markdown, compare, compare_markdown, run_markdown, summary
from llm_arena.runner import Results, run_suite
from llm_arena.suite import SuiteError, load_suite
from llm_arena.targets import HttpTarget, ModelTarget

EXIT_OK, EXIT_GATE, EXIT_USAGE = 0, 1, 2


def _provider(args):
    if args.provider == "fake":
        from llm_arena.fakes import FakeProvider

        base = FakeProvider()
    else:
        base = OllamaProvider(args.ollama_url)
    if args.replay:
        return RecordingProvider(args.replay, "replay")
    if args.record:
        return RecordingProvider(args.record, "record", base)
    return base


def _write(path: str | None, text: str) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
        summary_file = os.getenv("GITHUB_STEP_SUMMARY")
        if summary_file and path.endswith(".md"):
            with open(summary_file, "a", encoding="utf-8") as fh:
                fh.write(text)


def cmd_run(args) -> int:
    suite = load_suite(args.suite)
    provider = _provider(args)
    t = suite.target
    target: ModelTarget | HttpTarget
    if t["type"] == "model":
        target = ModelTarget(
            provider, args.target_model or t["model"], suite.system_prompt, float(t.get("temperature", 0.7))
        )
    else:
        target = HttpTarget(t["url"], t.get("message_field", "message"), t.get("reply_field", "reply"))
    judge_model = args.judge_model or suite.judge_model
    judge = Judge(provider, judge_model) if judge_model and not args.no_judge else None
    attacker_model = args.attacker_model or suite.attacker.get("model", "")
    generate = args.generate if args.generate is not None else int(suite.attacker.get("generate", 0))
    attacker = (
        Attacker(provider, attacker_model, suite.attacker.get("goal", "")) if attacker_model and generate else None
    )

    results = run_suite(
        suite, target, judge, attacker, generate, args.repeat, progress=lambda m: print(m, file=sys.stderr)
    )
    if isinstance(provider, RecordingProvider):
        provider.save()

    data = results.to_dict()
    data["summary"] = s = summary(results)
    report = run_markdown(results)
    _write(args.out, json.dumps(data, indent=2))
    _write(args.md, report)
    if not args.md:
        print(report)

    if args.replay and s["errors"]:
        print(
            f"arena: {s['errors']} case(s) had no recorded response - the cassette is out of date "
            "for this suite. Re-record it with --record.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    gate = []
    if args.max_asr is not None and (s["attack_success_rate"] or 0) > args.max_asr:
        gate.append(f"attack success rate {s['attack_success_rate']:.0%} > {args.max_asr:.0%}")
    if (
        args.min_quality is not None
        and s["quality_pass_rate"] is not None
        and s["quality_pass_rate"] < args.min_quality
    ):
        gate.append(f"quality pass rate {s['quality_pass_rate']:.0%} < {args.min_quality:.0%}")
    for g in gate:
        print(f"arena: gate failed: {g}", file=sys.stderr)
    return EXIT_GATE if gate else EXIT_OK


def cmd_calibrate(args) -> int:
    provider = _provider(args)
    examples = load_examples(args.file)
    result = calibrate(Judge(provider, args.judge_model), examples, args.repeat)
    if isinstance(provider, RecordingProvider):
        provider.save()
    report = calibration_markdown(result, args.judge_model)
    _write(args.md, report)
    _write(args.out, json.dumps(result.__dict__, indent=2))
    if not args.md:
        print(report)
    if args.min_accuracy is not None and result.accuracy < args.min_accuracy:
        print(f"arena: judge accuracy {result.accuracy:.0%} < {args.min_accuracy:.0%}", file=sys.stderr)
        return EXIT_GATE
    return EXIT_OK


def cmd_compare(args) -> int:
    a = Results.from_dict(json.loads(Path(args.a).read_text(encoding="utf-8")))
    b = Results.from_dict(json.loads(Path(args.b).read_text(encoding="utf-8")))
    diff = compare(a, b)
    report = compare_markdown(diff)
    _write(args.md, report)
    if not args.md:
        print(report)
    return EXIT_GATE if args.fail_on_regression and diff["regressions"] else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--provider", choices=["ollama", "fake"], default="ollama")
    common.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    rec = common.add_mutually_exclusive_group()
    rec.add_argument("--record", metavar="CASSETTE", help="save every model response to this file")
    rec.add_argument("--replay", metavar="CASSETTE", help="replay saved responses instead of calling models")

    p = argparse.ArgumentParser(prog="arena", description="Test AI with AI: attacker, target and judge models.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", parents=[common], help="run a test suite against a target")
    r.add_argument("suite")
    r.add_argument("--repeat", type=int, help="runs per case (default: suite setting)")
    r.add_argument("--generate", type=int, help="new attacks per category from the attacker model")
    r.add_argument("--target-model")
    r.add_argument("--judge-model")
    r.add_argument("--attacker-model")
    r.add_argument("--no-judge", action="store_true", help="deterministic checks only")
    r.add_argument("--out", help="results JSON")
    r.add_argument("--md", help="markdown report")
    r.add_argument("--max-asr", type=float, help="fail if attack success rate is above this (0-1)")
    r.add_argument("--min-quality", type=float, help="fail if quality pass rate is below this (0-1)")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("calibrate", parents=[common], help="measure the judge against human labels")
    c.add_argument("file")
    c.add_argument("--judge-model", required=True)
    c.add_argument("--repeat", type=int, default=1)
    c.add_argument("--out")
    c.add_argument("--md")
    c.add_argument("--min-accuracy", type=float)
    c.set_defaults(func=cmd_calibrate)

    m = sub.add_parser("compare", help="regressions between two result files")
    m.add_argument("a")
    m.add_argument("b")
    m.add_argument("--md")
    m.add_argument("--fail-on-regression", action="store_true")
    m.set_defaults(func=cmd_compare)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except (SuiteError, ProviderError, ValueError, OSError) as exc:
        print(f"arena: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
