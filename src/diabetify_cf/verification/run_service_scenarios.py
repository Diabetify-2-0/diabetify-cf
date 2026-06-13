from __future__ import annotations

import argparse
from pathlib import Path

from diabetify_cf.config import Settings
from diabetify_cf.engine import build_counterfactual_engine
from diabetify_cf.verification import (
    ExternalCounterfactualVerifier,
    ScenarioRunner,
    build_report_payload,
    load_verification_scenarios,
    write_report_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run production verification scenarios against the diabetify-cf service engine."
    )
    parser.add_argument(
        "--scenarios",
        required=True,
        help="Path to a verification scenario JSON file or a directory of JSON files.",
    )
    parser.add_argument(
        "--output",
        default=str(Path("artifacts") / "verification" / "service_verification_report.json"),
        help="Path to write the verification report JSON.",
    )
    parser.add_argument(
        "--include-tag",
        action="append",
        default=[],
        help="Include only scenarios that contain this tag. Can be provided multiple times.",
    )
    parser.add_argument(
        "--exclude-tag",
        action="append",
        default=[],
        help="Exclude scenarios that contain this tag. Can be provided multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    scenarios = load_verification_scenarios(
        args.scenarios,
        include_tags=tuple(args.include_tag),
        exclude_tags=tuple(args.exclude_tag),
    )
    engine = build_counterfactual_engine(settings)
    verifier = ExternalCounterfactualVerifier(settings=settings)
    runner = ScenarioRunner(engine=engine, verifier=verifier)
    aggregates = runner.run(scenarios)
    summary = runner.summarize(aggregates)
    payload = build_report_payload(aggregates=aggregates, summary=summary)
    write_report_json(path=args.output, payload=payload)


if __name__ == "__main__":
    main()
