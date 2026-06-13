from __future__ import annotations

import argparse
from pathlib import Path

from diabetify_cf.config import Settings
from diabetify_cf.verification import (
    BackendCounterfactualEngineAdapter,
    ExternalCounterfactualVerifier,
    HttpBackendCounterfactualGateway,
    ScenarioRunner,
    build_report_payload,
    load_verification_scenarios,
    wait_for_backend_health,
    write_report_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run production verification scenarios through the authenticated "
            "Diabetify backend async counterfactual flow."
        )
    )
    parser.add_argument(
        "--scenarios",
        required=True,
        help="Path to a verification scenario JSON file or a directory of JSON files.",
    )
    parser.add_argument(
        "--backend-base-url",
        required=True,
        help="Backend base URL, for example http://localhost:8080.",
    )
    parser.add_argument(
        "--backend-bearer-token",
        default=None,
        help="Bearer token for authenticated backend counterfactual routes.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=1.0,
        help="Polling interval in seconds for backend job status checks.",
    )
    parser.add_argument(
        "--poll-timeout-seconds",
        type=float,
        default=300.0,
        help="Maximum total polling time in seconds before the scenario is marked as timed out.",
    )
    parser.add_argument(
        "--output",
        default=str(Path("artifacts") / "verification" / "backend_verification_report.json"),
        help="Path to write the backend verification report JSON.",
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
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip backend /counterfactual/health readiness preflight.",
    )
    parser.add_argument(
        "--health-timeout-seconds",
        type=float,
        default=60.0,
        help="Maximum time in seconds to wait for backend health readiness.",
    )
    parser.add_argument(
        "--health-poll-interval-seconds",
        type=float,
        default=2.0,
        help="Polling interval in seconds for backend health readiness checks.",
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
    gateway = HttpBackendCounterfactualGateway(
        base_url=args.backend_base_url,
        bearer_token=args.backend_bearer_token,
    )
    health_payload: dict[str, object] | None = None
    if not args.skip_health_check:
        health_payload = wait_for_backend_health(
            gateway,
            timeout_seconds=args.health_timeout_seconds,
            poll_interval_seconds=args.health_poll_interval_seconds,
        )
    engine = BackendCounterfactualEngineAdapter(
        gateway=gateway,
        poll_interval_seconds=args.poll_interval_seconds,
        poll_timeout_seconds=args.poll_timeout_seconds,
    )
    verifier = ExternalCounterfactualVerifier(settings=settings)
    runner = ScenarioRunner(engine=engine, verifier=verifier)
    aggregates = runner.run(scenarios)
    summary = runner.summarize(aggregates)
    payload = build_report_payload(
        aggregates=aggregates,
        summary=summary,
        metadata={
            "runner_mode": "backend",
            "backend_base_url": args.backend_base_url,
            "include_tags": list(args.include_tag),
            "exclude_tags": list(args.exclude_tag),
            "health_check_enabled": not args.skip_health_check,
            "health_payload": health_payload,
        },
    )
    write_report_json(path=args.output, payload=payload)


if __name__ == "__main__":
    main()
