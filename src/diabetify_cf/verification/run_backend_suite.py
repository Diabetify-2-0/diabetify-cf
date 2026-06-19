from __future__ import annotations

import argparse
from pathlib import Path

from diabetify_cf.config import Settings
from diabetify_cf.verification import (
    BackendCounterfactualEngineAdapter,
    ExternalCounterfactualVerifier,
    HttpBackendCounterfactualGateway,
    ScenarioRunner,
    wait_for_backend_health,
    write_report_json,
)
from diabetify_cf.verification.suites import (
    build_backend_suite_index,
    build_suite_payload,
    load_suite_scenarios,
    select_verification_suites,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple tagged backend verification suites and write one JSON "
            "report per suite plus an index manifest."
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
        "--suite",
        action="append",
        default=[],
        help=(
            "Verification suite to run. Can be provided multiple times. "
            "Defaults to all predefined backend suites."
        ),
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
        "--output-dir",
        default=str(Path("artifacts") / "verification" / "backend_suites"),
        help="Directory where per-suite JSON reports and the suite index JSON will be written.",
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
    suites = select_verification_suites(tuple(args.suite))
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

    output_dir = Path(args.output_dir)
    suite_index_rows: list[dict[str, object]] = []

    for suite in suites:
        scenarios = load_suite_scenarios(args.scenarios, suite)
        aggregates = runner.run(scenarios)
        summary = runner.summarize(aggregates)
        payload = build_suite_payload(
            suite=suite,
            aggregates=aggregates,
            summary=summary,
            metadata={
                "runner_mode": "backend_suite_member",
                "backend_base_url": args.backend_base_url,
                "health_check_enabled": not args.skip_health_check,
                "health_payload": health_payload,
            },
        )
        suite_report_path = output_dir / f"{suite.name}.json"
        write_report_json(path=suite_report_path, payload=payload)
        suite_index_rows.append(
            {
                "suite_name": suite.name,
                "scenario_count": summary.total_scenarios,
                "total_runs": summary.total_runs,
                "passed": all(aggregate.passed for aggregate in aggregates),
                "report_path": str(suite_report_path),
            }
        )

    index_payload = build_backend_suite_index(
        backend_base_url=args.backend_base_url,
        suite_reports=suite_index_rows,
        output_dir=output_dir,
    )
    write_report_json(path=output_dir / "index.json", payload=index_payload)


if __name__ == "__main__":
    main()
