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
from diabetify_cf.verification.launcher import (
    LauncherConfig,
    load_launcher_config,
    resolve_backend_bearer_token,
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
            "Run backend verification suites using a single JSON launcher config "
            "that can provide either a bearer token or login credentials."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the backend suite launcher JSON config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_launcher_config(args.config)
    run_backend_suite_launcher(config)


def run_backend_suite_launcher(config: LauncherConfig) -> None:
    settings = Settings()
    bearer_token = resolve_backend_bearer_token(config)
    suites = select_verification_suites(config.suites)
    gateway = HttpBackendCounterfactualGateway(
        base_url=config.backend_base_url,
        bearer_token=bearer_token,
    )

    health_payload: dict[str, object] | None = None
    if not config.skip_health_check:
        health_payload = wait_for_backend_health(
            gateway,
            timeout_seconds=config.health_timeout_seconds,
            poll_interval_seconds=config.health_poll_interval_seconds,
        )

    engine = BackendCounterfactualEngineAdapter(
        gateway=gateway,
        poll_interval_seconds=config.poll_interval_seconds,
        poll_timeout_seconds=config.poll_timeout_seconds,
    )
    verifier = ExternalCounterfactualVerifier(settings=settings)
    runner = ScenarioRunner(engine=engine, verifier=verifier)

    output_dir = Path(config.output_dir)
    suite_index_rows: list[dict[str, object]] = []
    all_aggregates = []

    for suite in suites:
        scenarios = load_suite_scenarios(config.scenarios_path, suite)
        aggregates = runner.run(scenarios)
        all_aggregates.extend(aggregates)
        summary = runner.summarize(aggregates)
        payload = build_suite_payload(
            suite=suite,
            aggregates=aggregates,
            summary=summary,
            metadata={
                "runner_mode": "backend_suite_member",
                "launcher_mode": "config",
                "backend_base_url": config.backend_base_url,
                "health_check_enabled": not config.skip_health_check,
                "health_payload": health_payload,
                "auth_mode": config.auth_mode,
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
        backend_base_url=config.backend_base_url,
        suite_reports=suite_index_rows,
        output_dir=output_dir,
        overall_summary=runner.summarize(all_aggregates),
    )
    write_report_json(path=output_dir / "index.json", payload=index_payload)


if __name__ == "__main__":
    main()
