from __future__ import annotations

import json
from pathlib import Path

from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.schemas import CounterfactualRequest
from diabetify_cf.verification.runner import ScenarioExpectation, VerificationScenario


def load_verification_scenarios(
    path: str | Path,
    *,
    include_tags: tuple[str, ...] = (),
    exclude_tags: tuple[str, ...] = (),
) -> list[VerificationScenario]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"verification scenario path not found: {root}")

    if root.is_file():
        files = [root]
        directory_mode = False
    else:
        files = sorted(root.glob("*.json"))
        directory_mode = True
    scenarios: list[VerificationScenario] = []
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        if directory_mode and not _looks_like_scenario_payload(payload):
            continue
        parsed_scenarios = _parse_scenarios_payload(payload, source=file)
        for scenario in parsed_scenarios:
            if _should_include_scenario(
                scenario,
                include_tags=include_tags,
                exclude_tags=exclude_tags,
            ):
                scenarios.append(scenario)
    return scenarios


def _parse_scenarios_payload(
    payload: dict[str, object],
    *,
    source: Path,
) -> list[VerificationScenario]:
    scenarios_payload = payload.get("scenarios")
    if scenarios_payload is None:
        return [_parse_scenario_payload(payload, source=source)]

    if not isinstance(scenarios_payload, list):
        raise ValueError(
            f"Invalid verification scenario file '{source}': scenarios must be an array"
        )

    scenarios: list[VerificationScenario] = []
    for index, item in enumerate(scenarios_payload):
        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid verification scenario file '{source}': "
                f"scenarios[{index}] must be an object"
            )
        scenarios.append(_parse_scenario_payload(item, source=source))
    return scenarios


def _parse_scenario_payload(payload: dict[str, object], *, source: Path) -> VerificationScenario:
    try:
        name = str(payload["name"])
        request_payload = payload["request"]
        expectation_payload = payload["expectation"]
    except KeyError as err:
        raise ValueError(f"Invalid verification scenario file '{source}': missing {err}") from err

    if not isinstance(request_payload, dict):
        raise ValueError(
            f"Invalid verification scenario file '{source}': request must be an object"
        )
    if not isinstance(expectation_payload, dict):
        raise ValueError(
            f"Invalid verification scenario file '{source}': expectation must be an object"
        )

    request = CounterfactualRequest.model_validate(request_payload)
    expectation = _parse_expectation(expectation_payload, source=source)
    repeat_count = int(payload.get("repeat_count", 1))
    description = str(payload.get("description", ""))
    tags_raw = payload.get("tags", [])
    if not isinstance(tags_raw, list):
        raise ValueError(f"Invalid verification scenario file '{source}': tags must be an array")

    return VerificationScenario(
        name=name,
        request=request,
        expectation=expectation,
        repeat_count=repeat_count,
        description=description,
        tags=tuple(str(tag) for tag in tags_raw),
    )


def _parse_expectation(payload: dict[str, object], *, source: Path) -> ScenarioExpectation:
    status_raw = payload.get("expected_status")
    if not isinstance(status_raw, str):
        raise ValueError(
            f"Invalid verification scenario file '{source}': expected_status must be a string"
        )

    reason_codes_raw = payload.get("expected_reason_codes", [])
    if not isinstance(reason_codes_raw, list):
        raise ValueError(
            f"Invalid verification scenario file '{source}': expected_reason_codes must be an array"
        )

    return ScenarioExpectation(
        expected_status=Status(status_raw),
        expected_reason_codes=tuple(ReasonCode(item) for item in reason_codes_raw),
        no_solution_expected=bool(payload.get("no_solution_expected", False)),
    )


def _looks_like_scenario_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        "name" in payload and "request" in payload and "expectation" in payload
    ) or "scenarios" in payload


def _should_include_scenario(
    scenario: VerificationScenario,
    *,
    include_tags: tuple[str, ...],
    exclude_tags: tuple[str, ...],
) -> bool:
    scenario_tags = set(scenario.tags)
    normalized_include = {tag for tag in include_tags if tag}
    normalized_exclude = {tag for tag in exclude_tags if tag}

    if normalized_include and not normalized_include.issubset(scenario_tags):
        return False
    if normalized_exclude.intersection(scenario_tags):
        return False
    return True
