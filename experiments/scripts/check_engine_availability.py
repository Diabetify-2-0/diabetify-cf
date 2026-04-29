from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _bootstrap_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"
    for path in (repo_root, src_path):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_path()

from diabetify_cf.config import Settings  # noqa: E402
from experiments.engines.dice_adapter import DiceExperimentAdapter  # noqa: E402

ENGINE_IMPORTS = {
    "dice": ["dice_ml"],
    "carla": ["carla"],
    "ocean": ["ocean"],
    "focus": ["focus"],
}


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def check_availability(settings: Settings) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for engine_name, module_names in ENGINE_IMPORTS.items():
        missing = [name for name in module_names if not _module_available(name)]
        rows.append(
            {
                "engine": engine_name,
                "available": not missing,
                "missing_modules": missing,
            }
        )

    dice_row = next(row for row in rows if row["engine"] == "dice")
    if dice_row["available"]:
        adapter = DiceExperimentAdapter(settings=settings)
        dice_row["artifact_ready"] = adapter.engine.artifacts is not None
        dice_row["initialization_error"] = adapter.engine.initialization_error

    rows.append(
        {
            "engine": "artifacts",
            "available": True,
            "model_path_exists": Path(settings.model_path).exists(),
            "columns_path_exists": Path(settings.columns_path).exists(),
            "reference_data_path_exists": Path(settings.reference_data_path).exists(),
            "feature_registry_path_exists": Path(settings.feature_registry_path).exists(),
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Check counterfactual engine availability.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    rows = check_availability(Settings())
    if args.json:
        print(json.dumps(rows, indent=2))
        return

    for row in rows:
        if row["engine"] == "artifacts":
            print(
                "artifacts: "
                f"model={row['model_path_exists']} "
                f"columns={row['columns_path_exists']} "
                f"reference={row['reference_data_path_exists']} "
                f"registry={row['feature_registry_path_exists']}"
            )
            continue
        state = "available" if row["available"] else "missing"
        detail = ""
        if row.get("missing_modules"):
            detail = f" missing={','.join(row['missing_modules'])}"
        if "artifact_ready" in row:
            detail += f" artifact_ready={row['artifact_ready']}"
        print(f"{row['engine']}: {state}{detail}")


if __name__ == "__main__":
    main()
