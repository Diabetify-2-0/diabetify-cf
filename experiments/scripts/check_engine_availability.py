from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

try:
    from experiments.scripts._bootstrap import bootstrap_path
except ModuleNotFoundError:
    from _bootstrap import bootstrap_path


bootstrap_path(__file__)

from diabetify_cf.config import Settings
from experiments.engines.factory import SUPPORTED_ENGINES, build_experiment_engine

ENGINE_IMPORTS = {
    "dice_constrained_native": ["dice_ml"],
    "carla": ["carla"],
    "nn_production": ["sklearn"],
    "focus": ["focus"],
}


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def check_availability(settings: Settings) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for engine_name, module_names in ENGINE_IMPORTS.items():
        missing = [name for name in module_names if not _module_available(name)]
        row = {
            "engine": engine_name,
            "implemented": engine_name in SUPPORTED_ENGINES,
            "available": not missing,
            "missing_modules": missing,
        }
        if row["implemented"] and row["available"]:
            adapter = build_experiment_engine(engine_name, settings=settings)
            row["artifact_ready"] = adapter.is_ready
            row["initialization_error"] = adapter.initialization_error
        rows.append(row)

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
        if not row.get("implemented", False):
            detail += " implemented=False"
        if "artifact_ready" in row:
            detail += f" artifact_ready={row['artifact_ready']}"
        print(f"{row['engine']}: {state}{detail}")


if __name__ == "__main__":
    main()
