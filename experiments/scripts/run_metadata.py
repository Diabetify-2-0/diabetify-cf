from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = [
    "dice-ml",
    "numpy",
    "oceanpy",
    "pandas",
    "scikit-learn",
    "xgboost",
]


def _run_git(repo_root: Path, args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_run_metadata(
    *,
    repo_root: Path,
    run_type: str,
    engine_name: str,
    config_path: Path | None = None,
) -> dict[str, Any]:
    status_short = _run_git(repo_root, ["status", "--short"])
    return {
        "run_type": run_type,
        "engine_name": engine_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path.as_posix() if config_path is not None else None,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "git": {
            "commit": _run_git(repo_root, ["rev-parse", "HEAD"]),
            "branch": _run_git(repo_root, ["branch", "--show-current"]),
            "is_dirty": bool(status_short),
            "status_short": status_short,
        },
        "packages": _package_versions(),
    }
