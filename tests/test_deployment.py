from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_runs_as_non_root_user() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "USER appuser" in dockerfile
    assert "useradd" in dockerfile


def test_quality_workflow_runs_required_gates() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")

    assert "ruff check" in workflow
    assert "black --check" in workflow
    assert "mypy src" in workflow
    assert "pytest -q tests" in workflow
