from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_path(script_file: str) -> Path:
    repo_root = Path(script_file).resolve().parents[2]
    src_path = repo_root / "src"
    for path in (repo_root, src_path):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return repo_root
