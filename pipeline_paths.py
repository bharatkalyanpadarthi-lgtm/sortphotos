"""Per-Mac paths for the photo sorting pipeline.

All user data locations are resolved here.  Individual pipeline tools should
import these constants instead of embedding paths under ``~/Pictures``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


CONFIG_PATH = Path(
    os.environ.get(
        "FACE_PIPELINE_CONFIG",
        str(Path.home() / ".config" / "face-sort" / "paths.json"),
    )
).expanduser()


def configured_path(key: str, default: Path, env_name: str) -> Path:
    env_value = os.environ.get(env_name)
    if env_value:
        return Path(env_value).expanduser()
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        return default
    return Path(value).expanduser()


def configured_bool(key: str, default: bool, env_name: str) -> bool:
    """Resolve a per-Mac boolean preference from the environment or config."""
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return env_value.strip().casefold() in {"1", "true", "yes", "on"}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default
    value = data.get(key)
    return value if isinstance(value, bool) else default


EXTERNAL_DATA_ROOT = Path("/Volumes/SSD 2TB/Photo Sort Data")
LEGACY_SORTED_ROOT = Path.home() / "Pictures" / "sorted_all_pictures"
LEGACY_FACE_REFERENCES = Path.home() / "Pictures" / "Face References"

DATA_ROOT = configured_path(
    "data_root",
    EXTERNAL_DATA_ROOT if EXTERNAL_DATA_ROOT.exists() else LEGACY_SORTED_ROOT.parent,
    "FACE_DATA_ROOT",
)

SORTED_ROOT = configured_path(
    "sorted_root",
    EXTERNAL_DATA_ROOT / "sorted_all_pictures"
    if (EXTERNAL_DATA_ROOT / "sorted_all_pictures").exists()
    else LEGACY_SORTED_ROOT,
    "FACE_SORTED_ROOT",
)

PEOPLE_ROOT = SORTED_ROOT / "photos_by_person"
SOURCE_REVIEW = SORTED_ROOT / "_source_review"

FACE_REFERENCES = configured_path(
    "face_references",
    EXTERNAL_DATA_ROOT / "Face References"
    if (EXTERNAL_DATA_ROOT / "Face References").exists()
    else LEGACY_FACE_REFERENCES,
    "FACE_REFERENCES_DIR",
)

ANALYSIS_INDEX = configured_path(
    "analysis_index",
    DATA_ROOT / "analysis_cache" / "analysis_index.sqlite3",
    "FACE_ANALYSIS_INDEX",
)


TO_PROCESS = configured_path(
    "to_process",
    Path.home() / "Pictures" / "To Process",
    "FACE_TO_PROCESS_DIR",
)
