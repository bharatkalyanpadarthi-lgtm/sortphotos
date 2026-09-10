"""Resolve completed folder merges without changing recognition evidence."""

from functools import lru_cache
import json
from pathlib import Path


RULES_PATH = Path(__file__).with_name("person_folder_rules.json")


def _key(value: str) -> str:
    return " ".join(value.split()).casefold()


def _folder_name(value: str) -> bool:
    return bool(value and not value.startswith((".", "_"))
                and not any(c in value for c in "/\\:\0")
                and not any(ord(c) < 32 for c in value))


@lru_cache(maxsize=4)
def _merge_aliases(path: Path, modified_ns: int, size: int) -> dict[str, str]:
    rules = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for target, aliases in rules.get("merge", {}).items():
        if not _folder_name(target):
            raise ValueError("Unsafe canonical person folder in merge rules")
        for alias in aliases:
            if not _folder_name(alias):
                raise ValueError("Unsafe alias in person folder merge rules")
            result[_key(alias)] = target
    return result


def canonical_folder(name: str, people_root: Path, *, rules_path: Path = RULES_PATH) -> str:
    stat = rules_path.stat()
    target = _merge_aliases(rules_path, stat.st_mtime_ns, stat.st_size).get(_key(name))
    if target is None or _key(target) == _key(name):
        return name
    # A configured future merge is not permission to merge two live folders.
    if not (people_root / target).is_dir() or (people_root / name).exists():
        return name
    return target
