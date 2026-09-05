"""Read-only verification of the dependency/model snapshot used in validation."""

import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import content_identity
import secondary_identity_matcher
import sort_photos


def main():
    manifest = json.loads(Path(__file__).with_name("runtime_manifest.json").read_text())
    failures = []
    if manifest["python"] != ".".join(map(str, sys.version_info[:2])):
        failures.append("Python major/minor differs from the tested runtime")
    for name, expected in manifest["packages"].items():
        try:
            actual = version(name)
        except PackageNotFoundError:
            actual = "missing"
        if actual != expected:
            failures.append(f"{name}: expected {expected}, found {actual}")
    roots = {"primary": Path.home() / ".insightface" / "models",
             "secondary": secondary_identity_matcher.DEFAULT_MODEL_ROOT / "models"}
    for item in manifest["models"]:
        path = roots[item["role"]] / item["model"] / item["file"]
        try:
            valid = path.stat().st_size == item["byte_size"] and content_identity.content_sha256(path) == item["sha256"]
        except OSError:
            valid = False
        if not valid:
            failures.append(f"Model missing or different: {item['role']}/{item['model']}/{item['file']}")
    print(f"Runtime check: {len(manifest['packages'])} dependencies, {len(manifest['models'])} model files, {len(failures)} differences")
    for failure in failures:
        print(f"  {failure}")
    print(f"Active primary: {sort_photos.MODEL_NAME}; secondary: {secondary_identity_matcher.MODEL_NAME}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
