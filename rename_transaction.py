"""Recoverable two-phase, no-clobber filename changes."""
from pathlib import Path
import json
import os
import uuid

from file_operations import rename_exclusive, sha256_file, sync_directory


def save(path, payload):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def recover(path: Path) -> None:
    state = json.loads(path.read_text())
    if state.get("version") != 1:
        raise ValueError(f"Unsupported rename journal: {path}")
    for item in state["items"]:
        source, temporary, destination = (Path(item[key]) for key in ("source", "temporary", "destination"))
        target = source if state["phase"] == "staging" else destination
        if temporary.exists():
            if sha256_file(temporary) != item["sha256"]:
                raise ValueError(f"Rename recovery content mismatch: {temporary}")
            rename_exclusive(temporary, target)
        elif not target.is_file() or sha256_file(target) != item["sha256"]:
            raise ValueError(f"Rename recovery needs inspection: {target}")
    path.unlink()
    sync_directory(path.parent)


def recover_under(root: Path) -> None:
    for path in root.rglob(".rename_transaction_*.json"):
        recover(path)
    leftovers = next(root.rglob(".rename_tmp_*"), None)
    if leftovers is not None:
        raise ValueError(f"Unjournaled staged original requires recovery; not deleting: {leftovers}")


def apply(actions) -> None:
    if not actions:
        return
    sources = {Path(src).absolute() for src, _ in actions}
    destinations = [Path(dst).absolute() for _, dst in actions]
    if len(sources) != len(actions) or len(set(destinations)) != len(actions):
        raise ValueError("Duplicate source or destination in rename plan")
    for dest in destinations:
        if os.path.lexists(dest) and dest not in sources:
            raise FileExistsError(f"Rename destination occupied: {dest}")
    token = uuid.uuid4().hex
    path = Path(actions[0][0]).parent / f".rename_transaction_{token}.json"
    state = {"version": 1, "phase": "staging", "items": []}
    for index, (src, dest) in enumerate(actions):
        src, dest = Path(src).absolute(), Path(dest).absolute()
        if src.is_symlink():
            raise ValueError(f"Refusing to rename source symlink: {src}")
        state["items"].append({"source": str(src), "destination": str(dest),
                               "temporary": str(src.with_name(f".rename_tmp_{token}_{index}{src.suffix}")),
                               "sha256": sha256_file(src)})
    save(path, state)
    try:
        for item in state["items"]:
            rename_exclusive(Path(item["source"]), Path(item["temporary"]))
    except Exception:
        recover(path)
        raise
    state["phase"] = "committing"
    save(path, state)
    for item in state["items"]:
        Path(item["destination"]).parent.mkdir(parents=True, exist_ok=True)
    recover(path)
