"""Content identities with replacement-aware, bounded signature caching."""

import hashlib
from functools import lru_cache
from pathlib import Path


def file_version(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino


@lru_cache(maxsize=8192)
def _hash_version(path: str, version: tuple[int, ...]) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    if file_version(source) != version:
        raise OSError(f"File changed while hashing: {source}")
    return digest.hexdigest()


def content_sha256(path: Path | str) -> str:
    source = Path(path).expanduser().resolve()
    return _hash_version(str(source), file_version(source))


def face_identity(face) -> str:
    crop = bytes(getattr(face, "crop_jpeg", b"") or b"")
    if crop:
        return "crop:" + hashlib.sha256(crop).hexdigest()
    return ""
