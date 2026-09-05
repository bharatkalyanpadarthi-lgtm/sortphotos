#!/usr/bin/env python3
"""Verified, source-preserving file operations for the photo pipeline."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_copy(src: Path, dest: Path, *, use_hardlinks: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if use_hardlinks:
        try:
            os.link(str(src), str(dest))
            return
        except OSError:
            pass
    temporary = dest.parent / (dest.name + ".part")
    try:
        shutil.copy2(str(src), str(temporary))
        temporary.replace(dest)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def verify_original_copy(src: Path, dest: Path, expected_sha256: str = "") -> str:
    """Verify a new copy and remove only the unverified destination on failure."""
    try:
        if not dest.exists():
            raise OSError(f"copied destination is missing: {dest}")
        try:
            if os.path.samefile(src, dest):
                actual_hash = sha256_file(dest)
                if expected_sha256 and actual_hash != expected_sha256:
                    raise OSError(f"hardlink hash verification failed: {src} -> {dest}")
                return actual_hash
        except OSError:
            if not dest.exists():
                raise
        if src.stat().st_size != dest.stat().st_size:
            raise OSError(f"copy size verification failed: {src} -> {dest}")
        source_hash = expected_sha256 or sha256_file(src)
        destination_hash = sha256_file(dest)
        if destination_hash != source_hash:
            raise OSError(f"copy hash verification failed: {src} -> {dest}")
        return source_hash
    except Exception:
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        raise
