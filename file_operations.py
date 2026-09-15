#!/usr/bin/env python3
"""Verified, source-preserving file operations for the photo pipeline."""

from __future__ import annotations

import hashlib
import os
import shutil
import ctypes
import errno
import sys
import uuid
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def rename_exclusive(src: Path, dest: Path) -> None:
    """Publish without replacing an occupied path, including dangling links."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        result = libc.renamex_np(os.fsencode(src), os.fsencode(dest), 4)
    elif hasattr(libc, "renameat2"):
        result = libc.renameat2(-100, os.fsencode(src), -100, os.fsencode(dest), 1)
    else:
        raise OSError(errno.ENOTSUP, "exclusive rename is unavailable")
    if result:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(dest))
    sync_directory(dest.parent)
    if src.parent != dest.parent:
        sync_directory(src.parent)


def atomic_copy(src: Path, dest: Path, *, use_hardlinks: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(dest):
        raise FileExistsError(errno.EEXIST, "destination is occupied", str(dest))
    if use_hardlinks:
        try:
            os.link(str(src), str(dest))
            with dest.open("rb") as handle:
                os.fsync(handle.fileno())
            sync_directory(dest.parent)
            return
        except OSError as error:
            if os.path.lexists(dest) or error.errno not in {
                errno.EXDEV, errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP,
            }:
                raise
    temporary = dest.parent / ("." + dest.name + "." + uuid.uuid4().hex + ".part")
    try:
        with temporary.open("xb") as handle, src.open("rb") as source:
            shutil.copyfileobj(source, handle, 1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copystat(src, temporary)
        verify_original_copy(src, temporary)
        rename_exclusive(temporary, dest)
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
            if src.absolute() != dest.absolute():
                dest.unlink(missing_ok=True)
        except OSError:
            pass
        raise
