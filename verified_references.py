"""Run-scoped, replacement-aware content lookup for confirmed references.

Only file fingerprints are retained. No automatic labels or cached embeddings
authorize trust: every resolved file must match its confirmation's SHA-256.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

import content_identity
from pipeline_progress import StageProgress, terminal_progress


class ReferenceIndex:
    def __init__(self, candidates: Iterable[Path | str] = (), *,
                 faces_by_source: dict[str, list] | None = None,
                 progress: Callable[[str], None] | None = terminal_progress):
        self.progress = progress
        self.faces_by_source = faces_by_source
        self.hashes_read = 0
        self.hashes_reused = 0
        self._hashes = {}
        self._by_person = defaultdict(list)
        self._sizes = {}
        self._lookups = {}
        seen = set()
        for candidate in faces_by_source if faces_by_source is not None else candidates:
            path = Path(candidate)
            if path in seen:
                continue
            seen.add(path)
            for i, part in enumerate(path.parts[:-1]):
                if path.parts[i + 1] == "photos":
                    self._by_person[part.casefold()].append(path)

    def summary(self) -> str:
        return f"{self.hashes_read} files hashed, {self.hashes_reused} verified hashes reused"

    def _digest(self, path: Path) -> str:
        path = path.expanduser().resolve()
        version = content_identity.file_version(path)
        cached = self._hashes.get(path)
        if cached is not None and cached[0] == version:
            self.hashes_reused += 1
            return cached[1]
        digest = content_identity.content_sha256(path)
        if content_identity.file_version(path) != version:
            raise OSError(f"Reference changed while verifying: {path}")
        self._hashes[path] = (version, digest)
        self.hashes_read += 1
        return digest

    def refresh(self) -> None:
        """Rebuild lazy lookup buckets between stages; retain verified hashes.

        Candidate paths are a run snapshot, but their bytes may have changed.
        Rechecking stats also lets a temporarily unreadable file recover in the
        next stage, without retaining negative results across stages.
        """
        self._sizes.clear()
        self._lookups.clear()

    def _lookup(self, person: str, byte_size: int | None) -> dict[str, list[Path]]:
        if person not in self._sizes:
            sizes = defaultdict(list)
            paths = self._by_person.get(person, [])
            status = StageProgress(f"Indexing moved references for {person}", len(paths), self.progress)
            for path in status.items(paths):
                try:
                    if path.is_file():
                        sizes[path.stat().st_size].append(path)
                except OSError:
                    continue
            self._sizes[person] = sizes
        key = person, byte_size
        if key not in self._lookups:
            sizes = self._sizes[person]
            paths = (self._by_person.get(person, []) if byte_size is None
                     else sizes.get(byte_size, []))
            lookup = defaultdict(list)
            # The outer confirmation counter covers small buckets. Avoid two
            # log lines per file when legacy references have unique sizes.
            status = StageProgress(f"Verifying moved references for {person}", len(paths),
                                   self.progress if len(paths) >= 100 else None)
            for path in status.items(paths, self.summary):
                try:
                    lookup[self._digest(path)].append(path)
                except OSError:
                    continue
            self._lookups[key] = lookup
        return self._lookups[key]

    def resolve(self, item: dict) -> Path | None:
        digest = str(item.get("content_sha256", ""))
        if len(digest) != 64:
            return None
        try:
            size = int(item["byte_size"]) if item.get("byte_size") is not None else None
        except (ValueError, TypeError):
            return None

        def matches(path):
            try:
                if size is not None and path.stat().st_size != size:
                    return False
                return self._digest(path) == digest
            except (OSError, ValueError):
                return False

        source = Path(str(item.get("organized_path", ""))).expanduser()
        if matches(source):
            return source.resolve()
        person = str(item.get("person", "")).casefold()
        for candidate in self._lookup(person, size).get(digest, []):
            # Index hits are only hints until the current file is verified.
            if matches(candidate):
                return candidate.resolve()
        return None
