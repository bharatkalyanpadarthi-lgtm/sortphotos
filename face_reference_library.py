"""Incremental reference evidence for the canonical, benchmark-gated matcher.

Folder labels are hints, not manual confirmations. Only strong single-face
matches may supplement an established profile; originals are never modified.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import analysis_index
import content_identity
import identity_hard_negatives
import identity_profiles
import person_aliases
import pipeline_paths


POLICY_VERSION = 1
MAX_SUPPLEMENTS = 2
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif', '.bmp', '.tif', '.tiff'}


@dataclass
class Snapshot:
    samples: dict = field(default_factory=dict)
    signatures: dict = field(default_factory=dict)
    versions: dict = field(default_factory=dict)
    guards: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)

    def validate(self):
        for path, version in self.versions.items():
            if content_identity.file_version(Path(path)) != version:
                raise RuntimeError(f'Face Reference changed during profile validation: {path}')
        for path, version in self.guards.items():
            if optional_version(Path(path)) != version:
                raise RuntimeError(f'Recognition evidence changed during reference validation: {path}')


def optional_version(path):
    try:
        return content_identity.file_version(path)
    except FileNotFoundError:
        return None


def index_path():
    return pipeline_paths.ANALYSIS_INDEX.parent / 'face_reference_analysis.sqlite3'


def selection_signature(core, files, versions, detector_config):
    import sort_photos as sorter
    digest = hashlib.sha256()
    for source in (__file__, identity_profiles.__file__, identity_hard_negatives.__file__):
        digest.update(content_identity.content_sha256(Path(source)).encode())
    digest.update(json.dumps([POLICY_VERSION, detector_config, [(str(p), n, versions[str(p)]) for p, n in files],
                             sorter.AUTO_PERSON_SINGLE_MATCH_DIST, sorter.AUTO_PERSON_SINGLE_MATCH_MARGIN,
                             sorter.AUTO_PERSON_SINGLE_MIN_QUALITY, sorter.AUTO_PERSON_MATCH_MIN_REFERENCE_FACES],
                            sort_keys=True).encode())
    for name in sorted(core.identities):
        digest.update(json.dumps([name, core.source_counts.get(name), core.strict_thresholds.get(name),
                                 core.prototype_sources.get(name)], sort_keys=True).encode())
        for value in [core.identities[name], *core.prototypes.get(name, [])]:
            digest.update(np.asarray(value, dtype=np.float32).tobytes())
    negative_path = sorter.IDENTITY_HARD_NEGATIVES_FILE
    if negative_path.is_file():
        digest.update(content_identity.content_sha256(negative_path).encode())
    return digest.hexdigest()


def cached_selection(index, signature, guards):
    row = index.connection.execute('SELECT value FROM metadata WHERE key=?', ('prepared_reference_snapshot',)).fetchone()
    if not row:
        return None
    try:
        value = json.loads(row[0])
        if value['signature'] != signature:
            return None
        snapshot = Snapshot(signatures=value['signatures'], rows=value['rows'],
                            versions={p: tuple(v) for p, v in value['versions'].items()},
                            counts=Counter(value['counts']), guards=guards)
        snapshot.samples = {person: [identity_profiles.ReferenceSample(
            item['source'], np.asarray(item['embedding'], dtype=np.float32), item['quality'], item['pose_label'])
            for item in samples] for person, samples in value['samples'].items()}
        snapshot.counts['analyzed'] = 0
        snapshot.counts['reused'] = snapshot.counts['eligible']
        snapshot.validate()
        return snapshot
    except (KeyError, TypeError, ValueError):
        return None


def save_selection(index, signature, snapshot):
    value = {'signature': signature, 'signatures': snapshot.signatures,
             'versions': snapshot.versions, 'rows': snapshot.rows, 'counts': dict(snapshot.counts),
             'samples': {person: [dict(source=s.source, embedding=s.embedding.tolist(),
                                      quality=s.quality, pose_label=s.pose_label) for s in samples]
                         for person, samples in snapshot.samples.items()}}
    index.connection.execute('INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)',
                             ('prepared_reference_snapshot', json.dumps(value)))


def core_profiles(db):
    """Never use previously admitted supplements to authorize more supplements."""
    result = copy.deepcopy(db)
    for name, sources in getattr(db, 'reference_sources', {}).items():
        excluded = set(sources)
        if len(db.prototypes.get(name, [])) != len(db.prototype_sources.get(name, [])):
            raise ValueError(f'Missing reference provenance for {name}; profile update refused')
        pairs = [(v, p) for v, p in zip(db.prototypes.get(name, []), db.prototype_sources.get(name, []))
                 if p not in excluded]
        result.prototypes[name] = [v for v, _p in pairs]
        result.prototype_sources[name] = [p for _v, p in pairs]
    result.reference_sources = {}
    result.reference_signatures = {}
    return result


def reference_files(root, people_root, names):
    by_name = defaultdict(list)
    for name in names:
        by_name[' '.join(name.split()).casefold()].append(name)
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.is_symlink() or directory.name.startswith(('.', '_')):
            continue
        label = directory.name.replace('_', ' ')
        label = person_aliases.canonical_folder(label, people_root)
        matches = by_name.get(' '.join(label.split()).casefold(), [])
        person = matches[0] if len(matches) == 1 else ''
        for parent, directories, files in os.walk(directory, followlinks=False):
            directories[:] = sorted(d for d in directories if not d.startswith(('.', '_'))
                                    and d.casefold() not in {'review', 'duplicates', 'junk'}
                                    and not (Path(parent) / d).is_symlink())
            for filename in sorted(files):
                path = Path(parent) / filename
                if (not filename.startswith(('.', '_')) and path.suffix.lower() in IMAGE_EXTENSIONS
                        and not path.is_symlink() and path.is_file()):
                    yield path.resolve(), person


def detect_missing(paths, database, *, batch_size=25, progress=print):
    """Use the existing isolated detector, without writing the live face cache."""
    import sort_photos as sorter
    env = os.environ.copy()
    env.update(sorter.DETECTION_WORKER_ENV_LIMITS)
    with tempfile.TemporaryDirectory(prefix='face-reference-worker-') as temporary:
        job, output = Path(temporary) / 'job.pkl', Path(temporary) / 'result.pkl'
        for start in range(0, len(paths), batch_size):
            batch = paths[start:start + batch_size]
            output.unlink(missing_ok=True)
            with job.open('wb') as handle:
                pickle.dump({'input_paths': [str(p) for p in batch], 'output_path': str(output),
                             'det_size': sorter.DET_SIZE[0]}, handle)
            progress(f'Face References: analyzing {start + 1}-{start + len(batch)}/{len(paths)} changed images')
            result = subprocess.run([sys.executable, str(Path(sorter.__file__).resolve()),
                                     '--detect-batch', str(job)], env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
            if result.returncode or not output.is_file():
                raise RuntimeError(f'Reference detector failed; completed batches remain cached. {result.stdout[-1000:]}')
            sorter.install_pickle_class_aliases()
            with output.open('rb') as handle:
                payload = pickle.load(handle)
            faces = defaultdict(list)
            for face in payload['faces']:
                faces[os.path.realpath(face.src_str)].append(face)
            with analysis_index.AnalysisIndex(database) as index:
                for path in batch:
                    fingerprint = payload['fingerprints'].get(str(path), {})
                    status = payload['diagnostics'].get(str(path), '')
                    digest = fingerprint.get('sha256')
                    if (not digest or digest != index.content_sha256(path)
                            or status not in {'accepted_face', 'accepted_face_recovery', 'no_face_detected'}
                            and not status.startswith('face_quality_review:')):
                        continue
                    detected = faces[str(path)]
                    if any(face.content_sha256 != digest for face in detected):
                        raise RuntimeError(f'Reference detector returned stale content: {path}')
                    index.replace_detections(path, sorter.config_fingerprint(), status,
                        [sorter.cached_face_to_index_record(face) for face in detected], expected_sha256=digest)


def prepare(root, people_root, db, *, database=None, detector=detect_missing, progress=print):
    import sort_photos as sorter
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Face References unavailable; retaining active profiles: {root}')
    database = database or index_path()
    core = core_profiles(db)
    negatives = identity_hard_negatives.vectors_by_person(sorter.IDENTITY_HARD_NEGATIVES_FILE)
    rejections = identity_hard_negatives.ReferenceRejections(sorter.IDENTITY_HARD_NEGATIVES_FILE)
    compiled = identity_profiles.CompiledProfiles(core.identities, core.prototypes, hard_negatives=negatives)
    snapshot = Snapshot()
    import identity_evaluation
    snapshot.guards = {str(path): optional_version(path) for path in (
        sorter.IDENTITY_DB_FILE, sorter.IDENTITY_HARD_NEGATIVES_FILE,
        sorter.IDENTITY_CONFIRMATIONS_FILE, person_aliases.RULES_PATH,
        identity_evaluation.DEFAULT_PROTECTED_SET, identity_evaluation.DEFAULT_PROTECTED_BASELINE)}
    files = list(reference_files(root, people_root, core.identities))
    snapshot.versions = {str(path): content_identity.file_version(path) for path, _person in files}
    signature = selection_signature(core, files, snapshot.versions, sorter.config_fingerprint())
    pending, hashes, cached, by_hash = [], {}, {}, {}
    with analysis_index.AnalysisIndex(database) as index:
        saved = cached_selection(index, signature, snapshot.guards)
        if saved is not None:
            return saved
        for path, person in files:
            if not person:
                continue
            hashes[path] = index.content_sha256(path)
            value = index.cached_detections(path, sorter.config_fingerprint())
            if value is None:
                value = index.reusable_detections(path, sorter.config_fingerprint())
            if value is None:
                pending.append(path)
            else:
                cached[path] = value
                by_hash[hashes[path]] = value
    representatives = {}
    for path in pending:
        if hashes[path] not in by_hash:
            representatives.setdefault(hashes[path], path)
    changed = list(representatives.values())
    snapshot.counts.update(files=len(files), eligible=len(hashes), analyzed=len(changed), reused=len(hashes) - len(changed))
    if pending:
        if changed:
            detector(changed, database, progress=progress)
        with analysis_index.AnalysisIndex(database) as index:
            for path in changed:
                by_hash[hashes[path]] = index.cached_detections(path, sorter.config_fingerprint())
        for path in pending:
            cached[path] = by_hash.get(hashes[path])
    # A byte-identical file filed under different reference labels is not evidence.
    owners = defaultdict(set)
    for path, person in files:
        if person and hashes.get(path):
            owners[hashes[path]].add(person)
    accepted, seen = defaultdict(list), set()
    for path, person in files:
        row = {'source': str(path), 'person': person, 'status': 'unmapped_name'}
        snapshot.rows.append(row)
        value = cached.get(path)
        if not person:
            continue
        row['sha256'] = hashes[path]
        if len(core.prototypes.get(person, [])) != len(core.prototype_sources.get(person, [])):
            row['status'] = 'missing_profile_provenance'
            continue
        if value is None:
            row['status'] = 'analysis_unavailable'
            continue
        if len(owners[hashes[path]]) > 1:
            row['status'] = 'conflicting_labels'
            continue
        if len(value.detections) != 1:
            row['status'] = 'multiple_faces' if value.detections else 'no_usable_face'
            continue
        face = sorter.index_record_to_cached_face(path, value.detections[0])
        if (not np.isfinite(face.embedding).all() or not np.isfinite(face.quality)
                or face.quality < sorter.AUTO_PERSON_SINGLE_MIN_QUALITY):
            row['status'] = 'low_quality'
            continue
        candidates = compiled.rank(face.embedding)
        best = candidates[0] if candidates else None
        margin = identity_profiles.candidate_margin(candidates)
        row.update(candidate=best.name if best else '', distance=best.distance if best else None, margin=margin)
        if (best is None or best.name != person or rejections.rejects(person, face.embedding)
                or core.source_counts.get(person, 0) < sorter.AUTO_PERSON_MATCH_MIN_REFERENCE_FACES
                or best.distance > min(sorter.AUTO_PERSON_SINGLE_MATCH_DIST, core.strict_thresholds.get(person, 0.0))
                or margin < sorter.AUTO_PERSON_SINGLE_MATCH_MARGIN):
            row['status'] = 'held_for_identity_verification'
            continue
        if (person, hashes[path]) in seen:
            row['status'] = 'duplicate_content'
            continue
        seen.add((person, hashes[path]))
        values = core.prototypes.get(person, [])
        if values and min(1.0 - identity_profiles.normalize_matrix(values)
                          @ identity_profiles.normalize_vector(face.embedding)) < 1e-5:
            row['status'] = 'already_represented'
            continue
        row['status'] = 'verified_candidate'
        accepted[person].append(identity_profiles.ReferenceSample(
            str(path), face.embedding, face.quality, face.pose_label))
    snapshot.validate()
    for person, samples in accepted.items():
        selected = identity_profiles.select_diverse_samples(samples, limit=MAX_SUPPLEMENTS)
        snapshot.samples[person] = selected
        evidence = [(s.source, hashes[Path(s.source)], content_identity.face_identity(
            sorter.index_record_to_cached_face(Path(s.source), cached[Path(s.source)].detections[0]))) for s in selected]
        snapshot.signatures[person] = hashlib.sha256(json.dumps(
            [POLICY_VERSION, sorter.config_fingerprint(), evidence], sort_keys=True).encode()).hexdigest()
    snapshot.counts.update(row['status'] for row in snapshot.rows)
    snapshot.counts['selected'] = sum(map(len, snapshot.samples.values()))
    with analysis_index.AnalysisIndex(database) as index:
        save_selection(index, signature, snapshot)
    return snapshot


def augment(db, snapshot):
    result = core_profiles(db)
    for person, samples in snapshot.samples.items():
        if person not in result.identities:
            continue
        result.prototypes[person].extend(s.embedding for s in samples)
        result.prototype_sources[person].extend(s.source for s in samples)
        result.reference_sources[person] = [s.source for s in samples]
        result.reference_signatures[person] = snapshot.signatures[person]
    return result


def refresh(db, people_root, *, root=None, database=None, retry=False, progress=print):
    """One shared startup path for daily sorting, explicit refresh, and review."""
    import sort_photos as sorter
    if not db or not db.identities:
        return db
    root = Path(root or pipeline_paths.FACE_REFERENCES)
    database = database or index_path()
    try:
        snapshot = prepare(root, people_root, db, database=database, progress=progress)
        progress(f'Face References: {snapshot.counts["reused"]} cached, {snapshot.counts["analyzed"]} analyzed; '
                 f'{snapshot.counts["selected"]} verified supplements')
        report = {'counts': dict(snapshot.counts), 'items': snapshot.rows}
        with analysis_index.AnalysisIndex(database) as index:
            index.connection.execute('INSERT INTO metadata(key,value) VALUES(?,?) '
                                     'ON CONFLICT(key) DO UPDATE SET value=excluded.value WHERE value<>excluded.value',
                                     ('last_reference_scan', json.dumps(report)))
        if snapshot.signatures == getattr(db, 'reference_signatures', {}):
            return db
        # A blocked candidate is retried only when evidence, safety inputs, or code changes.
        import identity_evaluation
        import evaluation_enrollment
        evidence_files = [Path(__file__), Path(sorter.__file__), Path(identity_profiles.__file__),
                          Path(identity_hard_negatives.__file__), Path(identity_evaluation.__file__),
                          sorter.IDENTITY_DB_FILE,
                          sorter.IDENTITY_HARD_NEGATIVES_FILE, sorter.IDENTITY_CONFIRMATIONS_FILE,
                          identity_evaluation.DEFAULT_PROTECTED_SET,
                          identity_evaluation.DEFAULT_PROTECTED_BASELINE, evaluation_enrollment.DEFAULT_PATH]
        token = hashlib.sha256(json.dumps([snapshot.signatures, [
            content_identity.content_sha256(p) if p.is_file() else None for p in evidence_files
        ]], sort_keys=True).encode()).hexdigest()
        with analysis_index.AnalysisIndex(database) as index:
            prior = index.connection.execute('SELECT value FROM metadata WHERE key=?', ('blocked_profile_token',)).fetchone()
        if not retry and prior and prior[0] == token:
            progress('Face References: candidate held by safety checks; active profiles unchanged.')
            return db
        result = sorter.build_identity_db_from_person_folders(people_root, reference_snapshot=snapshot)
        if getattr(result, 'reference_signatures', {}) != snapshot.signatures:
            with analysis_index.AnalysisIndex(database) as index:
                index.connection.execute('INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)', ('blocked_profile_token', token))
        return result
    except (OSError, ValueError, RuntimeError, sqlite3.Error, subprocess.TimeoutExpired) as error:
        progress(f'Face References not activated: {error}. Existing profiles remain available.')
        return sorter.load_identity_db() or db


def main():
    import argparse
    import sort_photos as sorter
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retry', action='store_true', help='Retry a previously blocked candidate through the normal safety gate')
    args = parser.parse_args()
    db = sorter.load_identity_db()
    if db is None:
        parser.error('Build the canonical person profiles with face rebuild-id first')
    refresh(db, pipeline_paths.PEOPLE_ROOT, retry=args.retry)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
