"""Bounded, data-only serialization for a single automatic sweep's profiles."""

import base64
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

import benchmark_snapshot
import recognition_policy
import secondary_identity_matcher
import sort_photos

MAX_BYTES = 128 * 1024 * 1024


def _encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'fiu' or value.dtype.itemsize > 8:
            raise ValueError('Unsupported profile array type')
        return {'array': base64.b64encode(value.tobytes()).decode('ascii'),
                'dtype': value.dtype.str, 'shape': list(value.shape)}
    if isinstance(value, np.generic):
        return _encode(value.item())
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Non-finite profile value')
        return value
    if isinstance(value, (list, tuple)):
        return {'list': [_encode(item) for item in value]}
    if hasattr(value, 'items'):
        if any(not isinstance(key, str) for key in value):
            raise ValueError('Profile map keys must be strings')
        return {'map': {key: _encode(item) for key, item in value.items()}}
    raise TypeError(f'Unsupported profile value: {type(value).__name__}')


def _decode(value, depth=0):
    if depth > 20:
        raise ValueError('Profile nesting exceeds limit')
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Non-finite profile value')
        return value
    if not isinstance(value, dict):
        raise ValueError('Invalid profile value')
    if set(value) == {'list'} and isinstance(value['list'], list):
        return [_decode(item, depth + 1) for item in value['list']]
    if set(value) == {'map'} and isinstance(value['map'], dict):
        return {key: _decode(item, depth + 1) for key, item in value['map'].items()}
    if set(value) == {'array', 'dtype', 'shape'}:
        dtype = np.dtype(value['dtype'])
        shape = value['shape']
        if dtype.kind not in 'fiu' or dtype.itemsize > 8 or not isinstance(shape, list) or len(shape) > 4:
            raise ValueError('Invalid profile array')
        if any(type(size) is not int or size < 0 for size in shape):
            raise ValueError('Invalid profile shape')
        count = math.prod(shape)
        if count * dtype.itemsize > MAX_BYTES:
            raise ValueError('Profile array exceeds limit')
        raw = base64.b64decode(value['array'], validate=True)
        if len(raw) != count * dtype.itemsize:
            raise ValueError('Profile array length mismatch')
        return np.frombuffer(raw, dtype=dtype).reshape(shape)
    raise ValueError('Invalid profile encoding')


def encode(primary, secondary, recovery, negatives):
    primary_fields = {field.name: getattr(primary, field.name) for field in dataclasses.fields(primary)}
    secondary_fields = {field.name: getattr(secondary, field.name) for field in dataclasses.fields(secondary)
                        if field.name != 'crop_embeddings'}
    value = {'version': 1, 'primary': _encode(primary_fields), 'secondary': _encode(secondary_fields),
             'recovery': _encode(dict(recovery)), 'sources': _encode(dict(recovery.sources)),
             'negatives': _encode(negatives)}
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(data) > MAX_BYTES:
        raise ValueError('Profile snapshot exceeds limit')
    return data


def write(path: Path, data: bytes):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(data).hexdigest()


def read(path: Path, expected_hash: str):
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, 'rb') as handle:
        data = handle.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError('Profile snapshot hash or size mismatch')
    payload = json.loads(data)
    if set(payload) != {'version', 'primary', 'secondary', 'recovery', 'sources', 'negatives'} or payload['version'] != 1:
        raise ValueError('Unsupported profile snapshot')
    primary = sort_photos.IdentityDB(**_decode(payload['primary']))
    secondary = secondary_identity_matcher.SecondaryIdentityDB(**_decode(payload['secondary']))
    recovery = recognition_policy.RecoveryProfiles()
    recovery.update(_decode(payload['recovery']))
    recovery.sources = _decode(payload['sources'])
    return (benchmark_snapshot.freeze_profile(primary), benchmark_snapshot.freeze_profile(secondary),
            benchmark_snapshot.FrozenRecoveryProfiles(recovery),
            benchmark_snapshot.freeze_mapping(_decode(payload['negatives'])))
