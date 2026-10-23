"""Private immutable face inputs; never persist or train from these copies."""

import copy
from types import MappingProxyType

import numpy as np


def _freeze(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Benchmark arrays cannot contain Python objects")
        # Immutable bytes own the buffer, so setflags(write=True) cannot undo it.
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    return value


class FrozenFace:
    __slots__ = ("_values",)

    def __init__(self, face):
        object.__setattr__(self, "_values", MappingProxyType(
            {key: _freeze(value) for key, value in vars(face).items()}))

    @property
    def __dict__(self):
        return self._values

    def __getattr__(self, key):
        try:
            return self._values[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __setattr__(self, key, value):
        raise AttributeError("Benchmark faces are immutable")


def freeze_cache(cache, tested_faces, primary_faces):
    clones = {id(face): FrozenFace(face) for face in tested_faces}
    result = copy.copy(cache)
    result.faces = tuple(clones[id(face)] for face in tested_faces)
    result.file_signatures = MappingProxyType(dict(getattr(cache, "file_signatures", {})))
    return result, tuple(clones[id(face)] for face in primary_faces)
