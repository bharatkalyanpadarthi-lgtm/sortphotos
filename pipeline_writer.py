"""One process-tree writer lease for sorting and review commands."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import fcntl
import json
import os
import uuid

LOCK_PATH = Path.home() / ".face_sort_cache" / "pipeline_writer.lock"
OWNER_ENV = "FACE_PIPELINE_WRITER_TOKEN"
FD_ENV = "FACE_PIPELINE_WRITER_FD"


def child_process_options():
    try:
        fd = int(os.environ[FD_ENV])
        os.fstat(fd)
    except (KeyError, ValueError, OSError):
        return {}
    return {"pass_fds": (fd,)}


@contextmanager
def writer_lease(path=None):
    path = Path(path or LOCK_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    inherited_valid = False
    try:
        inherited = int(os.environ[FD_ENV])
        held = os.fstat(inherited)
        current = path.stat()
        owner = json.loads(os.pread(inherited, 4096, 0))
        inherited_valid = ((held.st_dev, held.st_ino) == (current.st_dev, current.st_ino)
                           and owner.get("token") == os.environ.get(OWNER_ENV))
    except (KeyError, ValueError, OSError):
        pass
    if inherited_valid:
        yield
        return
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another photo operation is active. Finish or close it before starting this command.") from None
        token = uuid.uuid4().hex
        previous = os.environ.get(OWNER_ENV)
        previous_fd = os.environ.get(FD_ENV)
        handle.seek(0)
        handle.truncate()
        json.dump({"pid": os.getpid(), "token": token}, handle)
        handle.flush()
        os.fsync(handle.fileno())
        os.environ[OWNER_ENV] = token
        os.environ[FD_ENV] = str(handle.fileno())
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(OWNER_ENV, None)
            else:
                os.environ[OWNER_ENV] = previous
            if previous_fd is None:
                os.environ.pop(FD_ENV, None)
            else:
                os.environ[FD_ENV] = previous_fd
            # Closing, rather than LOCK_UN, retains ownership in an inherited
            # worker descriptor if the parent exits before that worker.


def serialized(function):
    @wraps(function)
    def run(*args, **kwargs):
        with writer_lease():
            return function(*args, **kwargs)
    return run
