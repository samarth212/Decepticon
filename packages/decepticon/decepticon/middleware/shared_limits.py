"""Job-wide tool slots shared by sync/async agents and worker processes.

flock releases a slot on close or process death. Async waiters poll without
leaving a blocked executor thread behind when cancelled. These slots limit
in-flight tool calls, not connections created inside arbitrary commands.
"""
import asyncio
from contextlib import asynccontextmanager, contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import tempfile
import time


def _directory(workspace):
    if not workspace:
        raise RuntimeError('A workspace is required for a shared tool limit')
    key = hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()
    root = Path(os.environ.get('MEGATRON_LIMITS_DIR', str(Path(tempfile.gettempdir()) / 'megatron-tool-limits')))
    directory = root / key
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return directory


def _try_acquire(directory, limit):
    for slot in range(limit):
        fd = os.open(directory / f'{slot}.lock', os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            os.close(fd)
        except BaseException:
            os.close(fd)
            raise
    return None


@contextmanager
def sync_gate(workspace, limit):
    directory = _directory(workspace)
    fd = None
    try:
        while fd is None:
            fd = _try_acquire(directory, limit)
            if fd is None: time.sleep(.01)
        yield
    finally:
        if fd is not None: os.close(fd)


@asynccontextmanager
async def async_gate(workspace, limit):
    directory = _directory(workspace)
    fd = None
    try:
        while fd is None:
            fd = _try_acquire(directory, limit)
            if fd is None: await asyncio.sleep(.01)
        yield
    finally:
        if fd is not None: os.close(fd)
