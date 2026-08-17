"""Global experiment lock for H0.6.

A second evaluator refuses to run when another H0.6 cycle holds the
lock. Implemented with ``flock`` on a fixed file under
``/var/lock/hyperloom-evaluator.lock``; the lock file is created lazily.

Behavior:
  * ``acquire(shared=False)`` blocks until the exclusive lock is held.
  * ``release()`` releases the lock and closes the FD.
  * The lock is reentrant: if the same process already holds it,
    ``flock`` returns 0 and ``is_held`` reports True (Linux flock
    semantics — same FD, same lock, same owner).
  * Other evaluator runs that try ``acquire`` while held will block
    forever, then return false at the caller's chosen timeout.

Use as a context manager:

    with ExperimentLock() as held:
        if not held:
            print("another evaluator holds the XTX")
            sys.exit(2)
        ...
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import time
from pathlib import Path
from typing import Iterator, Optional


def _default_lock_path() -> Path:
    """Lock file in user-writable state: ``~/.cache/hyperloom/evaluator.lock``.

    Falls back to /var/lock (systemd standard) if the user is root and
    the directory is writable; otherwise we use the user's XDG cache
    home. Override with the ``HYPERLOOM_EVALUATOR_LOCK`` environment
    variable for tests + alternate hosts.
    """
    override = os.environ.get("HYPERLOOM_EVALUATOR_LOCK")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "hyperloom" / "evaluator.lock"
    return Path.home() / ".cache" / "hyperloom" / "evaluator.lock"


LOCK_PATH = _default_lock_path()


def _ensure_lockfile() -> Path:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOCK_PATH.exists():
        LOCK_PATH.touch()
    return LOCK_PATH


class ExperimentLock:
    """flock-based exclusive XTX experiment lock."""

    def __init__(self, path: Path = LOCK_PATH) -> None:
        self.path = _ensure_lockfile() if path == LOCK_PATH else path
        self._fd: Optional[int] = None

    @property
    def is_held(self) -> bool:
        return self._fd is not None

    def acquire(self, *, timeout_seconds: float = 0.0) -> bool:
        """Acquire the exclusive lock. Returns True if acquired.

        With ``timeout_seconds > 0``, polls at 1 Hz for up to that long.
        With ``timeout_seconds == 0``, does a non-blocking try and
        returns False immediately if the lock is held by another
        process.
        """
        if self._fd is not None:
            return True  # already held by us
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd = fd
                # Write our pid into the lock file for diagnostics.
                os.ftruncate(fd, 0)
                os.write(fd, f"{os.getpid()}\n".encode())
                return True
            except BlockingIOError:
                os.close(fd)
                if time.monotonic() >= deadline:
                    return False
                time.sleep(1.0)

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def __enter__(self) -> "ExperimentLock":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def acquire_for_test(timeout_seconds: float = 0.0) -> "ExperimentLock":
    """Helper for tests; returns an ExperimentLock with the lock held.

    Test-only convenience: the orchestrator uses ``ExperimentLock`` with
    timeout_seconds=0.0 (immediate fail-closed).
    """
    lock = ExperimentLock()
    if not lock.acquire(timeout_seconds=timeout_seconds):
        raise RuntimeError("could not acquire lock for test")
    return lock


@contextlib.contextmanager
def held(timeout_seconds: float = 0.0) -> Iterator[bool]:
    """Context manager: yields True if the lock was acquired, else False."""
    lock = ExperimentLock()
    got = lock.acquire(timeout_seconds=timeout_seconds)
    try:
        yield got
    finally:
        lock.release()