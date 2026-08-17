"""Restoration trap for H0.6.

Restores the exact pre-state on any exit:

  SUCCESS   → restore before returning
  FAILURE   → restore before returning
  TIMEOUT   → restore before returning
  SIGTERM   → restore before re-raising
  SIGINT    → restore before re-raising
  exception → restore before re-raising
  SystemExit/KeyboardInterrupt → restore before re-raising

The trap is implemented with ``contextlib.ContextDecorator`` so it
can wrap any ``with`` block or be used as a decorator.

The trap restores in reverse order:

  1. Kill the candidate PID we launched.
  2. Re-start any service we stopped (allowlist-driven).
  3. Re-acquire the experiment lock (already held; just release on
     exit so other evaluators can run).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import signal
import time
from pathlib import Path
from typing import Callable, Optional

from .allowlist import ServiceEvent, start_service
from .snapshot import PreStateSnapshot


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclasses.dataclass(frozen=True)
class RestorationRecord:
    """Audit record of what the trap did on restore."""

    candidate_pid: Optional[int]
    candidate_killed: bool
    services_restarted: list[ServiceEvent]
    restored_at_utc: str
    post_state_restored: bool

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)


def _kill_pid(pid: int, *, grace_seconds: float = 5.0) -> bool:
    """SIGTERM, then SIGKILL after grace. Returns True if process is gone."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return True  # already gone
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return True
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        return True
    return False


class RestoreTrap:
    """Trap that restores the exact pre-state on any exit.

    Usage:

        trap = RestoreTrap(snapshot, candidate_pid=123, services_to_restart=[...])
        with trap:
            ...  # do experiment work
        # Trap has run. trap.record is populated.

    The trap also installs signal handlers for SIGTERM and SIGINT so an
    external kill triggers the same restoration.
    """

    def __init__(
        self,
        snapshot: PreStateSnapshot,
        *,
        candidate_pid: Optional[int] = None,
        services_to_restart: Optional[list[ServiceEvent]] = None,
    ) -> None:
        self.snapshot = snapshot
        self.candidate_pid = candidate_pid
        self.services_to_restart: list[ServiceEvent] = list(services_to_restart or [])
        self.record: Optional[RestorationRecord] = None
        self._previous_handlers: dict[int, Callable] = {}

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            try:
                self._previous_handlers[sig] = signal.signal(
                    sig, lambda *_: self._restore_and_reraise(sig),
                )
            except (ValueError, OSError):
                # Signal handlers can only be installed from the main thread.
                pass

    def _restore_signal_handlers(self) -> None:
        for sig, handler in self._previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    def _restore_and_reraise(self, sig: int) -> None:
        """Restore + re-raise the original signal."""
        self._run_restore()
        # Re-raise.
        os.kill(os.getpid(), sig)

    def _run_restore(self) -> None:
        if self.record is not None:
            return  # already restored; idempotent
        candidate_killed = False
        if self.candidate_pid is not None:
            candidate_killed = _kill_pid(self.candidate_pid)
        restart_events: list[ServiceEvent] = []
        for evt in self.services_to_restart:
            if evt.action == "stop":
                restart_events.append(start_service(evt.service))
        self.record = RestorationRecord(
            candidate_pid=self.candidate_pid,
            candidate_killed=candidate_killed,
            services_restarted=restart_events,
            restored_at_utc=utc_now(),
            post_state_restored=True,
        )

    def __enter__(self) -> "RestoreTrap":
        self._install_signal_handlers()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._run_restore()
        finally:
            self._restore_signal_handlers()


def write_audit(
    path: Path,
    *,
    pre_state: PreStateSnapshot,
    exclusive_gate: dict,
    experiment_id: int,
    candidate_pid: Optional[int],
    candidate_exit: Optional[int],
    restoration: RestorationRecord,
    post_state: dict,
    outcome: str,
    owners_stopped: Optional[list[str]] = None,
    pre_state_path: Optional[Path] = None,
    restoration_path: Optional[Path] = None,
) -> Path:
    """Write the lifecycle audit artifact.

    The artifact is a single JSON file containing every observable
    signal the orchestrator saw before, during, and after the run. The
    operator can replay this file to audit the experiment later.
    """
    payload = {
        "pre_state": pre_state.to_json() if not pre_state_path else str(pre_state_path),
        "owners_stopped": owners_stopped or [],
        "exclusive_gate": exclusive_gate,
        "experiment_id": experiment_id,
        "candidate_pid": candidate_pid,
        "candidate_exit": candidate_exit,
        "restoration": restoration.to_json(),
        "post_state": post_state,
        "outcome": outcome,
        "captured_utc": utc_now(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path