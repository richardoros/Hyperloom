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
    candidate_pgid: Optional[int]
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


def _kill_pgid(pgid: int, *, grace_seconds: float = 5.0) -> bool:
    """SIGTERM the entire process group, then SIGKILL if survivors remain.

    Use this when the candidate launched with ``start_new_session=True``
    so its descendants (driver threads, child shells) are part of the
    same group.
    """
    if pgid <= 0:
        return False
    # SIGTERM the whole group; ignore "no such process" so we proceed.
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return True
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except (OSError, ProcessLookupError):
            return True
        time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        return True
    return False


class RestoreTrap:
    """Trap that restores the exact pre-state on any exit.

    Usage (H0.6.3 — trap is active from drain start through audit write):

        with RestoreTrap(snapshot) as trap:
            stop_allowlisted_services(trap=trap)  # drain FIRST
            trap.record_stop(name)                   # after each stop
            attributes = attribute_to_lab_gpu(...)
            if foreign:
                raise SystemExit  # trap still restarts
            measurement(ctx)                        # covered by trap too

        # trap.record is populated regardless of how we exited.

    The candidate PID is set via :meth:`set_candidate_pid` so the
    callable may register the PID AFTER the trap is active. For
    process-group kills, register the PGID via :meth:`set_candidate_pgid`.
    """

    def __init__(
        self,
        snapshot: PreStateSnapshot,
        *,
        candidate_pid: Optional[int] = None,
        candidate_pgid: Optional[int] = None,
        services_to_restart: Optional[list[ServiceEvent]] = None,
    ) -> None:
        self.snapshot = snapshot
        self._candidate_pid: Optional[int] = candidate_pid
        self._candidate_pgid: Optional[int] = candidate_pgid
        # Tracks services we've stopped so they MUST be restarted on
        # every exit (BLOCKED included). The orchestrator pushes
        # ServiceEvent entries here AS it stops them, not all upfront.
        self._stopped: dict[str, ServiceEvent] = {}
        if services_to_restart:
            for evt in services_to_restart:
                self._stopped[evt.service] = evt
        self.record: Optional[RestorationRecord] = None
        self._previous_handlers: dict[int, Callable] = {}

    @property
    def services_to_restart(self) -> list[ServiceEvent]:
        """Read-only snapshot of all services the trap must restart.

        Order is the order in which they were stopped.
        """
        return list(self._stopped.values())

    def record_stop(self, service: str) -> None:
        """Register ``service`` as stopped.

        Called by the orchestrator immediately after a successful
        ``stop_service(...)`` so the trap owns the restart contract.
        """
        if service in self._stopped:
            return  # idempotent
        self._stopped[service] = ServiceEvent(
            service=service,
            action="stop",
            started_utc=utc_now(),
            finished_utc=utc_now(),
            returncode=0,
        )

    def record_stop_event(self, event: ServiceEvent) -> None:
        """Register a precomputed ServiceEvent."""
        self._stopped[event.service] = event

    def set_candidate_pid(self, pid: int) -> None:
        """Register the candidate's PID AFTER the trap is active."""
        self._candidate_pid = pid

    def set_candidate_pgid(self, pgid: int) -> None:
        """Register the candidate's process-group ID for PG-level kill."""
        self._candidate_pgid = pgid

    @property
    def candidate_pid(self) -> Optional[int]:
        return self._candidate_pid

    @property
    def candidate_pgid(self) -> Optional[int]:
        return self._candidate_pgid

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            try:
                # H0.6.1 fix: bind ``sig`` as a default argument so the
                # lambda closes over its own value, not the loop's
                # final iteration. Without this, all three handlers
                # re-raise SIGHUP regardless of which signal arrived.
                handler = (
                    lambda *_, _sig=sig: self._restore_and_reraise(_sig)
                )
                self._previous_handlers[sig] = signal.signal(sig, handler)
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
        """Restore, restore the previous handler, THEN re-raise.

        Restoring the previous signal handler BEFORE ``os.kill`` is
        what breaks the recursion: without it, ``os.kill(self, sig)``
        re-fires this handler, which re-raises, which re-installs this
        handler's recursion indefinitely (C1 fix).

        Without the previous-handler restoration the trap becomes
        unsafe under any signal — the process cannot exit cleanly
        because every re-raise re-enters us.
        """
        self._run_restore()
        # Restore the previous handler for ``sig`` before re-raising so
        # the kernel's signal disposition matches what Python saw on
        # __enter__.
        previous = self._previous_handlers.get(sig, signal.SIG_DFL)
        try:
            signal.signal(sig, previous)
        except (ValueError, OSError):
            pass
        os.kill(os.getpid(), sig)

    def _kill_candidate(self) -> bool:
        """Best-effort kill of the candidate.

        PGID takes precedence over PID when both are set; the PGID
        is the entire process group the candidate spawned (e.g.
        llama-server + its CUDA/HIP driver threads).
        """
        if self._candidate_pgid is not None and self._candidate_pgid > 0:
            return _kill_pgid(self._candidate_pgid)
        if self._candidate_pid is not None and self._candidate_pid > 0:
            return _kill_pid(self._candidate_pid)
        return False

    def _run_restore(self) -> None:
        if self.record is not None:
            return  # already restored; idempotent
        candidate_killed = self._kill_candidate()
        restart_events: list[ServiceEvent] = []
        # Restart all services recorded via record_stop() — including
        # ones stopped after the trap was entered. The BLOCKED-during-
        # drain path is handled correctly: services stopped before
        # the BLOCK condition was discovered are still restarted.
        for service_name in list(self._stopped):
            restart_events.append(start_service(service_name))
        self.record = RestorationRecord(
            candidate_pid=self._candidate_pid,
            candidate_pgid=self._candidate_pgid,
            candidate_killed=candidate_killed,
            services_restarted=restart_events,
            restored_at_utc=utc_now(),
            # post_state_restored is computed by the orchestrator
            # after a fresh + authoritative post-state snapshot is
            # captured and compared with pre_state. The trap does
            # NOT hardcode this.
            post_state_restored=False,
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