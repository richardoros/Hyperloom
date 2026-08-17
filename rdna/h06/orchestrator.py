"""H0.6 exclusive-XTX lifecycle orchestrator (H0.6.1 corrections).

Pipeline (corrected order; drain before unknown-owner gate):

    snapshot pre_state
        ↓
    acquire exclusive flock
        ↓
    resolve MainPID(s) of --allow-service services
        ↓
    add those PIDs to --allowed-pids (implicit allowlist during drain)
        ↓
    stop allowlisted services (systemctl stop, never pkill)
        ↓
    wait briefly for VRAM / PIDs to clear
        ↓
    attribute foreign PIDs to lab GPU (BDF-filtered, UNKNOWN for
        unresolvable devices)
        ↓
    BLOCK if any unknown owner remains
        ↓
    run H0.5 measurement in a daemon thread with a wall deadline
        ↓
    on exit (SUCCESS / FAILURE / TIMEOUT / SIGTERM / SIGINT / exception):
        - kill candidate PID (trap captures via MeasurementContext)
        - restart stopped services
        - release lock
        ↓
    fresh post-state probes (services, listeners, prod health,
        GPU owners, GPU telemetry) — never copied from pre-state
        ↓
    machine returns to original working state (proved, not asserted)
        ↓
    emit lifecycle audit artifact

Distinct outcomes:

    PASS        candidate exited 0, post-state matches pre-state
    BLOCKED     environmental prerequisite not satisfied (lab GPU
                drained, OR unknown XTX owner, OR lock held by
                another evaluator)
    TIMEOUT     mechanical deadline hit; trap still restores
    FAIL        candidate exited non-zero, OR restore-trap caught an
                exception, OR post-state proves production is no
                longer healthy

Process ownership contract:

    * On SIGTERM / SIGINT / SIGHUP the trap fires IMMEDIATELY (signal
      handlers installed in __enter__); the measurement callable sees
      this as a Python signal but the candidate PID is killed before
      the trap returns.
    * The measurement callable may register the candidate's PID via
      ``ctx.register_candidate(pid)`` immediately after launching it.
      The trap then has the PID it must kill when the lifecycle ends.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Union

from . import allowlist
from .allowlist import DEFAULT_ALLOWLIST, ServiceEvent
# These are module-qualified: functions use allowlist.is_active etc. so
# tests can patch allowlist.is_active and the call still resolves.
from .lock import LOCK_PATH, ExperimentLock
from .ownership import ProcessGpuOwner, attribute_to_lab_gpu
from .restore import RestorationRecord, RestoreTrap, write_audit
from .snapshot import (
    PreStateSnapshot,
    gpu_telemetry,
    listener_state,
    production_health_ok,
    service_state,
    take_snapshot,
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Outcome(str, enum.Enum):
    PASS = "PASS"
    BLOCKED = "BLOCKED"
    TIMEOUT = "TIMEOUT"
    FAIL = "FAIL"


@dataclasses.dataclass(frozen=True)
class LifecycleAudit:
    outcome: Outcome
    reason: str
    pre_state: PreStateSnapshot
    owners_stopped: list[ServiceEvent]
    foreign_owners_at_acquire: list[ProcessGpuOwner]
    candidate_pid: Optional[int]
    candidate_pgid: Optional[int]
    candidate_exit: Optional[int]
    timed_out: bool
    restoration: Optional[RestorationRecord]
    post_state_fresh: dict
    captured_utc: str
    duration_seconds: float

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)


def _verify_port_closed(*, port: int, timeout_seconds: float = 5.0) -> bool:
    """Return True once the TCP probe shows the port has gone away.

    Used after the trap exits to verify the candidate's listening
    port is genuinely gone (proves the process tree was reaped).
    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return True
        except OSError:
            return True
        time.sleep(0.1)
    return False


# ---------------------------------------------------------------------------
# Measurement protocol
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MeasurementContext:
    """Handle handed to the measurement callable.

    The callable may call ``register_candidate_pid(pid)`` and/or
    ``register_candidate_pgid(pgid)`` immediately after launching the
    candidate process so the restore trap can kill it when the
    lifecycle ends (timeout, signal, exception, FAIL). Prefer PGID
    (process-group kill) so the candidate's descendants (driver
    threads, child shells) are reaped too.

    The callable should return ``MeasurementResult`` carrying both the
    exit code AND the candidate PID (last known) so the orchestrator
    can record provenance even if the candidate crashed without
    ``register_candidate_pid`` being called.

    A plain ``int`` return is also accepted (legacy contract): exit 0
    means PASS, non-zero means FAIL.
    """
    register_candidate_pid: Callable[[int], None]
    register_candidate_pgid: Callable[[int], None]
    pre_state: PreStateSnapshot
    lab_gpu_uuid: str
    lab_gpu_bdf: str
    experiment_id: int


@dataclasses.dataclass(frozen=True)
class MeasurementResult:
    exit_code: int
    candidate_pid: Optional[int] = None
    timed_out: bool = False


MeasurementReturn = Union[int, MeasurementResult]


def _coerce_measurement_result(value: MeasurementReturn) -> MeasurementResult:
    if isinstance(value, MeasurementResult):
        return value
    if isinstance(value, int):
        return MeasurementResult(exit_code=value, candidate_pid=None)
    raise TypeError(
        f"measurement callable must return int or MeasurementResult, got {type(value).__name__}"
    )


# ---------------------------------------------------------------------------
# Allowlist drain (H0.6.1: drain BEFORE unknown-owner gate)
# ---------------------------------------------------------------------------


def _resolve_service_main_pid(service_name: str) -> Optional[int]:
    """Return the MainPID of a systemd service, or None.

    Best-effort lookup. We parse ``systemctl show <name> -p MainPID`` and
    return the PID. The service may have already exited (returns a
    string like ``[not set]`` or an empty string); we return None
    in that case.
    """
    try:
        import subprocess
        out = subprocess.run(
            ["systemctl", "show", service_name, "-p", "MainPID", "--value"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    raw = out.stdout.strip()
    if not raw or raw.isalpha():
        return None
    try:
        pid = int(raw)
    except ValueError:
        return None
    return pid if pid > 0 else None


def _resolve_service_main_pid_and_pgid(
    service_name: str,
) -> tuple[Optional[int], Optional[int]]:
    """Resolve the service's MainPID AND its process-group ID.

    The MainPID alone is unreliable after a stop (the service may have
    already cleared it). For draining GPU services the PGID is more
    useful: a service launched by systemd in user mode joins a
    dedicated PGID.
    """
    pid = _resolve_service_main_pid(service_name)
    if pid is None:
        return None, None
    try:
        pgid = os.getpgid(pid)
    except (OSError, ProcessLookupError):
        pgid = None
    return pid, pgid


def _drain_allowlist_pre(
    services: tuple[str, ...],
    *,
    trap,
) -> tuple[set[int], set[int], dict[str, str]]:
    """Resolve PIDs/PGIDs BEFORE stopping services AND register each
    stop with the active RestoreTrap so the trap restarts them on
    every exit (including BLOCKED / exception paths).
    """
    pids: set[int] = set()
    pgids: set[int] = set()
    reasons: dict[str, str] = {}
    for name in services:
        if not allowlist.is_active(name):
            continue
        pid, pgid = _resolve_service_main_pid_and_pgid(name)
        if pid is not None:
            pids.add(pid)
        if pgid is not None:
            pgids.add(pgid)
        reasons[name] = "stopped for exclusive-XTX window"
        print('A', file=__import__('sys').stderr); evt = allowlist.stop_service(name); print('B', file=__import__('sys').stderr); print('evt=', repr(evt), file=__import__('sys').stderr)
        print('about to record', file=__import__('sys').stderr); trap.record_stop_event(evt); print('after record, _stopped=', list(trap._stopped), file=__import__('sys').stderr)  # H0.6.3: trap owns the restart contract
    return pids, pgids, reasons


def _wait_for_pids_to_clear(
    pids: set[int], *, timeout_seconds: float = 10.0
) -> set[int]:
    """Wait (briefly) for each PID to terminate; return the survivors.

    Uses ``os.kill(pid, 0)`` which is the correct primitive for a PID
    (not ``os.killpg`` — that operates on process-group IDs).

    Polls at 0.2s intervals up to ``timeout_seconds``. Returns the set
    of PIDs that have not exited by the deadline — those still consume
    VRAM and may need manual intervention.
    """
    return _alive_after_deadline(
        probe=lambda p: _pid_alive(p),
        pids=pids,
        timeout_seconds=timeout_seconds,
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _pgid_alive(pgid: int) -> bool:
    """Check process-group existence via ``os.killpg(pgid, 0)``.

    C5: a PGID is not necessarily a PID — usually a new session's PGID
    equals its leader PID but calling ``os.kill`` on a PGID treats it
    as if it were a PID and gets ESRCH for valid groups, which is
    misleading. Use ``os.killpg``.
    """
    try:
        os.killpg(pgid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _wait_for_pgids_to_clear(
    pgids: set[int], *, timeout_seconds: float = 10.0
) -> set[int]:
    """PGID companion to :func:`_wait_for_pids_to_clear` (C5)."""
    return _alive_after_deadline(
        probe=lambda p: _pgid_alive(p),
        pids=pgids,
        timeout_seconds=timeout_seconds,
    )


def _alive_after_deadline(
    *,
    probe: Callable[[int], bool],
    pids: set[int],
    timeout_seconds: float,
) -> set[int]:
    survivors: set[int] = set()
    deadline = time.monotonic() + timeout_seconds
    remaining = set(pids)
    while remaining and time.monotonic() < deadline:
        still_alive: set[int] = set()
        for pid in remaining:
            try:
                if probe(pid):
                    still_alive.add(pid)
            except (OSError, ProcessLookupError):
                pass
        if not still_alive:
            return survivors
        remaining = still_alive
        time.sleep(0.2)
    return remaining


# ---------------------------------------------------------------------------
# Fresh post-state probes (H0.6.1)
# ---------------------------------------------------------------------------


def _fresh_post_state(snapshot: PreStateSnapshot) -> dict:
    """Probe services / listeners / health / GPU owners FRESH (H0.6.1).

    Never reads from ``snapshot``. The pre-state is used only for
    comparison (was this state restored?).
    """
    services = [
        {"name": svc.name, "is_active": svc.is_active, "is_enabled": svc.is_enabled}
        for svc in [
            service_state(name) for name in (
                "qwen38-turboquant.service",
                *(s.name for s in snapshot.services if s.name != "qwen38-turboquant.service"),
            )
        ]
    ]
    listeners = [
        {
            "port": l.port,
            "listening": listener_state(l.port).listening,
            "process": listener_state(l.port).process,
        }
        for l in snapshot.listeners
    ]
    production_health = production_health_ok(port=18079) if listener_state(18079).listening else False
    telemetry = gpu_telemetry()
    foreign_owners = []
    if snapshot.lab_gpu:
        foreign_owners = [
            dataclasses.asdict(o)
            for o in attribute_to_lab_gpu(
                lab_gpu_bdf=snapshot.lab_gpu.bdf,
                lab_gpu_uuid=snapshot.lab_gpu.uuid,
            )
        ]
    return {
        "services": services,
        "listeners": listeners,
        "production_health_ok": production_health,
        "gpu_telemetry": dataclasses.asdict(telemetry),
        "foreign_owners": foreign_owners,
        "captured_utc": utc_now(),
    }


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def run_lifecycle(
    *,
    measurement: Callable[[MeasurementContext], MeasurementReturn],
    experiment_id: int,
    artifact_dir: str | Path,
    allowlist: tuple[str, ...] = DEFAULT_ALLOWLIST(),
    lab_gpu_uuid: Optional[str] = None,
    lab_gpu_bdf: Optional[str] = None,
    allowed_pids: tuple[int, ...] = (),
    snapshot_override: Optional[PreStateSnapshot] = None,
    timeout_seconds: Optional[float] = None,
    drain_wait_seconds: float = 10.0,
) -> LifecycleAudit:
    """Run the H0.6 lifecycle around the caller's measurement callable.

    H0.6.3 sequencing — the RestoreTrap is active from drain start
    through audit write:

        snapshot
        → acquire flock
        → with RestoreTrap:
            stop allowlisted services → trap.record_stop(...)
            wait for pre-stop PIDs / PGIDs
            attribute foreign owners → BLOCKED if any
            run measurement (timeout via SIGALRM, trap record on exit)
            kill candidate + restart services (trap __exit__)
            capture ONE post-state snapshot
            derive post_state_restored from pre vs post
        → verify production, port, descendants → can flip PASS to FAIL
        → release lock
        → write audit (with the SAME post_state used for the decision)

    The trap wraps drain + measurement + post-state capture. A
    BLOCKED during drain still restarts anything that was stopped,
    because the trap's __exit__ runs on every exit (success, raise,
    SystemExit).
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    pre_state = snapshot_override or take_snapshot()
    effective_uuid = lab_gpu_uuid or (
        pre_state.lab_gpu.uuid if pre_state.lab_gpu else ""
    )
    effective_bdf = lab_gpu_bdf or (
        pre_state.lab_gpu.bdf if pre_state.lab_gpu else ""
    )

    ctx = MeasurementContext(
        register_candidate_pid=lambda _pid: None,
        register_candidate_pgid=lambda _pgid: None,
        pre_state=pre_state,
        lab_gpu_uuid=effective_uuid,
        lab_gpu_bdf=effective_bdf,
        experiment_id=experiment_id,
    )

    lock = ExperimentLock(LOCK_PATH)
    lock_held = lock.acquire(timeout_seconds=0.0)
    if not lock_held:
        outcome = Outcome.BLOCKED
        reason = "another evaluator holds the exclusive XTX lock"
        audit = _finalize_audit(
            outcome, reason, pre_state, [], [], None, None, None, False, None,
            start,
        )
        write_audit(
            artifact_dir / f"lifecycle_{experiment_id}.json",
            pre_state=pre_state,
            exclusive_gate={"lab_gpu_uuid": effective_uuid, "lab_gpu_bdf": effective_bdf,
                             "foreign_owners": [], "services_stopped": []},
            experiment_id=experiment_id, candidate_pid=None, candidate_exit=None,
            restoration=RestorationRecord(
                candidate_pid=None, candidate_pgid=None, candidate_killed=False,
                services_restarted=[], restored_at_utc=utc_now(),
                post_state_restored=False,
            ),
            post_state={}, outcome=outcome.value,
        )
        return audit

    candidate_pid: Optional[int] = None
    candidate_pgid: Optional[int] = None
    candidate_exit: Optional[int] = None
    timed_out: bool = False
    restoration: Optional[RestorationRecord] = None
    outcome: Outcome = Outcome.FAIL
    reason: str = "lifecycle did not reach PASS"
    foreign_owners: list[ProcessGpuOwner] = []
    post_state: dict = {}

    try:
        with RestoreTrap(pre_state) as trap:
            pre_stop_pids, pre_stop_pgids, _drain_reasons = (
                _drain_allowlist_pre(services=allowlist, trap=trap)
            )

            survivors_p = _wait_for_pids_to_clear(
                pre_stop_pids, timeout_seconds=drain_wait_seconds,
            )
            survivors_g = _wait_for_pgids_to_clear(
                pre_stop_pgids, timeout_seconds=drain_wait_seconds,
            )
            survivors = survivors_p | survivors_g
            if survivors:
                outcome = Outcome.BLOCKED
                reason = (
                    f"{len(survivors)} allowlisted service PID(s) survived "
                    f"drain: {sorted(survivors)}; not safe to proceed"
                )
            else:
                explicit_allowed = set(allowed_pids) | pre_stop_pids
                if effective_bdf and effective_uuid:
                    foreign_owners = attribute_to_lab_gpu(
                        lab_gpu_bdf=effective_bdf,
                        lab_gpu_uuid=effective_uuid,
                        allowed_pids=explicit_allowed,
                    )
                unknown_owners = [
                    o for o in foreign_owners if o.backend == "unknown"
                ]
                known_foreign = [
                    o for o in foreign_owners if o.backend != "unknown"
                ]
                allowed_for_filter = set(allowed_pids) | pre_stop_pids
                unknown_owners = [
                    o for o in unknown_owners if o.pid not in allowed_for_filter
                ]
                known_foreign = [
                    o for o in known_foreign if o.pid not in allowed_for_filter
                ]
                if unknown_owners:
                    outcome = Outcome.BLOCKED
                    reason = (
                        f"{len(unknown_owners)} UNKNOWN XTX owner(s); their "
                        "GPU attribution could not be resolved (CPU vs. GPU?). "
                        "Operator must investigate before allowing H0.6 to "
                        "attribute them."
                    )
                elif known_foreign:
                    outcome = Outcome.BLOCKED
                    reason = (
                        f"{len(known_foreign)} foreign XTX owner(s) not in "
                        "allowlist: must be killed or allowlisted explicitly."
                    )
                else:
                    ctx.register_candidate_pid = lambda pid: (
                        trap.set_candidate_pid(pid)
                    )
                    ctx.register_candidate_pgid = lambda pgid: (
                        trap.set_candidate_pgid(pgid)
                    )
                    ctx._deadline_seconds = timeout_seconds  # type: ignore[attr-defined]

                    raw_result, error = _run_measurement_with_sigalrm(
                        measurement=measurement,
                        ctx=ctx,
                        timeout_seconds=timeout_seconds,
                    )

                    if error is not None:
                        candidate_exit = -1
                        if (
                            "deadline" in str(error).lower()
                            or "timeout" in str(error).lower()
                        ):
                            timed_out = True
                            outcome = Outcome.TIMEOUT
                            reason = (
                                f"measurement raised a timeout-shaped "
                                f"error: {error!r}"
                            )
                        else:
                            outcome = Outcome.FAIL
                            reason = f"measurement raised: {error!r}"
                    elif raw_result is None:
                        candidate_exit = -1
                        outcome = Outcome.FAIL
                        reason = "measurement returned None"
                    else:
                        try:
                            coerced = _coerce_measurement_result(raw_result)
                        except TypeError as exc:
                            candidate_exit = -2
                            outcome = Outcome.FAIL
                            reason = str(exc)
                        else:
                            candidate_exit = coerced.exit_code
                            if coerced.candidate_pid is not None:
                                candidate_pid = coerced.candidate_pid
                            if (
                                coerced.timed_out
                                or (candidate_exit == 137 and timeout_seconds is not None)
                            ):
                                timed_out = True
                                outcome = Outcome.TIMEOUT
                                reason = (
                                    f"candidate exited with SIGKILL (137) "
                                    f"after {timeout_seconds:.0f}s deadline"
                                )
                            elif candidate_exit != 0:
                                outcome = Outcome.FAIL
                                reason = (
                                    f"candidate exited with non-zero code "
                                    f"{candidate_exit}"
                                )
                            else:
                                outcome = Outcome.PASS
                                reason = (
                                    "candidate exited 0; post-state to be verified"
                                )

                    port_check = int(
                        os.environ.get("HYPERLOOM_EXPERIMENT_PORT", "18180")
                    )
                    port_gone = _verify_port_closed(
                        port=port_check, timeout_seconds=5.0
                    )
                    if (candidate_pgid is not None or candidate_pid is not None):
                        post_kill_owners = (
                            attribute_to_lab_gpu(
                                lab_gpu_bdf=effective_bdf,
                                lab_gpu_uuid=effective_uuid,
                                allowed_pids=set(allowed_pids) | pre_stop_pids,
                            )
                            if effective_bdf and effective_uuid
                            else []
                        )
                        descendants = [
                            o for o in post_kill_owners
                            if o.pid not in set(allowed_pids) | pre_stop_pids
                        ]
                        if not port_gone and outcome == Outcome.PASS:
                            outcome = Outcome.FAIL
                            reason = f"{reason}; candidate port still listening"
                        if descendants and outcome == Outcome.PASS:
                            outcome = Outcome.FAIL
                            reason = (
                                f"{reason}; {len(descendants)} descendant "
                                f"XTX owner(s) still on GPU"
                            )

            # H0.6.3 / 6.3.3: ONE authoritative post-state snapshot is
            # captured AND used to verify restoration. The same
            # dict is persisted into the audit; we don't take a
            # second snapshot later (drift risk).
            post_state = _fresh_post_state(pre_state)

        # Capture the trap's restoration record AFTER the with-block
        # (the trap's record is populated in its __exit__).
        restoration = trap.record
        # H0.6.3 / 6.3.3: derive post_state_restored from the same
        # authoritative post_state we just captured.
        post_state_restored = _is_post_state_restored(pre_state, post_state)
        if restoration is not None:
            restoration = dataclasses.replace(
                restoration, post_state_restored=post_state_restored
            )
            candidate_pid = restoration.candidate_pid
            candidate_pgid = restoration.candidate_pgid
        if outcome == Outcome.PASS and not (
            restoration is not None and restoration.post_state_restored
        ):
            outcome = Outcome.FAIL
            reason = "post_state_restored is False; check restoration.services_restarted vs pre_state"
    finally:
        lock.release()

    services_restarted_objects = (
        restoration.services_restarted if restoration is not None else []
    )
    foreign_owners_at_acquire = foreign_owners
    audit = _finalize_audit(
        outcome, reason, pre_state,
        services_restarted_objects, foreign_owners_at_acquire,
        candidate_pid, candidate_pgid, candidate_exit, timed_out, restoration,
        post_state, start,
    )
    write_audit(
        artifact_dir / f"lifecycle_{experiment_id}.json",
        pre_state=pre_state,
        exclusive_gate={
            "lab_gpu_uuid": effective_uuid,
            "lab_gpu_bdf": effective_bdf,
            "foreign_owners": [dataclasses.asdict(o) for o in foreign_owners_at_acquire],
            "services_stopped": [
                e.service for e in services_restarted_objects
            ],
            "timed_out": timed_out,
            "candidate_pgid": candidate_pgid,
            "post_state_restored": (
                restoration.post_state_restored if restoration is not None else False
            ),
        },
        experiment_id=experiment_id,
        candidate_pid=candidate_pid,
        candidate_exit=candidate_exit,
        restoration=restoration or RestorationRecord(
            candidate_pid=candidate_pid, candidate_pgid=candidate_pgid,
            candidate_killed=False,
            services_restarted=[], restored_at_utc=utc_now(),
            post_state_restored=False,
        ),
        post_state=post_state,
        outcome=outcome.value,
    )
    return audit


class _MeasurementTimeout(BaseException):
    def __init__(self, seconds: float) -> None:
        super().__init__(f"measurement exceeded {seconds:.2f}s deadline")
        self.seconds = seconds


def _run_measurement_with_sigalrm(
    *,
    measurement: Callable[[MeasurementContext], MeasurementReturn],
    ctx: MeasurementContext,
    timeout_seconds: Optional[float],
) -> tuple[Optional[object], Optional[BaseException]]:
    """Run ``measurement`` with a SIGALRM hard deadline.

    H0.6.3 / 6.3.5: a real SIGALRM (Linux-only) is raised on the
    main thread. ``RestoreTrap.__exit__`` still runs on timeout,
    killing the candidate. Returns ``(raw_result, error)`` where
    ``error`` is a ``_MeasurementTimeout`` on timeout.
    """
    raw_result: Optional[object] = None
    error: Optional[BaseException] = None

    if timeout_seconds is None or not hasattr(signal, "SIGALRM"):
        try:
            raw_result = measurement(ctx)
        except BaseException as exc:  # noqa: BLE001
            error = exc
        return raw_result, error

    def _on_alarm(sig, frame):  # noqa: ARG001
        raise _MeasurementTimeout(timeout_seconds)

    previous = signal.signal(signal.SIGALRM, _on_alarm)
    interval = max(0.5, float(timeout_seconds))
    try:
        signal.setitimer(signal.ITIMER_REAL, interval)
        try:
            raw_result = measurement(ctx)
        except BaseException as exc:  # noqa: BLE001
            error = exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    return raw_result, error


def _is_post_state_restored(
    pre_state: PreStateSnapshot,
    post_state: dict,
) -> bool:
    """H0.6.3 / 6.3.3: deterministic pre-vs-post comparison.

    Compares the captured pre_state against the freshly probed post_state.
    All four checks must pass for True:

    1. listeners dict matches (port -> listening).
    2. services dict matches (name -> is_active).
    3. production 18079 listening AND healthy if pre had it up.
    4. no NEW foreign XTX owners appear (pre → post subset).
    """
    if not post_state:
        return False
    pre_listeners = {l.port: l.listening for l in pre_state.listeners}
    post_listeners = {
        l["port"]: l["listening"] for l in post_state.get("listeners", [])
    }
    if pre_listeners != post_listeners:
        return False
    pre_services = {s.name: s.is_active for s in pre_state.services}
    post_services = {
        s["name"]: s["is_active"] for s in post_state.get("services", [])
    }
    if pre_services != post_services:
        return False
    pre_prod_listening = next(
        (l.listening for l in pre_state.listeners if l.port == 18079), False
    )
    if pre_prod_listening and not post_state.get("production_health_ok"):
        return False
    # H0.6.3 / 6.3.3: post_state.foreign_owners must be a SUBSET of
    # pre_state.foreign_owners (no NEW XTX owners appeared).
    pre_owners = {o.pid for o in pre_state.gpu_processes}
    post_owners = {o["pid"] for o in post_state.get("foreign_owners", [])}
    if not post_owners.issubset(pre_owners):
        return False
    return True


def _finalize_audit(
    outcome: Outcome, reason: str, pre_state: PreStateSnapshot,
    owners_stopped: list[ServiceEvent],
    foreign_owners: list[ProcessGpuOwner],
    candidate_pid: Optional[int], candidate_pgid: Optional[int],
    candidate_exit: Optional[int],
    timed_out: bool, restoration: Optional[RestorationRecord],
    post_state_fresh: dict, start: float,
) -> LifecycleAudit:
    return LifecycleAudit(
        outcome=outcome, reason=reason, pre_state=pre_state,
        owners_stopped=owners_stopped,
        foreign_owners_at_acquire=foreign_owners,
        candidate_pid=candidate_pid,
        candidate_pgid=candidate_pgid,
        candidate_exit=candidate_exit,
        timed_out=timed_out,
        restoration=restoration,
        post_state_fresh=post_state_fresh,
        captured_utc=utc_now(),
        duration_seconds=time.monotonic() - start,
    )
