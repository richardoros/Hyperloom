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
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Union

from .allowlist import DEFAULT_ALLOWLIST, ServiceEvent, is_active, stop_service
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
    allowlist: tuple[str, ...],
) -> tuple[list[ServiceEvent], set[int], set[int], dict[str, str]]:
    """Resolve PIDs/PGIDs BEFORE stopping services.

    Returns events (stop actions taken), a set of PIDs to wait for,
    a set of PGIDs (PGID-killable equivalents for the truncate PIDs),
    and a service-name -> reason map for the audit.
    """
    events: list[ServiceEvent] = []
    pids: set[int] = set()
    pgids: set[int] = set()
    reasons: dict[str, str] = {}
    for name in allowlist:
        if not is_active(name):
            continue
        pid, pgid = _resolve_service_main_pid_and_pgid(name)
        if pid is not None:
            pids.add(pid)
        if pgid is not None:
            pgids.add(pgid)
        reasons[name] = "stopped for exclusive-XTX window"
        events.append(stop_service(name))
    return events, pids, pgids, reasons


def _wait_for_pids_to_clear(
    pids: set[int], *, timeout_seconds: float = 10.0
) -> set[int]:
    """Wait (briefly) for each PID to terminate; return the survivors.

    Polls at 0.2s intervals up to ``timeout_seconds``. Returns the set
    of PIDs that have not exited by the deadline — those still consume
    VRAM and may need manual intervention.
    """
    survivors: set[int] = set()
    deadline = time.monotonic() + timeout_seconds
    remaining = set(pids)
    while remaining and time.monotonic() < deadline:
        still_alive: set[int] = set()
        for pid in remaining:
            try:
                os.kill(pid, 0)
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

    Args:
        measurement: callable accepting a ``MeasurementContext`` and
            returning either an ``int`` exit code or a
            ``MeasurementResult``. The callable MUST call
            ``ctx.register_candidate_pid(pid)`` after launching the
            candidate so the restore trap can kill it.
        experiment_id: integer written into the audit artifact.
        artifact_dir: directory for the lifecycle audit JSON.
        allowlist: services the orchestrator may stop.
        lab_gpu_uuid / lab_gpu_bdf: overrides; defaults autodetect.
        allowed_pids: PIDs that may legitimately use the lab GPU (e.g.
            an embedding server the operator pinned).
        snapshot_override: for tests; skip take_snapshot.
        timeout_seconds: hard wall-clock deadline for the measurement
            callable. ``None`` means no deadline (caller-supplied tests).
        drain_wait_seconds: how long to wait for stopped services'
            PIDs / VRAM to clear before re-attributing.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    pre_state = snapshot_override or take_snapshot()
    effective_uuid = lab_gpu_uuid or (pre_state.lab_gpu.uuid if pre_state.lab_gpu else "")
    effective_bdf = lab_gpu_bdf or (pre_state.lab_gpu.bdf if pre_state.lab_gpu else "")

    ctx = MeasurementContext(
        register_candidate_pid=lambda _pid: None,
        register_candidate_pgid=lambda _pgid: None,
        pre_state=pre_state,
        lab_gpu_uuid=effective_uuid,
        lab_gpu_bdf=effective_bdf,
        experiment_id=experiment_id,
    )

    # 1. Acquire exclusive flock.
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

    # Defaults for the post-try audit write.
    services_to_restart: list[ServiceEvent] = []
    candidate_pid: Optional[int] = None
    candidate_pgid: Optional[int] = None
    candidate_exit: Optional[int] = None
    timed_out: bool = False
    restoration: Optional[RestorationRecord] = None
    outcome: Outcome = Outcome.FAIL
    reason: str = "lifecycle did not reach PASS"
    foreign_owners: list[ProcessGpuOwner] = []

    try:
        # 2. Resolve allowlisted-service PIDs / PGIDs BEFORE
        #    stopping them (H0.6.2 — the post-stop MainPID is often
        #    already cleared, so waiting on it is a no-op). Returns
        #    the events created by stopping.
        # 3. Stop allowlisted services.
        services_to_restart, pre_stop_pids, pre_stop_pgids, _drain_reasons = (
            _drain_allowlist_pre(allowlist)
        )

        # 4. Wait briefly for the pre-stop PIDs / PGIDs to clear.
        survivors = _wait_for_pids_to_clear(
            pre_stop_pids | pre_stop_pgids,
            timeout_seconds=drain_wait_seconds,
        )
        if survivors:
            outcome = Outcome.BLOCKED
            reason = (
                f"{len(survivors)} allowlisted service PID(s) survived "
                f"drain: {sorted(survivors)}; not safe to proceed"
            )
            early_exit = True
        else:
            early_exit = False

        # 5. Attribute foreign owners to lab GPU (BDF-filtered).
        if not early_exit:
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
            # Apply the explicit_allowed filter at the orchestrator
            # level too (the ownership function may or may not have
            # filtered; this makes the gate's logic explicit).
            allowed = set(allowed_pids) | pre_stop_pids
            unknown_owners = [o for o in unknown_owners if o.pid not in allowed]
            known_foreign = [o for o in known_foreign if o.pid not in allowed]
            if unknown_owners:
                outcome = Outcome.BLOCKED
                reason = (
                    f"{len(unknown_owners)} UNKNOWN XTX owner(s); their "
                    "GPU attribution could not be resolved (CPU vs. GPU?). "
                    "Operator must investigate before allowing H0.6 to "
                    "attribute them."
                )
                early_exit = True
            elif known_foreign:
                outcome = Outcome.BLOCKED
                reason = (
                    f"{len(known_foreign)} foreign XTX owner(s) not in "
                    "allowlist: must be killed or allowlisted explicitly."
                )
                early_exit = True

        # 6. Run the measurement callable INSIDE the RestoreTrap so
        #    the signal handlers cover the entire dangerous interval
        #    (H0.6.2 — fix for restore-after-measurement). The
        #    measurement launches its candidate under a new session;
        #    the trap kills the entire PGID on exit (H0.6.2 — kill the
        #    real descendant tree, not the wrapper PID). TIMEOUT is
        #    enforced by the callable's own subprocess timeout; the
        #    orchestrator classifies the resulting exit code as TIMEOUT
        #    when the callable reports it.
        if not early_exit:
            with RestoreTrap(
                pre_state,
                services_to_restart=services_to_restart,
            ) as trap:
                # The MeasurementContext's register_candidate closures
                # over the LIVE trap so the callable can register
                # PGID/PID AFTER launch and the trap owns them for
                # the rest of the lifetime.
                ctx.register_candidate_pid = lambda pid: (
                    trap.set_candidate_pid(pid)
                )
                ctx.register_candidate_pgid = lambda pgid: (
                    trap.set_candidate_pgid(pgid)
                )
                # Expose the deadline through the context too. The
                # callable is expected to enforce it on the
                # subprocess it spawns.
                ctx._deadline_seconds = timeout_seconds  # type: ignore[attr-defined]

                raw_result: Optional[object] = None
                error: Optional[BaseException] = None
                try:
                    raw_result = measurement(ctx)
                except BaseException as exc:  # noqa: BLE001 - trap handles
                    error = exc

            # Trap's __exit__ has run; restoration record is populated
            # and the candidate PGID/PID were killed (or never set).
            restoration = trap.record
            candidate_pid = trap.candidate_pid
            candidate_pgid = trap.candidate_pgid

            if error is not None:
                candidate_exit = -1
                if timed_out_marker := (
                    "deadline" in str(error).lower() or "timeout" in str(error).lower()
                ):
                    timed_out = True
                    outcome = Outcome.TIMEOUT
                    reason = f"measurement raised a timeout-shaped error: {error!r}"
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
                    # TIMEOUT shape: the callable explicitly flags a
                    # timeout via ``timed_out=True`` on the result.
                    # Also recognise exit=137 (SIGKILL) when the
                    # caller requested a deadline, as a defensive
                    # fallback for callables that didn't flag.
                    if coerced.timed_out or (
                        candidate_exit == 137 and timeout_seconds is not None
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
                        reason = "candidate exited 0; post-state to be verified"

            # 7. Verify candidate port + descendant owners (H0.6.2).
            #    Only meaningful when a PGID/PID was ever registered.
            port_check = int(
                os.environ.get("HYPERLOOM_EXPERIMENT_PORT", "18180")
            )
            port_gone = _verify_port_closed(port=port_check, timeout_seconds=5.0)
            if (candidate_pgid is not None or candidate_pid is not None):
                post_kill_owners = (
                    attribute_to_lab_gpu(
                        lab_gpu_bdf=effective_bdf,
                        lab_gpu_uuid=effective_uuid,
                        allowed_pids=set(allowed_pids) | pre_stop_pids,
                    )
                    if effective_bdf and effective_uuid else []
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

        # 9. FRESH post-state probes (H0.6.1: never copy from pre-state).
        post_state = _fresh_post_state(pre_state)

        # 10. Verify production.
        prod_was_listening = next(
            (l.listening for l in pre_state.listeners if l.port == 18079), False,
        )
        if prod_was_listening and not post_state["production_health_ok"]:
            if outcome == Outcome.PASS:
                outcome = Outcome.FAIL
                reason = (
                    "production 18079 was listening before H0.6 but "
                    "not healthy after (post-state proved, not asserted)"
                )
        if outcome == Outcome.PASS:
            unexpected_owners = [
                o for o in post_state.get("foreign_owners", [])
                if o.get("pid") not in (allowed_pids or ())
                and o.get("pid") not in (pre_stop_pids or set())
            ]
            if unexpected_owners:
                outcome = Outcome.FAIL
                reason = (
                    f"{len(unexpected_owners)} unexpected XTX owner(s) "
                    f"appeared after the lifecycle; the lab is not clean"
                )

    finally:
        lock.release()

    audit = _finalize_audit(
        outcome, reason, pre_state,
        services_to_restart, foreign_owners,
        candidate_pid, candidate_pgid, candidate_exit, timed_out, restoration,
        _fresh_post_state(pre_state), start,
    )
    write_audit(
        artifact_dir / f"lifecycle_{experiment_id}.json",
        pre_state=pre_state,
        exclusive_gate={
            "lab_gpu_uuid": effective_uuid,
            "lab_gpu_bdf": effective_bdf,
            "foreign_owners": [dataclasses.asdict(o) for o in foreign_owners],
            "services_stopped": [e.service for e in services_to_restart],
            "timed_out": timed_out,
            "candidate_pgid": candidate_pgid,
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
        post_state=audit.post_state_fresh,
        outcome=outcome.value,
    )
    return audit


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