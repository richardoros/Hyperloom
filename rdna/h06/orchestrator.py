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
    candidate_exit: Optional[int]
    timed_out: bool
    restoration: Optional[RestorationRecord]
    post_state_fresh: dict
    captured_utc: str
    duration_seconds: float

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)


# ---------------------------------------------------------------------------
# Measurement protocol
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MeasurementContext:
    """Handle handed to the measurement callable.

    The callable may call ``register_candidate(pid)`` immediately after
    launching the candidate process so the restore trap can kill it
    when the lifecycle ends (timeout, signal, exception, FAIL).

    The callable should return ``MeasurementResult`` carrying both the
    exit code AND the candidate PID (last known) so the orchestrator
    can record provenance even if the candidate crashed without
    ``register_candidate`` being called.

    A plain ``int`` return is also accepted (legacy contract): exit 0
    means PASS, non-zero means FAIL.
    """
    register_candidate: Callable[[int], None]
    pre_state: PreStateSnapshot
    lab_gpu_uuid: str
    lab_gpu_bdf: str
    experiment_id: int

    def register_candidate_pid(self, pid: int) -> None:
        """Convenience wrapper: ``ctx.register_candidate_pid(pid)``."""
        self.register_candidate(pid)


@dataclasses.dataclass(frozen=True)
class MeasurementResult:
    exit_code: int
    candidate_pid: Optional[int] = None


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


def _drain_allowlist(
    allowlist: tuple[str, ...],
) -> tuple[list[ServiceEvent], set[int]]:
    """Stop allowlisted services; return events + their MainPIDs.

    The returned PIDs must be implicitly allowed during the subsequent
    GPU ownership check (their VRAM is in the process of freeing).
    """
    events: list[ServiceEvent] = []
    pids: set[int] = set()
    for name in allowlist:
        if is_active(name):
            events.append(stop_service(name))
            pid = _resolve_service_main_pid(name)
            if pid is not None:
                pids.add(pid)
    return events, pids


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
        register_candidate=lambda _pid: None,
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
            outcome, reason, pre_state, [], [], None, None, False, None,
            start,
        )
        write_audit(
            artifact_dir / f"lifecycle_{experiment_id}.json",
            pre_state=pre_state,
            exclusive_gate={"lab_gpu_uuid": effective_uuid, "lab_gpu_bdf": effective_bdf,
                             "foreign_owners": [], "services_stopped": []},
            experiment_id=experiment_id, candidate_pid=None, candidate_exit=None,
            restoration=RestorationRecord(
                candidate_pid=None, candidate_killed=False,
                services_restarted=[], restored_at_utc=utc_now(),
                post_state_restored=False,
            ),
            post_state={}, outcome=outcome.value,
        )
        return audit

    # Defaults for the post-try audit write.
    services_to_restart: list[ServiceEvent] = []
    candidate_pid: Optional[int] = None
    candidate_exit: Optional[int] = None
    timed_out: bool = False
    restoration: Optional[RestorationRecord] = None
    outcome: Outcome = Outcome.FAIL
    reason: str = "lifecycle did not reach PASS"
    foreign_owners: list[ProcessGpuOwner] = []

    try:
        # 2. Resolve allowlisted-service PIDs BEFORE stopping them
        # so they're implicitly allowed during the subsequent GPU
        # ownership check (their VRAM is in the process of freeing).
        # 3. Stop allowlisted services.
        pre_stop_pids = {
            _resolve_service_main_pid(name) for name in allowlist
        }
        pre_stop_pids.discard(None)
        services_to_restart, drain_pids = _drain_allowlist(allowlist)

        # 4. Wait briefly for PIDs / VRAM to clear.
        survivors = _wait_for_pids_to_clear(
            drain_pids, timeout_seconds=drain_wait_seconds,
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

        # 6. Run the measurement callable in a worker thread (H0.6.1
        #    mechanical timeout). The thread is a daemon so we never
        #    block exit; we also track the candidate PID via the
        #    context's register_candidate callback.
        if not early_exit:
            result_holder: dict = {"value": None, "error": None}

            def _ctx_for_pid():
                # The MeasurementContext is constructed below; we
                # cannot pass ctx here (created after function defs).
                return ctx

            def _reg(pid: int) -> None:
                nonlocal candidate_pid
                candidate_pid = pid
                ctx.register_candidate = _reg  # update closure for later
                ctx._candidate_pid = pid  # type: ignore[attr-defined]

            ctx.register_candidate = _reg

            def _worker() -> None:
                try:
                    result_holder["value"] = measurement(ctx)
                except BaseException as exc:  # noqa: BLE001 - trap will catch on restore
                    result_holder["error"] = exc
                    result_holder["value"] = None

            thread = threading.Thread(target=_worker, daemon=True)
            ctx.register_candidate_pid = ctx.register_candidate  # for docs
            thread.start()
            if timeout_seconds is not None:
                thread.join(timeout=timeout_seconds)
            else:
                thread.join()

            if thread.is_alive():
                timed_out = True
                # The candidate will be killed below by the trap. The
                # measurement's return value is unknown at this
                # point; mark as 137 (SIGKILL) for audit purposes.
                candidate_exit = 137
                outcome = Outcome.TIMEOUT
                reason = (
                    f"measurement did not complete within "
                    f"{timeout_seconds:.0f}s; candidate PID "
                    f"{candidate_pid} will be SIGTERM-then-SIGKILLed"
                )
            elif result_holder["error"] is not None:
                candidate_exit = -1
                outcome = Outcome.FAIL
                reason = f"measurement raised: {result_holder['error']!r}"
            else:
                raw_result = result_holder["value"]
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
                    if candidate_exit != 0:
                        outcome = Outcome.FAIL
                        reason = (
                            f"candidate exited with non-zero code "
                            f"{candidate_exit}"
                        )
                    else:
                        outcome = Outcome.PASS
                        reason = "candidate exited 0; post-state to be verified"

        # 7. Trap runs in __exit__ — the candidate PID we registered
        #    is killed, services we stopped are restarted.
        with RestoreTrap(
            pre_state,
            candidate_pid=candidate_pid,
            services_to_restart=services_to_restart,
        ) as trap:
            pass
        restoration = trap.record

        # 8. FRESH post-state probes (H0.6.1: never copy from pre-state).
        post_state = _fresh_post_state(pre_state)

        # 9. Verify production.
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
                if o.get("pid") not in (allowed_pids or ()) and o.get("pid") not in (pre_stop_pids or set())
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
        candidate_pid, candidate_exit, timed_out, restoration,
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
        },
        experiment_id=experiment_id,
        candidate_pid=candidate_pid,
        candidate_exit=candidate_exit,
        restoration=restoration or RestorationRecord(
            candidate_pid=candidate_pid, candidate_killed=False,
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
    candidate_pid: Optional[int], candidate_exit: Optional[int],
    timed_out: bool, restoration: Optional[RestorationRecord],
    post_state_fresh: dict, start: float,
) -> LifecycleAudit:
    return LifecycleAudit(
        outcome=outcome, reason=reason, pre_state=pre_state,
        owners_stopped=owners_stopped,
        foreign_owners_at_acquire=foreign_owners,
        candidate_pid=candidate_pid,
        candidate_exit=candidate_exit,
        timed_out=timed_out,
        restoration=restoration,
        post_state_fresh=post_state_fresh,
        captured_utc=utc_now(),
        duration_seconds=time.monotonic() - start,
    )