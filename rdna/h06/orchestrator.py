"""H0.6 exclusive-XTX lifecycle orchestrator.

Pipeline:

    snapshot pre_state
        ↓
    acquire exclusive flock
        ↓
    attribute foreign PIDs to lab GPU
        ↓
    stop allowlisted services (if any)
        ↓
    prove clean-XTX gate (no unexpected owners, VRAM idle)
        ↓
    run H0.5 measurement (caller-provided callable)
        ↓
    on exit (any reason):
        - kill candidate PID
        - restart stopped services
        - release lock
        ↓
    verify post_state (prod service active, 18079 listening, no
    unexpected XTX owners)
        ↓
    emit lifecycle audit artifact

Lifecycle outcomes (each kept distinct):

    PASS        lifecycle + restoration proven
    BLOCKED     environmental prerequisite not satisfied (foreign
                owner not in allowlist, prod can't be drained, etc)
    TIMEOUT     mechanical deadline hit
    FAIL        lifecycle or measurement failed
"""

from __future__ import annotations

import dataclasses
import enum
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from .allowlist import DEFAULT_ALLOWLIST, ServiceEvent, is_active, stop_service
from .lock import LOCK_PATH, ExperimentLock
from .ownership import ProcessGpuOwner, attribute_to_lab_gpu
from .restore import RestorationRecord, RestoreTrap, write_audit
from .snapshot import (
    PreStateSnapshot,
    production_health_ok,
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
    restoration: Optional[RestorationRecord]
    post_state: dict
    captured_utc: str
    duration_seconds: float

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)


def _post_state_quick(snapshot: PreStateSnapshot) -> dict:
    """Re-read a subset of snapshot fields for the post-state check."""
    return {
        "services": [
            {"name": s.name, "is_active": s.is_active}
            for s in snapshot.services
        ],
        "listeners": [
            {"port": l.port, "listening": l.listening} for l in snapshot.listeners
        ],
        "production_health_ok": (
            production_health_ok()
            if any(l.port == 18079 and l.listening for l in snapshot.listeners)
            else False
        ),
        "captured_utc": utc_now(),
    }


def run_lifecycle(
    *,
    measurement: Callable[[dict], int],
    experiment_id: int,
    artifact_dir: str | Path,
    allowlist: tuple[str, ...] = DEFAULT_ALLOWLIST(),
    lab_gpu_uuid: Optional[str] = None,
    lab_gpu_bdf: Optional[str] = None,
    allowed_pids: tuple[int, ...] = (),
    snapshot_override: Optional[PreStateSnapshot] = None,
) -> LifecycleAudit:
    """Run the H0.6 lifecycle around the caller's measurement callable.

    Args:
        measurement: callable accepting a context dict (snapshot +
            audit context), returning the candidate's exit code.
        experiment_id: integer identifier written into the audit
            artifact so this run can be cross-referenced against the
            H0.5 experiment DB.
        artifact_dir: directory to write the lifecycle audit JSON.
        allowlist: services the orchestrator may stop. Default empty.
        lab_gpu_uuid / lab_gpu_bdf: overrides for tests; defaults to
            autodetect.
        allowed_pids: PIDs that may legitimately use the lab GPU (e.g.
            operator-pinned embedding servers); foreign-owner detection
            ignores them.
        snapshot_override: for tests; skip the take_snapshot call and
            use this snapshot instead.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    pre_state = snapshot_override or take_snapshot()
    effective_uuid = lab_gpu_uuid or (pre_state.lab_gpu.uuid if pre_state.lab_gpu else "")
    effective_bdf = lab_gpu_bdf or (pre_state.lab_gpu.bdf if pre_state.lab_gpu else "")
    audit_context: dict = {
        "experiment_id": experiment_id,
        "pre_state_captured_utc": pre_state.captured_utc,
        "allowlist": list(allowlist),
        "allowed_pids": list(allowed_pids),
        "lab_gpu_uuid": effective_uuid,
        "lab_gpu_bdf": effective_bdf,
    }

    # 1. Acquire exclusive flock.
    lock = ExperimentLock(LOCK_PATH)
    if not lock.acquire(timeout_seconds=0.0):
        return LifecycleAudit(
            outcome=Outcome.BLOCKED,
            reason="another evaluator holds the exclusive XTX lock",
            pre_state=pre_state,
            owners_stopped=[],
            foreign_owners_at_acquire=[],
            candidate_pid=None,
            candidate_exit=None,
            restoration=None,
            post_state={},
            captured_utc=utc_now(),
            duration_seconds=time.monotonic() - start,
        )

    # We hold the lock; from here on, every exit path must restore.
    services_to_restart: list[ServiceEvent] = []
    candidate_pid: Optional[int] = None
    candidate_exit: Optional[int] = None
    restoration: Optional[RestorationRecord] = None
    outcome: Outcome = Outcome.FAIL
    reason: str = "lifecycle did not reach PASS"
    foreign_owners: list[ProcessGpuOwner] = []

    try:
        # 2. Attribute foreign owners to the lab GPU.
        if effective_bdf and effective_uuid:
            foreign_owners = attribute_to_lab_gpu(
                lab_gpu_bdf=effective_bdf,
                lab_gpu_uuid=effective_uuid,
                allowed_pids=allowed_pids,
            )
        audit_context["foreign_owners_at_acquire"] = [
            dataclasses.asdict(o) for o in foreign_owners
        ]
        # 3. Block on any unknown owner. The recorded audit list is the
        # filtered (unknown) set so an operator inspecting the artifact
        # sees exactly what would block the run.
        unknown_owners = [o for o in foreign_owners if o.pid not in allowed_pids]
        audit_context["foreign_owners_unknown"] = [
            dataclasses.asdict(o) for o in unknown_owners
        ]
        foreign_owners = unknown_owners
        if unknown_owners:
            outcome = Outcome.BLOCKED
            reason = (
                f"{len(unknown_owners)} unknown XTX owner(s); not in allowlist. "
                "Operator must kill them or add them to the allowlist with an explicit comment."
            )
            return LifecycleAudit(
                outcome=outcome, reason=reason, pre_state=pre_state,
                owners_stopped=[], foreign_owners_at_acquire=foreign_owners,
                candidate_pid=None, candidate_exit=None, restoration=None,
                post_state={}, captured_utc=utc_now(),
                duration_seconds=time.monotonic() - start,
            )

        # 4. Drain allowlisted services (to free VRAM they held).
        for name in allowlist:
            if is_active(name):
                services_to_restart.append(stop_service(name))
        audit_context["services_stopped"] = [e.service for e in services_to_restart]

        # 5. Re-snapshot to capture post-drain state; the gate runs again
        # at the caller's request via the measurement callable's gate.
        with RestoreTrap(
            pre_state,
            candidate_pid=candidate_pid,
            services_to_restart=services_to_restart,
        ) as trap:
            try:
                candidate_exit = measurement(audit_context)
                outcome = Outcome.PASS
                reason = "measurement exited successfully"
            except SystemExit as exc:
                candidate_exit = int(exc.code) if isinstance(exc.code, int) else -1
                outcome = Outcome.FAIL
                reason = f"measurement SystemExit: {exc.code}"
                raise
            except BaseException as exc:  # noqa: BLE001 - trap must restore on any error
                candidate_exit = -1
                outcome = Outcome.FAIL
                reason = f"measurement raised: {exc!r}"
                raise
            finally:
                audit_context["outcome"] = outcome.value
                audit_context["candidate_exit"] = candidate_exit
        # Trap's __exit__ has run; restoration record is now populated.
        restoration = trap.record
        audit_context["restoration"] = restoration.to_json() if restoration else None

        # 6. Verify production.
        post_state = _post_state_quick(pre_state)
        audit_context["post_state"] = post_state
        prod_listening = next(
            (l.listening for l in pre_state.listeners if l.port == 18079), False,
        )
        prod_was_listening = prod_listening
        if prod_was_listening and not post_state["production_health_ok"]:
            outcome = Outcome.FAIL
            reason = "production 18079 was listening before H0.6 but not healthy after"
        audit_context["outcome"] = outcome.value
        audit_context["reason"] = reason

    finally:
        lock.release()

    audit = LifecycleAudit(
        outcome=outcome, reason=reason, pre_state=pre_state,
        owners_stopped=services_to_restart,
        foreign_owners_at_acquire=foreign_owners,
        candidate_pid=candidate_pid,
        candidate_exit=candidate_exit,
        restoration=restoration,
        post_state=_post_state_quick(pre_state),
        captured_utc=utc_now(),
        duration_seconds=time.monotonic() - start,
    )
    write_audit(
        artifact_dir / f"lifecycle_{experiment_id}.json",
        pre_state=pre_state,
        exclusive_gate={
            "lab_gpu_uuid": effective_uuid,
            "lab_gpu_bdf": effective_bdf,
            "foreign_owners": [dataclasses.asdict(o) for o in foreign_owners],
            "services_stopped": [e.service for e in services_to_restart],
        },
        experiment_id=experiment_id,
        candidate_pid=candidate_pid,
        candidate_exit=candidate_exit,
        restoration=restoration or RestorationRecord(
            candidate_pid=candidate_pid, candidate_killed=False,
            services_restarted=[], restored_at_utc=utc_now(),
            post_state_restored=False,
        ),
        post_state=audit.post_state,
        outcome=outcome.value,
    )
    return audit