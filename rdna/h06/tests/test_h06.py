"""Unit tests for H0.6 modules."""

from __future__ import annotations

import importlib
import dataclasses
import json
import os
import socket
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "rdna"))

# Load the h06 modules via importlib so the relative imports inside the
# orchestrator resolve to the same h06 package. Inserting on sys.path
# alone would make orchestrator's ``from .allowlist import ...`` fail
# with "no known parent package".
importlib.import_module("h06")  # noqa: E402
for name in ("snapshot", "lock", "ownership", "allowlist", "restore", "orchestrator"):
    full = f"h06.{name}"
    if full not in sys.modules:
        importlib.import_module(full)
snap_mod = sys.modules["h06.snapshot"]
snap = snap_mod  # alias for the rest of the file
lock_mod = sys.modules["h06.lock"]
ownership = sys.modules["h06.ownership"]
allowlist = sys.modules["h06.allowlist"]
restore_mod = sys.modules["h06.restore"]
orchestrator = sys.modules["h06.orchestrator"]
Outcome = orchestrator.Outcome
run_lifecycle = orchestrator.run_lifecycle


# ---------------------------------------------------------------------------
# snapshot.py
# ---------------------------------------------------------------------------


class TestSnapshotProbes:
    def test_listener_state_free_port(self):
        # Bind a socket to claim a free port, close it, then probe.
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        finally:
            s.close()
        ls = snap.listener_state(port)
        assert ls.port == port
        assert ls.listening is False

    def test_listener_state_bound_port(self):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
            ls = snap.listener_state(port)
            assert ls.listening is True
        finally:
            s.close()


class TestTakeSnapshot:
    def test_take_snapshot_with_mocked_probes(self, monkeypatch):
        monkeypatch.setattr(snap, "detect_lab_gpu", lambda: snap.GpuIdentity(
            uuid="0x26592ee0cc915973", bdf="0000:c8:00.0", gfx_arch="gfx1100",
            board_name="AMD Radeon RX 7900 XTX", vendor="Advanced Micro Devices, Inc. [AMD/ATI]",
        ))
        monkeypatch.setattr(snap, "rocm_gpu_processes", lambda: [])
        monkeypatch.setattr(snap, "gpu_telemetry", lambda: snap.GpuTelemetry(
            vram_total_bytes=25 * 1024**3, vram_used_bytes=1 * 1024**3,
            vram_free_bytes=24 * 1024**3, temperature_c=45.0,
            sclk_mhz=2500, mclk_mhz=900, utilization_pct=2.0,
        ))
        monkeypatch.setattr(snap, "service_state", lambda n: snap.ServiceState(
            name=n, is_active="inactive", is_enabled=False,
        ))
        monkeypatch.setattr(snap, "_is_listening", lambda port, host="127.0.0.1": port == 18179)
        monkeypatch.setattr(snap, "production_health_ok", lambda **kw: False)
        s = snap.take_snapshot()
        assert s.lab_gpu is not None
        assert s.lab_gpu.uuid == "0x26592ee0cc915973"
        assert s.gpu_processes == []
        assert s.gpu_telemetry.vram_free_bytes == 24 * 1024**3
        # Service name should be the default.
        names = {s.name for s in s.services}
        assert "qwen38-turboquant.service" in names
        # Listener states.
        ports = {l.port: l.listening for l in s.listeners}
        assert ports[18079] is False
        assert ports[18179] is True
        assert s.production_health_ok is False


# ---------------------------------------------------------------------------
# lock.py
# ---------------------------------------------------------------------------


class TestExperimentLock:
    def test_acquire_release_round_trip(self, tmp_path):
        lock_path = tmp_path / "h06.lock"
        lock = lock_mod.ExperimentLock(lock_path)
        assert lock.acquire(timeout_seconds=0.0) is True
        assert lock.is_held is True
        lock.release()
        assert lock.is_held is False

    def test_second_acquire_blocks_non_blocking(self, tmp_path):
        lock_path = tmp_path / "h06.lock"
        a = lock_mod.ExperimentLock(lock_path)
        b = lock_mod.ExperimentLock(lock_path)
        assert a.acquire(timeout_seconds=0.0) is True
        # Non-blocking: b must fail.
        assert b.acquire(timeout_seconds=0.0) is False
        a.release()
        # After release b can acquire.
        assert b.acquire(timeout_seconds=0.0) is True
        b.release()


# ---------------------------------------------------------------------------
# ownership.py
# ---------------------------------------------------------------------------


class TestOwnership:
    def test_no_owners_on_clean_lab_gpu(self, monkeypatch):
        monkeypatch.setattr(ownership, "_rocm_pid_gpu_map", lambda: {})
        monkeypatch.setattr(ownership, "_drm_pid_gpu_map", lambda: {})
        owners = ownership.attribute_to_lab_gpu(
            lab_gpu_bdf="0000:c8:00.0", lab_gpu_uuid="0x26592ee0cc915973",
        )
        assert owners == []

    def test_allowed_pid_is_filtered(self, monkeypatch):
        monkeypatch.setattr(ownership, "_rocm_pid_gpu_map", lambda: {12345: 0})
        monkeypatch.setattr(ownership, "_drm_pid_gpu_map", lambda: {})
        owners = ownership.attribute_to_lab_gpu(
            lab_gpu_bdf="0000:c8:00.0", lab_gpu_uuid="0x26592ee0cc915973",
            allowed_pids=(12345,),
        )
        assert owners == []

    def test_rocm_owner_is_returned(self, monkeypatch):
        monkeypatch.setattr(ownership, "_rocm_pid_gpu_map", lambda: {4242: 0})
        monkeypatch.setattr(ownership, "_drm_pid_gpu_map", lambda: {})
        owners = ownership.attribute_to_lab_gpu(
            lab_gpu_bdf="0000:c8:00.0", lab_gpu_uuid="0x26592ee0cc915973",
        )
        assert len(owners) == 1
        assert owners[0].pid == 4242
        assert owners[0].backend == "rocm"

    def test_drm_owner_with_matching_bdf_returned(self, monkeypatch):
        monkeypatch.setattr(ownership, "_rocm_pid_gpu_map", lambda: {})
        monkeypatch.setattr(ownership, "_drm_pid_gpu_map", lambda: {9001: "/dev/dri/card0"})
        monkeypatch.setattr(ownership, "_drm_card_to_bdf", lambda: {"/dev/dri/card0": "0000:c8:00.0"})
        owners = ownership.attribute_to_lab_gpu(
            lab_gpu_bdf="0000:c8:00.0", lab_gpu_uuid="0x26592ee0cc915973",
        )
        assert len(owners) == 1
        assert owners[0].pid == 9001
        assert owners[0].backend == "drm"

    def test_drm_owner_on_other_gpu_filtered(self, monkeypatch):
        monkeypatch.setattr(ownership, "_rocm_pid_gpu_map", lambda: {})
        monkeypatch.setattr(ownership, "_drm_pid_gpu_map", lambda: {9001: "/dev/dri/card1"})
        monkeypatch.setattr(ownership, "_drm_card_to_bdf", lambda: {"/dev/dri/card1": "0000:01:00.0"})
        owners = ownership.attribute_to_lab_gpu(
            lab_gpu_bdf="0000:c8:00.0", lab_gpu_uuid="0x26592ee0cc915973",
        )
        assert owners == []


# ---------------------------------------------------------------------------
# allowlist.py
# ---------------------------------------------------------------------------


class TestAllowlist:
    def test_default_allowlist_is_empty(self):
        assert allowlist.DEFAULT_ALLOWLIST() == ()

    def test_stop_service_records_event(self, monkeypatch):
        # Simulate a successful systemctl stop.
        monkeypatch.setattr(
            allowlist, "_systemctl",
            lambda *args, **kw: (0, ""),
        )
        evt = allowlist.stop_service("qwen38-turboquant.service")
        assert evt.service == "qwen38-turboquant.service"
        assert evt.action == "stop"
        assert evt.returncode == 0


# ---------------------------------------------------------------------------
# restore.py
# ---------------------------------------------------------------------------


class TestRestoreTrap:
    def _fake_snapshot(self) -> snap.PreStateSnapshot:
        return snap.PreStateSnapshot(
            captured_utc="2026-08-17T00:00:00Z",
            lab_gpu=None, gpu_processes=[], gpu_telemetry=snap.GpuTelemetry(
                vram_total_bytes=0, vram_used_bytes=0, vram_free_bytes=0,
                temperature_c=None, sclk_mhz=None, mclk_mhz=None, utilization_pct=None,
            ),
            services=[], listeners=[],
            production_health_ok=False,
        )

    def test_candidate_kill_on_exit(self, monkeypatch):
        # Pretend we launched candidate PID 55555; kill succeeds.
        kill_calls = []
        monkeypatch.setattr(restore_mod, "_kill_pid", lambda pid, **kw: kill_calls.append(pid) or True)
        monkeypatch.setattr(restore_mod, "start_service",
                            lambda *a, **kw: allowlist.ServiceEvent(
                                service="x", action="start",
                                started_utc=utc_now(),  # noqa: F821 - defined in test fixture
                                finished_utc=utc_now(),
                                returncode=0,
                            ))
        trap = restore_mod.RestoreTrap(
            self._fake_snapshot(), candidate_pid=55555, services_to_restart=[],
        )
        with trap:
            pass
        assert kill_calls == [55555]
        assert trap.record is not None
        assert trap.record.candidate_killed is True

    def test_service_restart_on_exit(self, monkeypatch):
        kill_calls = []
        monkeypatch.setattr(restore_mod, "_kill_pid", lambda pid, **kw: kill_calls.append(pid) or True)
        starts = []
        monkeypatch.setattr(restore_mod, "start_service",
                            lambda name, **kw: starts.append(name) or allowlist.ServiceEvent(
                                service=name, action="start",
                                started_utc="2026-08-17T00:00:00Z",
                                finished_utc="2026-08-17T00:00:00Z",
                                returncode=0,
                            ))
        trap = restore_mod.RestoreTrap(
            self._fake_snapshot(), candidate_pid=None,
            services_to_restart=[
                allowlist.ServiceEvent(
                    service="foo.service", action="stop",
                    started_utc="2026-08-17T00:00:00Z",
                    finished_utc="2026-08-17T00:00:00Z",
                    returncode=0,
                ),
            ],
        )
        with trap:
            pass
        assert starts == ["foo.service"]
        assert trap.record is not None
        assert len(trap.record.services_restarted) == 1

    def test_restore_is_idempotent(self, monkeypatch):
        monkeypatch.setattr(restore_mod, "_kill_pid", lambda pid, **kw: True)
        trap = restore_mod.RestoreTrap(self._fake_snapshot(), candidate_pid=1)
        with trap:
            pass
        first_record = trap.record
        with trap:  # nested — should not re-execute
            pass
        assert trap.record is first_record


# ---------------------------------------------------------------------------
# orchestrator.py
# ---------------------------------------------------------------------------


class TestRunLifecycle:
    def _fake_snapshot(self, *, gpu_uuid="0x26592ee0cc915973", gpu_bdf="0000:c8:00.0") -> snap.PreStateSnapshot:
        return snap.PreStateSnapshot(
            captured_utc="2026-08-17T00:00:00Z",
            lab_gpu=snap.GpuIdentity(
                uuid=gpu_uuid, bdf=gpu_bdf, gfx_arch="gfx1100",
                board_name="AMD Radeon RX 7900 XTX", vendor="AMD",
            ),
            gpu_processes=[],
            gpu_telemetry=snap.GpuTelemetry(
                vram_total_bytes=25 * 1024**3, vram_used_bytes=1 * 1024**3,
                vram_free_bytes=24 * 1024**3, temperature_c=45.0,
                sclk_mhz=2500, mclk_mhz=900, utilization_pct=2.0,
            ),
            services=[
                snap.ServiceState(name="qwen38-turboquant.service",
                                   is_active="inactive", is_enabled=False),
            ],
            listeners=[
                snap.ListenerState(port=18079, listening=False, process=None),
                snap.ListenerState(port=18179, listening=False, process=None),
            ],
            production_health_ok=False,
        )

    def _patch_foreign_owners(self, monkeypatch, owners):
        monkeypatch.setattr(orchestrator, "attribute_to_lab_gpu",
                            lambda **kw: owners)

    def test_pass_on_clean_lab_no_unknowns(self, tmp_path, monkeypatch):
        snap = self._fake_snapshot()
        self._patch_foreign_owners(monkeypatch, [])
        # Redirect the lock file to a tmp_path we can create.
        monkeypatch.setattr(orchestrator, "LOCK_PATH", tmp_path / "h06.lock")
        monkeypatch.setattr(orchestrator, "is_active", lambda name: False)
        monkeypatch.setattr(orchestrator, "production_health_ok", lambda **kw: True)
        monkeypatch.setattr(orchestrator, "stop_service",
                            lambda name, **kw: allowlist.ServiceEvent(
                                service=name, action="stop",
                                started_utc="2026-08-17T00:00:00Z",
                                finished_utc="2026-08-17T00:00:00Z",
                                returncode=0,
                            ))
        audit = run_lifecycle(
            measurement=lambda _ctx: 0,
            experiment_id=1,
            artifact_dir=tmp_path,
            snapshot_override=snap,
        )
        assert audit.outcome == Outcome.PASS
        # No services in the allowlist -> no service events.
        assert audit.owners_stopped == []
        assert audit.restoration is not None
        # Audit artifact was written.
        assert (tmp_path / "lifecycle_1.json").is_file()

    def test_blocked_on_unknown_owner(self, tmp_path, monkeypatch):
        snap = self._fake_snapshot()
        unknown = ownership.ProcessGpuOwner(
            pid=9999, gpu_uuid=snap.lab_gpu.uuid, backend="rocm", detail="device_index=0",
        )
        self._patch_foreign_owners(monkeypatch, [unknown])
        monkeypatch.setattr(orchestrator, "LOCK_PATH", tmp_path / "h06.lock")
        audit = run_lifecycle(
            measurement=lambda _ctx: 0,
            experiment_id=2,
            artifact_dir=tmp_path,
            snapshot_override=snap,
        )
        assert audit.outcome == Outcome.BLOCKED
        assert "unknown XTX owner" in audit.reason
        # The measurement callable must NOT have been invoked.
        assert audit.candidate_pid is None
        assert audit.restoration is None

    def test_unknown_owner_in_allowlist_passes(self, tmp_path, monkeypatch):
        snap = self._fake_snapshot()
        known = ownership.ProcessGpuOwner(
            pid=9999, gpu_uuid=snap.lab_gpu.uuid, backend="rocm", detail="device_index=0",
        )
        self._patch_foreign_owners(monkeypatch, [known])
        monkeypatch.setattr(orchestrator, "LOCK_PATH", tmp_path / "h06.lock")
        monkeypatch.setattr(orchestrator, "is_active", lambda name: False)
        monkeypatch.setattr(orchestrator, "production_health_ok", lambda **kw: True)
        audit = run_lifecycle(
            measurement=lambda _ctx: 0,
            experiment_id=3,
            artifact_dir=tmp_path,
            snapshot_override=snap,
            allowed_pids=(9999,),
        )
        assert audit.outcome == Outcome.PASS
        # PID 9999 was filtered out of foreign_owners_at_acquire (the audit
        # field records the post-filter set so the operator sees what
        # would actually block).
        assert audit.foreign_owners_at_acquire == []

    def test_fail_when_measurement_raises(self, tmp_path, monkeypatch):
        snap = self._fake_snapshot()
        self._patch_foreign_owners(monkeypatch, [])
        monkeypatch.setattr(orchestrator, "LOCK_PATH", tmp_path / "h06.lock")
        monkeypatch.setattr(orchestrator, "is_active", lambda name: False)
        monkeypatch.setattr(orchestrator, "production_health_ok", lambda **kw: True)
        def boom(_ctx):
            raise RuntimeError("synthetic")
        with pytest.raises(RuntimeError):
            run_lifecycle(
                measurement=boom,
                experiment_id=4,
                artifact_dir=tmp_path,
                snapshot_override=snap,
            )

    def test_fail_when_production_drops(self, tmp_path, monkeypatch):
        snap = self._fake_snapshot()
        # Make 18079 listening before, but unhealthy after.
        snap = dataclasses.replace(
            snap,
            listeners=[
                snap_mod.ListenerState(port=18079, listening=True, process=None),
                snap_mod.ListenerState(port=18179, listening=False, process=None),
            ],
        )
        self._patch_foreign_owners(monkeypatch, [])
        monkeypatch.setattr(orchestrator, "LOCK_PATH", tmp_path / "h06.lock")
        monkeypatch.setattr(orchestrator, "is_active", lambda name: False)
        monkeypatch.setattr(orchestrator, "production_health_ok", lambda **kw: False)
        audit = run_lifecycle(
            measurement=lambda _ctx: 0,
            experiment_id=5,
            artifact_dir=tmp_path,
            snapshot_override=snap,
        )
        assert audit.outcome == Outcome.FAIL
        assert "production" in audit.reason.lower()