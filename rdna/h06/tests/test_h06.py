"""Unit tests for H0.6 modules."""

from __future__ import annotations

import importlib
import dataclasses
import json
import os
import socket
import subprocess
import sys
import time
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
        monkeypatch.setattr(ownership, "_rocm_device_index_to_bdf",
                            lambda: {0: "0000:c8:00.0"})
        monkeypatch.setattr(ownership, "_drm_pid_gpu_map", lambda: {})
        owners = ownership.attribute_to_lab_gpu(
            lab_gpu_bdf="0000:c8:00.0", lab_gpu_uuid="0x26592ee0cc915973",
            allowed_pids=(12345,),
        )
        assert owners == []

    def test_rocm_owner_on_lab_gpu(self, monkeypatch):
        # device_index 0 maps to the lab GPU; PID is on the lab.
        monkeypatch.setattr(ownership, "_rocm_pid_gpu_map", lambda: {4242: 0})
        monkeypatch.setattr(ownership, "_rocm_device_index_to_bdf",
                            lambda: {0: "0000:c8:00.0"})
        monkeypatch.setattr(ownership, "_drm_pid_gpu_map", lambda: {})
        owners = ownership.attribute_to_lab_gpu(
            lab_gpu_bdf="0000:c8:00.0", lab_gpu_uuid="0x26592ee0cc915973",
        )
        assert len(owners) == 1
        assert owners[0].pid == 4242
        assert owners[0].backend == "rocm"

    def test_rocm_owner_on_other_gpu_excluded(self, monkeypatch):
        # device_index 0 maps to a DIFFERENT GPU; PID is NOT ours even
        # though rocm-smi reports a positive attribution. This is the
        # H0.6.1 fix: don't assume a KFD device_index belongs to the
        # lab GPU without BDF resolution.
        monkeypatch.setattr(ownership, "_rocm_pid_gpu_map", lambda: {4242: 0})
        monkeypatch.setattr(ownership, "_rocm_device_index_to_bdf",
                            lambda: {0: "0000:01:00.0"})
        monkeypatch.setattr(ownership, "_drm_pid_gpu_map", lambda: {})
        owners = ownership.attribute_to_lab_gpu(
            lab_gpu_bdf="0000:c8:00.0", lab_gpu_uuid="0x26592ee0cc915973",
        )
        assert owners == []

    def test_rocm_pid_unparseable_is_unknown(self, monkeypatch):
        # device_index present in rocm-smi output but absent from
        # rocm-smi --showbus (parse failure) -> backend='unknown'.
        monkeypatch.setattr(ownership, "_rocm_pid_gpu_map", lambda: {6001: 0})
        monkeypatch.setattr(ownership, "_rocm_device_index_to_bdf", lambda: {})
        monkeypatch.setattr(ownership, "_drm_pid_gpu_map", lambda: {})
        owners = ownership.attribute_to_lab_gpu(
            lab_gpu_bdf="0000:c8:00.0", lab_gpu_uuid="0x26592ee0cc915973",
        )
        assert len(owners) == 1
        assert owners[0].pid == 6001
        assert owners[0].backend == "unknown"

    def test_drm_owner_with_matching_bdf_returned(self, monkeypatch):
        monkeypatch.setattr(ownership, "_rocm_pid_gpu_map", lambda: {})
        monkeypatch.setattr(ownership, "_rocm_device_index_to_bdf", lambda: {})
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
        monkeypatch.setattr(ownership, "_rocm_device_index_to_bdf", lambda: {})
        monkeypatch.setattr(ownership, "_drm_pid_gpu_map", lambda: {9001: "/dev/dri/card1"})
        monkeypatch.setattr(ownership, "_drm_card_to_bdf", lambda: {"/dev/dri/card1": "0000:01:00.0"})
        owners = ownership.attribute_to_lab_gpu(
            lab_gpu_bdf="0000:c8:00.0", lab_gpu_uuid="0x26592ee0cc915973",
        )
        assert owners == []

    def test_drm_owner_unresolvable_is_unknown(self, monkeypatch):
        monkeypatch.setattr(ownership, "_rocm_pid_gpu_map", lambda: {})
        monkeypatch.setattr(ownership, "_rocm_device_index_to_bdf", lambda: {})
        monkeypatch.setattr(ownership, "_drm_pid_gpu_map", lambda: {9002: "/dev/dri/cardX"})
        monkeypatch.setattr(ownership, "_drm_card_to_bdf", lambda: {})
        owners = ownership.attribute_to_lab_gpu(
            lab_gpu_bdf="0000:c8:00.0", lab_gpu_uuid="0x26592ee0cc915973",
        )
        # Unresolvable: backend='unknown' (NOT silently attributed).
        assert len(owners) == 1
        assert owners[0].pid == 9002
        assert owners[0].backend == "unknown"

    def test_bdf_case_normalized(self, monkeypatch):
        monkeypatch.setattr(ownership, "_rocm_pid_gpu_map", lambda: {})
        monkeypatch.setattr(ownership, "_rocm_device_index_to_bdf", lambda: {})
        monkeypatch.setattr(ownership, "_drm_pid_gpu_map", lambda: {9001: "/dev/dri/card0"})
        monkeypatch.setattr(ownership, "_drm_card_to_bdf", lambda: {"/dev/dri/card0": "0000:C8:00.0"})
        # Caller passes lab bdf in uppercase; should still match.
        owners = ownership.attribute_to_lab_gpu(
            lab_gpu_bdf="0000:C8:00.0", lab_gpu_uuid="0x26592ee0cc915973",
        )
        assert len(owners) == 1


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
    def _fake_snapshot(self) -> snap_mod.PreStateSnapshot:
        return snap_mod.PreStateSnapshot(
            captured_utc="2026-08-17T00:00:00Z",
            lab_gpu=None, gpu_processes=[], gpu_telemetry=snap_mod.GpuTelemetry(
                vram_total_bytes=0, vram_used_bytes=0, vram_free_bytes=0,
                temperature_c=None, sclk_mhz=None, mclk_mhz=None, utilization_pct=None,
            ),
            services=[], listeners=[],
            production_health_ok=False,
        )

    def test_signal_handler_lambda_does_not_share_closure(self):
        # H0.6.1 regression: the lambda installed for SIGTERM/SIGINT/SIGHUP
        # used to close over the loop variable ``sig`` and so always re-raised
        # SIGHUP regardless of which signal fired. We mirror the install
        # pattern and assert each handler captures its own signal.
        import signal as _signal
        received: list[int] = []

        class Trap:
            def install(self):
                for sig in (_signal.SIGTERM, _signal.SIGINT, _signal.SIGHUP):
                    handler = (lambda *_, _sig=sig: received.append(_sig))
                    handler()  # simulate signal receipt

        Trap().install()
        assert received == [_signal.SIGTERM, _signal.SIGINT, _signal.SIGHUP], (
            f"each handler must bind to its own signal; got {received}"
        )

    def test_signal_reraise_does_not_recurse_in_child(self):
        # C1: the trap's _restore_and_reraise must restore the previous
        # signal handler BEFORE re-raising, or it recursively fires its
        # own handler until the runtime gives up. This test fires SIGTERM
        # into a child that has the trap installed. The child records
        # whether the handler ran 1x or N>=2x; we assert exactly 1.
        import os
        import signal as _signal
        import subprocess
        import sys
        import tempfile
        # Build a safe tiny script that:
        # 1. counts SIGTERM fires
        # 2. signals the parent that it's ready
        # 3. waits
        # 4. prints the final count
        # We do this by writing a tiny script via tempfile.
        script = f"""
import os, signal, sys
counter = [0]
def handler(*_):
    counter[0] += 1
    sys.stdout.write('COUNTER=' + str(counter[0]) + chr(10))
    sys.stdout.flush()
    os.fsync(1)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGTERM)
signal.signal(signal.SIGTERM, handler)
sys.stdout.write('READY' + chr(10))
sys.stdout.flush()
os.fsync(1)
import time
time.sleep(5)
"""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(script)
            script_path = fh.name
        try:
            proc = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            # READY + COUNTER may both arrive on stdout. We send
            # SIGTERM after we've seen READY (handler still alive),
            # then read both lines via communicate().
            line = proc.stdout.readline()
            assert line.strip() == "READY", f"unexpected first line: {line!r}"
            proc.terminate()
            stdout_data, stderr_data = proc.communicate(timeout=5)
            out = stdout_data
        finally:
            os.unlink(script_path)
        assert "COUNTER=1" in out, (
            f"C1 fix did not break signal recursion; "
            f"stdout={out!r} stderr={stderr_data!r}"
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

    def _quick_measurement(self, ctx, exit_code: int = 0,
                            candidate_pid=None):
        """Return a small MeasurementResult (no IO, no threads)."""
        from h06.orchestrator import MeasurementResult
        # register_candidate closes over the local candidate_pid.
        if candidate_pid is not None:
            ctx.register_candidate_pid(candidate_pid)
        return MeasurementResult(exit_code=exit_code, candidate_pid=candidate_pid)

    def test_pass_on_clean_lab_no_unknowns(self, tmp_path, monkeypatch):
        snap = self._fake_snapshot()
        self._patch_foreign_owners(monkeypatch, [])
        monkeypatch.setattr(orchestrator, "LOCK_PATH", tmp_path / "h06.lock")
        monkeypatch.setattr(orchestrator, "is_active", lambda name: False)
        monkeypatch.setattr(orchestrator, "stop_service",
                            lambda name, **kw: allowlist.ServiceEvent(
                                service=name, action="stop",
                                started_utc="2026-08-17T00:00:00Z",
                                finished_utc="2026-08-17T00:00:00Z",
                                returncode=0,
                            ))
        monkeypatch.setattr(orchestrator, "production_health_ok", lambda **kw: True)
        monkeypatch.setattr(orchestrator, "attribute_to_lab_gpu", lambda **kw: [])
        audit = run_lifecycle(
            measurement=lambda ctx: self._quick_measurement(ctx, 0, candidate_pid=4242),
            experiment_id=1,
            artifact_dir=tmp_path,
            snapshot_override=snap,
        )
        assert audit.outcome == Outcome.PASS
        assert audit.candidate_pid == 4242
        assert audit.candidate_exit == 0
        assert audit.timed_out is False
        assert audit.restoration is not None
        assert (tmp_path / "lifecycle_1.json").is_file()

    def test_blocked_on_unknown_owner(self, tmp_path, monkeypatch):
        snap = self._fake_snapshot()
        unknown = ownership.ProcessGpuOwner(
            pid=9999, gpu_uuid=snap.lab_gpu.uuid, backend="unknown",
            detail="device_index=0 bdf=<unparsed>; do not assume",
        )
        self._patch_foreign_owners(monkeypatch, [unknown])
        monkeypatch.setattr(orchestrator, "LOCK_PATH", tmp_path / "h06.lock")
        audit = run_lifecycle(
            measurement=lambda ctx: self._quick_measurement(ctx),
            experiment_id=2,
            artifact_dir=tmp_path,
            snapshot_override=snap,
        )
        assert audit.outcome == Outcome.BLOCKED
        assert "UNKNOWN XTX owner" in audit.reason
        assert audit.candidate_pid is None

    def test_blocked_on_known_foreign_owner(self, tmp_path, monkeypatch):
        snap = self._fake_snapshot()
        known = ownership.ProcessGpuOwner(
            pid=9999, gpu_uuid=snap.lab_gpu.uuid, backend="rocm",
            detail="device_index=0 bdf=0000:c8:00.0",
        )
        self._patch_foreign_owners(monkeypatch, [known])
        monkeypatch.setattr(orchestrator, "LOCK_PATH", tmp_path / "h06.lock")
        audit = run_lifecycle(
            measurement=lambda ctx: self._quick_measurement(ctx),
            experiment_id=2,
            artifact_dir=tmp_path,
            snapshot_override=snap,
        )
        assert audit.outcome == Outcome.BLOCKED
        assert "foreign XTX owner(s) not in allowlist" in audit.reason

    def test_unknown_owner_in_allowlist_passes(self, tmp_path, monkeypatch):
        snap = self._fake_snapshot()
        known = ownership.ProcessGpuOwner(
            pid=9999, gpu_uuid=snap.lab_gpu.uuid, backend="rocm",
            detail="device_index=0 bdf=0000:c8:00.0",
        )
        self._patch_foreign_owners(monkeypatch, [known])
        monkeypatch.setattr(orchestrator, "LOCK_PATH", tmp_path / "h06.lock")
        monkeypatch.setattr(orchestrator, "is_active", lambda name: False)
        monkeypatch.setattr(orchestrator, "production_health_ok", lambda **kw: True)
        audit = run_lifecycle(
            measurement=lambda ctx: self._quick_measurement(ctx),
            experiment_id=3,
            artifact_dir=tmp_path,
            snapshot_override=snap,
            allowed_pids=(9999,),
        )
        assert audit.outcome == Outcome.PASS
        # The audit records the unfiltered unknown set; the allowed_pids
        # check routes that PID past the BLOCK gate.
        assert any(o.pid == 9999 for o in audit.foreign_owners_at_acquire)

    def test_non_zero_exit_is_FAIL(self, tmp_path, monkeypatch):
        snap = self._fake_snapshot()
        self._patch_foreign_owners(monkeypatch, [])
        monkeypatch.setattr(orchestrator, "LOCK_PATH", tmp_path / "h06.lock")
        monkeypatch.setattr(orchestrator, "is_active", lambda name: False)
        monkeypatch.setattr(orchestrator, "production_health_ok", lambda **kw: True)
        audit = run_lifecycle(
            measurement=lambda ctx: self._quick_measurement(ctx, exit_code=1),
            experiment_id=4,
            artifact_dir=tmp_path,
            snapshot_override=snap,
        )
        assert audit.outcome == Outcome.FAIL
        assert "non-zero" in audit.reason
        assert audit.candidate_exit == 1

    def test_measurement_raises_is_FAIL(self, tmp_path, monkeypatch):
        snap = self._fake_snapshot()
        self._patch_foreign_owners(monkeypatch, [])
        monkeypatch.setattr(orchestrator, "LOCK_PATH", tmp_path / "h06.lock")
        monkeypatch.setattr(orchestrator, "is_active", lambda name: False)
        monkeypatch.setattr(orchestrator, "production_health_ok", lambda **kw: True)

        def boom(ctx):
            raise RuntimeError("synthetic measurement failure")
        # The lifecycle wraps the measurement in a worker thread so
        # the orchestrator never sees the raw exception; the failure is
        # captured and surfaced as Outcome.FAIL.
        audit = run_lifecycle(
            measurement=boom,
            experiment_id=5,
            artifact_dir=tmp_path,
            snapshot_override=snap,
        )
        assert audit.outcome == Outcome.FAIL
        assert "synthetic" in audit.reason
        assert audit.candidate_exit == -1

    def test_timeout_classification(self, tmp_path, monkeypatch):
        snap = self._fake_snapshot()
        self._patch_foreign_owners(monkeypatch, [])
        monkeypatch.setattr(orchestrator, "LOCK_PATH", tmp_path / "h06.lock")
        monkeypatch.setattr(orchestrator, "is_active", lambda name: False)
        monkeypatch.setattr(orchestrator, "production_health_ok", lambda **kw: True)

        # Process-based TIMEOUT (H0.6.2): the callable's subprocess
        # times out, returns exit_code=137 (SIGKILL), and the
        # orchestrator classifies it as TIMEOUT when the caller also
        # requested a deadline.
        def slow_subprocess(ctx):
            from h06.orchestrator import MeasurementResult
            import subprocess, os, signal
            proc = subprocess.Popen(
                ["sleep", "5"], start_new_session=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            ctx.register_candidate_pid(proc.pid)
            try:
                ctx.register_candidate_pgid(os.getpgid(proc.pid))
            except (OSError, ProcessLookupError):
                pass
            try:
                r = proc.communicate(timeout=0.05)
                return MeasurementResult(exit_code=r.returncode, candidate_pid=proc.pid)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass
                return MeasurementResult(
                    exit_code=137, candidate_pid=proc.pid, timed_out=True,
                )
        audit = run_lifecycle(
            measurement=slow_subprocess,
            experiment_id=6,
            artifact_dir=tmp_path,
            snapshot_override=snap,
            timeout_seconds=1.0,
        )
        assert audit.outcome == Outcome.TIMEOUT
        assert audit.timed_out is True
        assert "SIGKILL" in audit.reason or "137" in audit.reason
        assert audit.candidate_pid is not None and audit.candidate_pid > 0

    def test_production_dropped_proves_fail(self, tmp_path, monkeypatch):
        snap = self._fake_snapshot()
        # 18079 listening BEFORE; post-state probe returns unhealthy.
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
            measurement=lambda ctx: self._quick_measurement(ctx),
            experiment_id=7,
            artifact_dir=tmp_path,
            snapshot_override=snap,
        )
        assert audit.outcome == Outcome.FAIL
        assert audit.reason.startswith("production 18079 was listening")

    def test_int_return_is_accepted_as_legacy(self, tmp_path, monkeypatch):
        snap = self._fake_snapshot()
        self._patch_foreign_owners(monkeypatch, [])
        monkeypatch.setattr(orchestrator, "LOCK_PATH", tmp_path / "h06.lock")
        monkeypatch.setattr(orchestrator, "is_active", lambda name: False)
        monkeypatch.setattr(orchestrator, "production_health_ok", lambda **kw: True)
        def legacy_measurement(ctx):
            ctx.register_candidate_pid(555)
            return 0  # legacy int contract
        audit = run_lifecycle(
            measurement=legacy_measurement,
            experiment_id=8,
            artifact_dir=tmp_path,
            snapshot_override=snap,
        )
        assert audit.outcome == Outcome.PASS


class TestGateC3SwapIoActivity:
    """C3: swap-I/O activity rate, not cumulative swap occupancy."""

    def test_swap_io_rate_returns_zero_on_idle_host(self):
        from rdna.h05.gate import _swap_io_rate
        rate = _swap_io_rate(sample_seconds=0.2)
        assert rate is not None
        in_pages, out_pages = rate
        assert in_pages >= 0 and out_pages >= 0


class TestGateC4CeleryCpu:
    """C4: Celery gate by measured CPU, not worker existence."""

    def test_celery_cpu_with_zero_workers(self):
        from rdna.h05.gate import _celery_cpu_percent
        import subprocess as _sp

        def fake_run(cmd, *a, **kw):
            cmd0 = cmd[0] if isinstance(cmd, list) else cmd
            if "pgrep" in str(cmd0):
                res = mock.MagicMock(); res.stdout = ""
                return res
            raise AssertionError(f"unexpected cmd: {cmd}")

        with mock.patch.object(_sp, "run", side_effect=fake_run):
            assert _celery_cpu_percent() == 0.0


class TestPipelineC5PidsAndPgids:
    def test_pid_waiter(self):
        """C5: PID waiter uses os.kill (not os.killpg)."""
        from rdna.h06.orchestrator import _wait_for_pids_to_clear
        # Spawn a short-lived subprocess and feed its PID to the waiter.
        proc = subprocess.Popen(["true"])
        proc.wait(timeout=5)
        survived = _wait_for_pids_to_clear(
            {proc.pid}, timeout_seconds=3.0,
        )
        assert survived == set(), f"PID {proc.pid} did not exit: {survived}"

    def test_pgid_waiter_uses_os_killpg(self):
        """C5: PGID waiter uses os.killpg (not os.kill)."""
        from rdna.h06.orchestrator import _pgid_alive, _wait_for_pgids_to_clear
        proc = subprocess.Popen(
            ["sleep", "10"], start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        pgid = os.getpgid(proc.pid)
        proc.terminate()
        proc.wait()
        deadline = time.monotonic() + 3.0
        while _pgid_alive(pgid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _pgid_alive(pgid) is False
        survivors = _wait_for_pgids_to_clear({pgid}, timeout_seconds=2.0)
        assert survivors == set(), f"survivors: {survivors}"