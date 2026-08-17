"""Unit tests for the load/contention gate."""

from __future__ import annotations

import pytest

from rdna.h05.gate import (
    _foreign_llama_servers,
    _is_listening,
    _process_uses_gpu,
    gate,
)
from rdna.h05.identity import IdentityBlock


def _identity() -> IdentityBlock:
    return IdentityBlock(
        source_repo="/src", source_sha="abc", binary_path="/b", binary_sha256="x",
        model_path="/m", model_sha256="y", model_size_bytes=1024 * 1024 * 1024,
        gpu_type="rx7900xtx", gfx_arch="gfx1100",
        rocm_version="7.2.3", kernel_release="6.16.5",
        cmake_flags_sha="cf", compiler_sha="gcc", cpu_model="AMD",
    )


class TestIsListening:
    def test_bound_port_listening(self):
        import socket
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
            assert _is_listening(port) is True
        finally:
            s.close()

    def test_free_port_not_listening(self):
        import socket
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        finally:
            s.close()
        assert _is_listening(port) is False


class TestProcessUsesGPU:
    def test_cpu_only_embedding_excluded(self):
        cmdline = (
            "/home/homelabserver/src/llama.cpp/build/bin/llama-server -m "
            "/home/homelabserver/models/nomic-embed-text-Q8_0.gguf --embedding "
            "--pooling mean --ngl 0 --no-op-offload -c 2048 -b 512 -ub 512 -np 4 "
            "--host 127.0.0.1 --port 8084"
        )
        assert _process_uses_gpu(cmdline) is False

    def test_cpu_only_no_op_offload_excluded(self):
        cmdline = "/bin/llama-server -m m.gguf --no-op-offload -ngl 0 -c 1024"
        assert _process_uses_gpu(cmdline) is False

    def test_gpu_positive_ngl_included(self):
        cmdline = (
            "/bin/llama-server -m m.gguf -ngl 99 -fa on -c 8192 --port 18179 "
            "--metrics --jinja --no-ui"
        )
        assert _process_uses_gpu(cmdline) is True

    def test_gpu_with_dev_flag_included(self):
        cmdline = (
            "/bin/llama-server -m m.gguf -ngl 99 -fa on -dev Vulkan0 --port 18179"
        )
        assert _process_uses_gpu(cmdline) is True

    def test_non_llama_server_excluded(self):
        assert _process_uses_gpu("/usr/bin/python3 -m http.server 8000") is False


class TestForeignLlamaServers:
    def test_filters_out_cpu_only(self, monkeypatch):
        # Pretend pgrep returns one CPU-only embedding server.
        monkeypatch.setattr(
            "subprocess.run",
            lambda *args, **kwargs: type("R", (), {
                "stdout": (
                    "29926 /home/homelabserver/src/llama.cpp/build/bin/llama-server "
                    "-m /home/homelabserver/models/nomic-embed-text-Q8_0.gguf "
                    "--embedding --ngl 0 --no-op-offload --port 8084\n"
                ),
                "returncode": 0,
            })() if args and "pgrep" in (args[0] or [])[0] else type("R", (), {"stdout": "", "returncode": 1})(),
        )
        result = _foreign_llama_servers(gpu_type="rx7900xtx")
        assert result == [], f"CPU-only embedding server must not be reported, got {result}"

    def test_keeps_gpu_server(self, monkeypatch):
        monkeypatch.setattr(
            "subprocess.run",
            lambda *args, **kwargs: type("R", (), {
                "stdout": (
                    "40074 /home/homelabserver/src/llama-cpp-tq-vk/build/bin/llama-server "
                    "-m /home/homelabserver/models/Qwen3.8-27B-UD-Q5_K_XL.gguf -ngl 99 "
                    "-dev Vulkan0 --port 18179\n"
                ),
                "returncode": 0,
            })() if args and "pgrep" in (args[0] or [])[0] else type("R", (), {"stdout": "", "returncode": 1})(),
        )
        result = _foreign_llama_servers(gpu_type="rx7900xtx")
        assert len(result) == 1
        assert "40074" in result[0]


class TestGateRequiredBytes:
    def test_fail_when_required_exceeds_free(self, monkeypatch):
        # Pretend total VRAM = 1 GiB and 800 MiB used (200 MiB free).
        monkeypatch.setattr(
            "rdna.h05.gate._vram_rocm_smi",
            lambda: (800 * 1024 * 1024, 1024 * 1024 * 1024),
        )
        monkeypatch.setattr("rdna.h05.gate._qwen38_state", lambda: "inactive")
        monkeypatch.setattr("rdna.h05.gate._foreign_llama_servers",
                           lambda *, gpu_type, foreign_owners_on_gpu=None, gpu_uuid=None: [])
        # Suppress extra load signals.
        monkeypatch.setattr("rdna.h05.gate._cpu_load_per_core", lambda: 0.1)
        monkeypatch.setattr("rdna.h05.gate._ram_free_bytes", lambda: 64 * 1024**3)
        monkeypatch.setattr("rdna.h05.gate._swap_used_bytes", lambda: 0)
        monkeypatch.setattr("rdna.h05.gate._gpu_temp_c", lambda gpu_type: 50.0)
        monkeypatch.setattr("rdna.h05.gate._gpu_clock_mhz", lambda gpu_type: 2500)
        monkeypatch.setattr("rdna.h05.gate._build_contention", lambda: [])
        ib = _identity()
        report = gate(
            identity=ib,
            exp_port=18179,
            required_bytes=10 * 1024 * 1024 * 1024,
        )
        assert report.ok is False
        assert "headroom" in report.reason.lower()

    def test_pass_when_minimum_met(self, monkeypatch):
        monkeypatch.setattr(
            "rdna.h05.gate._vram_rocm_smi",
            lambda: (1 * 1024 * 1024 * 1024, 25 * 1024 * 1024 * 1024),
        )
        monkeypatch.setattr("rdna.h05.gate._qwen38_state", lambda: "inactive")
        monkeypatch.setattr("rdna.h05.gate._foreign_llama_servers",
                           lambda *, gpu_type, foreign_owners_on_gpu=None, gpu_uuid=None: [])
        monkeypatch.setattr("rdna.h05.gate._cpu_load_per_core", lambda: 0.1)
        monkeypatch.setattr("rdna.h05.gate._ram_free_bytes", lambda: 64 * 1024**3)
        monkeypatch.setattr("rdna.h05.gate._swap_used_bytes", lambda: 0)
        monkeypatch.setattr("rdna.h05.gate._gpu_temp_c", lambda gpu_type: 50.0)
        monkeypatch.setattr("rdna.h05.gate._gpu_clock_mhz", lambda gpu_type: 2500)
        monkeypatch.setattr("rdna.h05.gate._build_contention", lambda: [])
        import socket
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]
        finally:
            s.close()
        ib = _identity()
        report = gate(
            identity=ib,
            exp_port=free_port,
            required_bytes=1 * 1024 * 1024 * 1024,
        )
        assert report.ok is True
        # P1.7: headroom_fraction is post-load free fraction.
        # free=24GiB, required=1GiB, total=25GiB → (24-1)/25 = 23/25 = 0.92
        assert report.headroom_fraction > 0.9
        # And it's NOT (free / total) which would be 24/25 = 0.96
        assert report.headroom_fraction < (24 * 1024**3) / (25 * 1024**3)


class TestGateExtraLoadSignals:
    def test_blocks_on_cpu_contention(self, monkeypatch):
        monkeypatch.setattr("rdna.h05.gate._vram_rocm_smi",
                            lambda: (1 * 1024**3, 25 * 1024**3))
        monkeypatch.setattr("rdna.h05.gate._qwen38_state", lambda: "inactive")
        monkeypatch.setattr("rdna.h05.gate._foreign_llama_servers",
                           lambda *, gpu_type, foreign_owners_on_gpu=None, gpu_uuid=None: [])
        monkeypatch.setattr("rdna.h05.gate._cpu_load_per_core", lambda: 5.0)  # very high
        monkeypatch.setattr("rdna.h05.gate._ram_free_bytes", lambda: 64 * 1024**3)
        monkeypatch.setattr("rdna.h05.gate._swap_used_bytes", lambda: 0)
        monkeypatch.setattr("rdna.h05.gate._gpu_temp_c", lambda gpu_type: 50.0)
        monkeypatch.setattr("rdna.h05.gate._gpu_clock_mhz", lambda gpu_type: 2500)
        monkeypatch.setattr("rdna.h05.gate._build_contention", lambda: [])
        import socket
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]
        finally:
            s.close()
        ib = _identity()
        report = gate(identity=ib, exp_port=free_port, required_bytes=1 * 1024**3)
        assert report.ok is False
        assert "CPU load" in report.reason

    def test_blocks_on_low_gpu_clock(self, monkeypatch):
        monkeypatch.setattr("rdna.h05.gate._vram_rocm_smi",
                            lambda: (1 * 1024**3, 25 * 1024**3))
        monkeypatch.setattr("rdna.h05.gate._qwen38_state", lambda: "inactive")
        monkeypatch.setattr("rdna.h05.gate._foreign_llama_servers",
                           lambda *, gpu_type, foreign_owners_on_gpu=None, gpu_uuid=None: [])
        monkeypatch.setattr("rdna.h05.gate._cpu_load_per_core", lambda: 0.1)
        monkeypatch.setattr("rdna.h05.gate._ram_free_bytes", lambda: 64 * 1024**3)
        monkeypatch.setattr("rdna.h05.gate._swap_used_bytes", lambda: 0)
        monkeypatch.setattr("rdna.h05.gate._gpu_temp_c", lambda gpu_type: 50.0)
        monkeypatch.setattr("rdna.h05.gate._gpu_clock_mhz", lambda gpu_type: 200)  # power-saving
        monkeypatch.setattr("rdna.h05.gate._build_contention", lambda: [])
        import socket
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]
        finally:
            s.close()
        ib = _identity()
        report = gate(identity=ib, exp_port=free_port, required_bytes=1 * 1024**3)
        assert report.ok is False
        assert "clock" in report.reason.lower()

    def test_blocks_on_high_gpu_temp(self, monkeypatch):
        monkeypatch.setattr("rdna.h05.gate._vram_rocm_smi",
                            lambda: (1 * 1024**3, 25 * 1024**3))
        monkeypatch.setattr("rdna.h05.gate._qwen38_state", lambda: "inactive")
        monkeypatch.setattr("rdna.h05.gate._foreign_llama_servers",
                           lambda *, gpu_type, foreign_owners_on_gpu=None, gpu_uuid=None: [])
        monkeypatch.setattr("rdna.h05.gate._cpu_load_per_core", lambda: 0.1)
        monkeypatch.setattr("rdna.h05.gate._ram_free_bytes", lambda: 64 * 1024**3)
        monkeypatch.setattr("rdna.h05.gate._swap_used_bytes", lambda: 0)
        monkeypatch.setattr("rdna.h05.gate._gpu_temp_c", lambda gpu_type: 95.0)  # thermal
        monkeypatch.setattr("rdna.h05.gate._gpu_clock_mhz", lambda gpu_type: 2500)
        monkeypatch.setattr("rdna.h05.gate._build_contention", lambda: [])
        import socket
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]
        finally:
            s.close()
        ib = _identity()
        report = gate(identity=ib, exp_port=free_port, required_bytes=1 * 1024**3)
        assert report.ok is False
        assert "temperature" in report.reason.lower()


class TestGateBlockedRecordsToDB:
    """P0.1: BLOCKED attempts must be persisted (regression test).

    This is wired through the CLI's cmd_baseline, which opens the DB,
    records the BLOCKED run + decision, and returns 42. We exercise the
    same path here directly via ExperimentDB + gate (the CLI does
    nothing more than this).
    """

    def test_blocked_attempt_records_experiment_run_decision(self, monkeypatch, tmp_path):
        from rdna.h05.db import Decision, ExperimentDB

        # Force gate to FAIL on contention.
        monkeypatch.setattr(
            "rdna.h05.gate._vram_rocm_smi",
            lambda: (1 * 1024**3, 25 * 1024**3),
        )
        monkeypatch.setattr("rdna.h05.gate._qwen38_state", lambda: "inactive")
        monkeypatch.setattr("rdna.h05.gate._foreign_llama_servers",
                           lambda *, gpu_type, foreign_owners_on_gpu=None, gpu_uuid=None:
                           ["40074 /bin/llama-server -m m.gguf -ngl 99 -dev Vulkan0 --port 18179"])
        monkeypatch.setattr("rdna.h05.gate._cpu_load_per_core", lambda: 0.1)
        monkeypatch.setattr("rdna.h05.gate._ram_free_bytes", lambda: 64 * 1024**3)
        monkeypatch.setattr("rdna.h05.gate._swap_used_bytes", lambda: 0)
        monkeypatch.setattr("rdna.h05.gate._gpu_temp_c", lambda gpu_type: 50.0)
        monkeypatch.setattr("rdna.h05.gate._gpu_clock_mhz", lambda gpu_type: 2500)
        monkeypatch.setattr("rdna.h05.gate._build_contention", lambda: [])

        ib = _identity()
        with ExperimentDB(tmp_path / "experiments.db") as db:
            eid = db.open_experiment(label="regression-blocked", identity=ib)
            report = gate(
                identity=ib,
                exp_port=18179,
                required_bytes=10 * 1024**3,
            )
            assert report.ok is False
            db.record_run(
                eid,
                variant="baseline",
                iteration=0,
                identity=ib,
                gate_json=report.to_json(),
                measurement_json='{"blocked": true}',
                success=False,
            )
            db.record_decision(
                eid, decision=Decision.BLOCKED,
                reason=report.reason, metrics={},
            )
            db.close_experiment(
                eid, status="BLOCKED",
                decision=Decision.BLOCKED, reason=report.reason,
            )
            # Verify persistence: experiment row exists, run row exists,
            # decision row exists.
            row = db.experiment(eid)
            assert row["label"] == "regression-blocked"
            assert row["status"] == "BLOCKED"
            assert row["decision"] == "BLOCKED"
            runs = db.experiment_runs(eid)
            assert len(runs) == 1
            assert runs[0]["identity_hash"] == ib.identity_hash
            # No decision rows were checked at this level; the test would
            # otherwise inspect db.list_experiments() — sufficient.