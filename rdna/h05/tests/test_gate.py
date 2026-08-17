"""Unit tests for the load/contention gate."""

from __future__ import annotations

import pytest

from rdna.h05.gate import _is_listening, gate
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
        # Bind and release to claim a free port, then verify free.
        import socket
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        finally:
            s.close()
        assert _is_listening(port) is False


class TestGateRequiredBytes:
    def test_fail_when_required_exceeds_free(self, monkeypatch):
        # Pretend total VRAM = 1 GiB and 800 MiB used (200 MiB free).
        monkeypatch.setattr(
            "rdna.h05.gate._vram_rocm_smi",
            lambda: (800 * 1024 * 1024, 1024 * 1024 * 1024),
        )
        # Force qwen38 inactive, no foreign procs (no real reads needed here).
        monkeypatch.setattr("rdna.h05.gate._qwen38_state", lambda: "inactive")
        monkeypatch.setattr("rdna.h05.gate._foreign_llama_servers", lambda gpu: [])
        ib = _identity()
        report = gate(
            identity=ib,
            exp_port=18179,
            required_bytes=10 * 1024 * 1024 * 1024,  # 10 GiB > 200 MiB free
        )
        assert report.ok is False
        assert "headroom" in report.reason.lower()

    def test_pass_when_minimum_met(self, monkeypatch):
        monkeypatch.setattr(
            "rdna.h05.gate._vram_rocm_smi",
            lambda: (1 * 1024 * 1024 * 1024, 25 * 1024 * 1024 * 1024),
        )
        monkeypatch.setattr("rdna.h05.gate._qwen38_state", lambda: "inactive")
        monkeypatch.setattr("rdna.h05.gate._foreign_llama_servers", lambda gpu: [])
        # Find a free port to avoid 18179 which may be occupied on the host.
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
        assert report.headroom_fraction > 0.5