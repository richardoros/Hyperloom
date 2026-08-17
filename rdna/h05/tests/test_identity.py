"""Unit tests for the identity block."""

from __future__ import annotations

import json
import subprocess

import pytest

from rdna.h05.identity import (
    IdentityBlock,
    _GPU_TO_GFX,
    detect_gpu_type,
    detect_gpu_type as _detect_gpu_type_alias,  # alias used in CLI
    _infer_gfx_arch,
    _read_git_head,
    collect_identity,
)


class TestGfxArchTable:
    @pytest.mark.parametrize("gpu_type,expected", [
        ("rx7900xtx", "gfx1100"),
        ("radeon890m", "gfx1150"),
        ("mi300x", "gfx942"),
        ("mi355x", "gfx950"),
    ])
    def test_static_mapping(self, gpu_type, expected):
        assert _GPU_TO_GFX[gpu_type] == expected
        assert _infer_gfx_arch(gpu_type) == expected


class TestReadGitHead:
    def test_returns_none_for_non_repo(self, tmp_path):
        assert _read_git_head(tmp_path) is None

    def test_returns_sha_for_actual_repo(self):
        # The fork is a real git repo.
        head = _read_git_head("/home/homelabserver/hyperloom-rdna")
        assert head is not None
        assert len(head) >= 7  # git SHA hex


class TestIdentityBlock:
    def test_to_json_is_stable(self):
        ib = IdentityBlock(
            source_repo="/a",
            source_sha="abc",
            binary_path="/b",
            binary_sha256="x",
            model_path="/c",
            model_sha256="y",
            model_size_bytes=123,
            gpu_type="rx7900xtx",
            gfx_arch="gfx1100",
            rocm_version="7.2.3",
            kernel_release="6.16.5-200.fc42.x86_64",
            cmake_flags_sha="cf",
            compiler_sha="gcc",
            cpu_model="AMD Ryzen",
        )
        # Same input → same JSON.
        assert ib.to_json() == ib.to_json()
        # Identity hash is stable and excludes model_size_bytes.
        h1 = ib.identity_hash
        ib2 = dataclasses_replace(ib, model_size_bytes=999)  # noqa: F821
        assert ib2.identity_hash == h1

    def test_identity_hash_changes_with_real_changes(self):
        ib1 = IdentityBlock(
            source_repo="/a", source_sha="abc", binary_path="/b", binary_sha256="x",
            model_path="/c", model_sha256="y", model_size_bytes=123,
            gpu_type="rx7900xtx", gfx_arch="gfx1100",
            rocm_version="7.2.3", kernel_release="6.16.5",
            cmake_flags_sha="cf", compiler_sha="gcc", cpu_model="AMD",
        )
        ib2 = dataclasses_replace(ib1, binary_sha256="z")  # noqa: F821
        assert ib2.identity_hash != ib1.identity_hash


def dataclasses_replace(ib, **kwargs):
    """Tiny helper for dataclasses.replace without importing it everywhere."""
    import dataclasses as _dc
    return _dc.replace(ib, **kwargs)


class TestCollectIdentityAgainstProductionPins:
    def test_production_pin_path(self):
        """Verify collect_identity runs against the production pins without
        crashing; the values are operator-specific so we just assert shape."""
        ib = collect_identity(
            source_repo="/home/homelabserver/src/llama.cpp-turboquant",
            binary_path="/home/homelabserver/src/llama.cpp-turboquant/build/bin/llama-server",
            model_path="/home/homelabserver/models/Qwen3.8-27B-UD-Q5_K_XL.gguf",
            gpu_type="rx7900xtx",
        )
        assert ib.source_sha is not None and len(ib.source_sha) >= 7
        assert ib.binary_sha256 is not None and len(ib.binary_sha256) == 64
        assert ib.model_sha256 is not None and len(ib.model_sha256) == 64
        assert ib.model_size_bytes and ib.model_size_bytes > 1024**3
        assert ib.gpu_type == "rx7900xtx"
        assert ib.gfx_arch == "gfx1100"