"""Unit tests for build-from-SHA provenance."""

from __future__ import annotations

import pytest

from rdna.h05.build_provenance import hash_build_config


class TestHashBuildConfig:
    def test_whitespace_normalized(self):
        a = hash_build_config("-DGGML_HIPBLAS=ON  -DUSE_VULKAN=OFF")
        b = hash_build_config("-DGGML_HIPBLAS=ON -DUSE_VULKAN=OFF")
        assert a == b

    def test_empty(self):
        # Stable hash, never raises.
        assert len(hash_build_config("")) == 64

    def test_distinct_configs_differ(self):
        a = hash_build_config("-DGGML_HIPBLAS=ON")
        b = hash_build_config("-DGGML_HIPBLAS=OFF")
        assert a != b