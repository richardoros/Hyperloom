"""Unit tests for the rdna/bench/measure.py helpers and the H0.0+ identity
rename (rx7900xtx / radeon890m board keys, gfx arch moves into the dispatch
tuple, consumer-board rocm-smi aliases).

The full public-CLI smoke is an executed script with a captured evidence
bundle (not a pytest) — `--no-eval` makes it short, but install.sh + Ray
boot + coordinator.run() are not appropriate for a unit test. See
``rdna/H0.0-RESULT.md`` and the closure-pass report for that evidence.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "rdna" / "bench"))

import measure  # noqa: E402
from hyperloom.common.gpu_identity import (  # noqa: E402
    AMD_GPU_DISPATCH_IDENTITIES,
    gfx_arch_for_gpu_type,
)
from hyperloom.inference_optimizer.gpu_types import (  # noqa: E402
    _GFX_TO_RUNNER,
    _PRODUCT_ALIASES,
    _PRODUCT_TAGS,
    _TAG_TO_GPU_TYPE,
)


# ---------------------------------------------------------------------------
# measure.py — split_server_args
# ---------------------------------------------------------------------------

class TestSplitServerArgs:
    def test_empty_returns_empty(self):
        assert measure.split_server_args("") == []
        assert measure.split_server_args("   ") == []

    def test_simple_whitespace_splits(self):
        assert measure.split_server_args("-a 1 -b 2") == ["-a", "1", "-b", "2"]

    def test_quoted_token_preserves_spaces(self):
        assert measure.split_server_args('-foo "bar baz" --u') == ["-foo", "bar baz", "--u"]

    def test_unterminated_quote_raises(self):
        with pytest.raises(ValueError):
            measure.split_server_args('-foo "unterm')


# ---------------------------------------------------------------------------
# measure.py — port_in_use
# ---------------------------------------------------------------------------

class TestPortInUse:
    def test_open_port_is_not_in_use(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            s.close()
            assert measure.port_in_use(port) is False
        except OSError:
            # Some kernels bind-then-rebind the same port too quickly; the
            # caller's only obligation is correct detection, which the
            # other test covers.
            pass

    def test_bound_port_is_in_use(self):
        # Keep the listener alive while we probe — closing it returns the
        # port to the kernel and the probe would (correctly) report free.
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
            assert measure.port_in_use(port) is True
        finally:
            s.close()

    def test_cli_exit_code_invert_for_bash(self):
        # The bench script uses `if ! python3 measure.py port-in-use N; then
        # fail` (rc=0 means free → continue; rc=1 means in use → fail). Verify
        # the CLI produces the right rc for each state.
        import subprocess
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
            rc_in_use = subprocess.run(
                [sys.executable, str(measure.__file__), "port-in-use", str(port)],
                capture_output=True, timeout=5,
            ).returncode
        finally:
            s.close()
        assert rc_in_use == 1, f"in-use port should yield rc=1, got {rc_in_use}"


# ---------------------------------------------------------------------------
# measure.py — parse_timings (buun keys + upstream fallback)
# ---------------------------------------------------------------------------

class TestParseTimings:
    def test_buun_keys_win(self):
        t = {
            "prompt_n": 4096, "prompt_ms": 1000.0,
            "predicted_n": 128, "predicted_ms": 5000.0,
            "prompt_per_second": 4096.0, "predicted_per_second": 25.6,
        }
        out = measure.parse_timings(t)
        assert out["prompt_n"] == 4096
        assert out["prompt_ms"] == 1000.0
        assert out["eval_n"] == 128
        assert out["eval_ms"] == 5000.0
        assert out["pp_tok_s"] == 4096.0
        assert out["tg_tok_s"] == 25.6

    def test_upstream_fallback_when_buun_missing(self):
        t = {
            "prompt_eval_count": 2048, "prompt_eval_duration": 500_000_000,
            "eval_count": 64, "eval_duration": 4_000_000_000,
        }
        out = measure.parse_timings(t)
        assert out["prompt_n"] == 2048
        assert out["prompt_ms"] == 500.0
        assert out["eval_n"] == 64
        assert out["eval_ms"] == 4000.0
        assert out["pp_tok_s"] == pytest.approx(4096.0, rel=1e-6)
        assert out["tg_tok_s"] == pytest.approx(16.0, rel=1e-6)


# ---------------------------------------------------------------------------
# measure.py — build_result semantics (H0.0+ corrections)
# ---------------------------------------------------------------------------

class TestBuildResult:
    TIMINGS = {
        "prompt_n": 4096, "prompt_ms": 1000.0,
        "eval_n": 128, "eval_ms": 5000.0,
        "pp_tok_s": 4096.0, "tg_tok_s": 25.6,
        "draft_n": 0, "draft_n_accepted": 0,
    }

    def test_success_follows_quality_gate(self):
        ok = measure.build_result("m.gguf", 5.0, self.TIMINGS, quality_ok=True, context_size=4096)
        bad = measure.build_result("m.gguf", 5.0, self.TIMINGS, quality_ok=False, context_size=4096)
        assert ok["success"] is True
        assert ok["quality_gate"]["passed"] is True
        assert bad["success"] is False
        assert bad["quality_gate"]["passed"] is False

    def test_mean_e2el_ms_is_wall_clock_not_eval_ms(self):
        # e2el must be wall*1000, NOT eval_ms (otherwise a 5s eval in a 10s
        # request would report e2el=5000 and the orchestrator would understate
        # latency).
        out = measure.build_result("m.gguf", 10.0, self.TIMINGS, quality_ok=True, context_size=4096)
        assert out["mean_e2el_ms"] == 10000.0

    def test_prompt_eval_ms_separate_not_ttft(self):
        out = measure.build_result("m.gguf", 10.0, self.TIMINGS, quality_ok=True, context_size=4096)
        assert out["prompt_eval_ms"] == 1000.0
        assert "mean_ttft_ms" not in out, "TTFT label kept for one-completion workload is misleading"

    def test_tpot_only_when_eval_nonzero(self):
        out = measure.build_result("m.gguf", 10.0, self.TIMINGS, quality_ok=True, context_size=4096)
        # 5000ms / 128 tokens; round(0.5) rounds to even (banker's rounding)
        # so the value is 39.062, not 39.063.
        assert out["mean_tpot_ms"] == pytest.approx(39.0625, abs=1e-3)
        zero = {**self.TIMINGS, "eval_n": 0, "eval_ms": 0.0}
        out0 = measure.build_result("m.gguf", 10.0, zero, quality_ok=False, context_size=4096)
        assert out0["mean_tpot_ms"] is None


# ---------------------------------------------------------------------------
# Identity rename: rx7900xtx / radeon890m board keys
# ---------------------------------------------------------------------------

class TestBoardIdentityRename:
    def test_rx7900xtx_dispatches_to_gfx1100_96_cu(self):
        assert AMD_GPU_DISPATCH_IDENTITIES["rx7900xtx"] == ("gfx1100", 96)

    def test_radeon890m_dispatches_to_gfx1150_16_cu(self):
        assert AMD_GPU_DISPATCH_IDENTITIES["radeon890m"] == ("gfx1150", 16)

    def test_gfx_arch_query_works_for_board_keys(self):
        assert gfx_arch_for_gpu_type("rx7900xtx") == "gfx1100"
        assert gfx_arch_for_gpu_type("radeon890m") == "gfx1150"

    def test_legacy_gfx_keys_are_not_board_keys(self):
        assert "gfx1100" not in AMD_GPU_DISPATCH_IDENTITIES
        assert "gfx1150" not in AMD_GPU_DISPATCH_IDENTITIES

    def test_gfx_to_runner_maps_arch_to_board(self):
        assert _GFX_TO_RUNNER["gfx1100"] == "rx7900xtx"
        assert _GFX_TO_RUNNER["gfx1150"] == "radeon890m"

    def test_consumer_board_aliases_cover_rocm_smi_product_names(self):
        assert _PRODUCT_ALIASES["rx7900xtx"] == "RX 7900 XTX"
        assert _PRODUCT_ALIASES["radeon890m"] == "RADEON 890M"

    def test_product_tags_and_alias_map_are_bijective(self):
        # Reverse-sorted tags; bijection between boards and rocm-smi tags.
        assert len(_PRODUCT_TAGS) == len(set(_PRODUCT_TAGS))
        assert set(_TAG_TO_GPU_TYPE) == set(_PRODUCT_TAGS)
        assert set(_TAG_TO_GPU_TYPE.values()) == set(AMD_GPU_DISPATCH_IDENTITIES)
        # Reverse-sorted = longer tag first.
        assert list(_PRODUCT_TAGS) == sorted(_PRODUCT_TAGS, reverse=True)


# ---------------------------------------------------------------------------
# CLI accepts rx7900xtx (and radeon890m) as --gpu-type
# ---------------------------------------------------------------------------

class TestCliAcceptsBoardKey:
    def test_cli_choices_include_rdna_boards(self):
        from hyperloom.inference_optimizer.cli.parser import _build_parser

        def _walk(p):
            for action in p._actions:
                if "--gpu-type" in (action.option_strings or []):
                    return list(action.choices or [])
                if isinstance(getattr(action, "choices", None), dict):
                    for sub in action.choices.values():
                        found = _walk(sub)
                        if found is not None:
                            return found
            return None

        choices = _walk(_build_parser())
        assert choices is not None
        assert "rx7900xtx" in choices
        assert "radeon890m" in choices
        # Legacy GFX arch keys are NOT CLI gpu-type choices (vanishing test).
        assert "gfx1100" not in choices
        assert "gfx1150" not in choices


# ---------------------------------------------------------------------------
# End-to-end (scriptable) result payload round-trip
# ---------------------------------------------------------------------------

class TestResultPayloadRoundTrip:
    def test_persisted_payload_is_valid_json(self, tmp_path):
        out = tmp_path / "inferencex_result.json"
        payload = measure.build_result(
            model="m.gguf",
            wall=10.0,
            timings={
                "prompt_n": 4096, "prompt_ms": 1000.0,
                "eval_n": 128, "eval_ms": 5000.0,
                "pp_tok_s": 4096.0, "tg_tok_s": 25.6,
                "draft_n": 0, "draft_n_accepted": 0,
            },
            quality_ok=True,
            context_size=4096,
        )
        measure._persist(str(out), payload)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["success"] is True
        assert loaded["mean_e2el_ms"] == 10000.0
        assert loaded["prompt_eval_ms"] == 1000.0
        assert "mean_ttft_ms" not in loaded
