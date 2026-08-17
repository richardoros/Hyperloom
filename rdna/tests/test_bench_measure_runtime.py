"""Runtime tests for rdna/bench/measure.py run_completion().

These tests exercise the actual HTTP + persistence path with a mocked
urlopen so they catch the kind of NameError / structural bug that the
H0.5.1 review caught (missing hashlib import, fixture schema, prompt_n
mismatch). The unit tests in test_decision / test_identity / test_db
do not exercise this path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "rdna" / "bench"))

import measure  # noqa: E402, PLC0415 - sys.path adjustment before import


def _fake_completion_body(prompt_n: int, eval_n: int = 128) -> dict:
    """Build a llama-server /completion response with buun timings keys."""
    return {
        "content": "The quick brown fox.",
        "timings": {
            "prompt_n": prompt_n,
            "prompt_ms": 4800.0,
            "predicted_n": eval_n,
            "predicted_ms": 5000.0,
            "prompt_per_second": prompt_n / 4.8,
            "predicted_per_second": eval_n / 5.0,
        },
        "context_size": prompt_n + eval_n,
    }


def _fixture(n_tokens: int = 1500) -> dict:
    return {
        "model_sha256": "deadbeef" * 8,
        "prompt_tokens": list(range(n_tokens)),
        "n_predict": 128,
        "seed": 0,
        "temperature": 0.0,
    }


def _make_response_mock(body: dict) -> mock.MagicMock:
    """Build a context-manager mock whose .read() returns the JSON body."""
    resp = mock.MagicMock()
    resp.read.return_value = json.dumps(body).encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


class TestRunCompletion:
    def test_sends_numeric_prompt_tokens(self, tmp_path):
        fixture = _fixture(n_tokens=1500)
        body = _fake_completion_body(prompt_n=1500)
        with mock.patch.object(measure, "urllib") as murl:
            murl.request.urlopen.return_value = _make_response_mock(body)
            rc = measure.run_completion(
                port=18179, model="/m.gguf",
                result_file=str(tmp_path / "out.json"),
                fixture=fixture, model_sha256="deadbeef" * 8,
            )
        assert rc == 0
        # The request payload sent to urlopen must carry the numeric
        # prompt_tokens (NOT a string). The Request is constructed as
        # urllib.request.Request(url, data=..., headers=...); the data
        # is bytes; decode it via .decode() rather than JSONDecoder
        # (MagicMock attribute, not a string).
        from urllib.request import Request
        sent_url, sent_data, sent_headers, *_ = (
            murl.request.Request.call_args[0],
        ) if False else (None,) * 4
        # Robust path: walk all Request() constructor calls and find
        # the one with data that contains our tokens.
        request_calls = murl.request.Request.call_args_list
        matched = []
        for c in request_calls:
            try:
                sent_data = c.kwargs.get("data") or (c.args[1] if len(c.args) > 1 else None)
                if sent_data is None:
                    continue
                payload = json.loads(sent_data)
                if "prompt" in payload and isinstance(payload["prompt"], list):
                    matched.append(payload)
            except (TypeError, json.JSONDecodeError):
                continue
        assert matched, "expected at least one Request with numeric prompt"
        sent = matched[-1]
        assert sent["prompt"] == list(range(1500))
        assert sent["n_predict"] == 128
        assert sent["seed"] == 0
        assert sent["temperature"] == 0.0

    def test_prompt_n_mismatch_is_fail_closed(self, tmp_path):
        # Server consumed fewer tokens than fixture provided → fail-closed.
        fixture = _fixture(n_tokens=1500)
        body = _fake_completion_body(prompt_n=1000)  # wrong
        with mock.patch.object(measure, "urllib") as murl:
            murl.request.urlopen.return_value = _make_response_mock(body)
            rc = measure.run_completion(
                port=18179, model="/m.gguf",
                result_file=str(tmp_path / "out.json"),
                fixture=fixture, model_sha256="deadbeef" * 8,
            )
        assert rc == 1
        payload = json.loads((tmp_path / "out.json").read_text())
        assert payload["success"] is False
        assert "prompt_n mismatch" in str(payload["errors"])

    def test_fixture_missing_prompt_tokens_rejected(self, tmp_path):
        fixture = {"model_sha256": "deadbeef" * 8, "n_predict": 128}
        with mock.patch.object(measure, "urllib") as murl:
            rc = measure.run_completion(
                port=18179, model="/m.gguf",
                result_file=str(tmp_path / "out.json"),
                fixture=fixture,
            )
        assert rc == 1
        assert murl.request.urlopen.call_count == 0  # fail closed before HTTP

    def test_fixture_with_non_integer_tokens_rejected(self, tmp_path):
        fixture = {"prompt_tokens": [1, "two", 3.0], "n_predict": 128}
        with mock.patch.object(measure, "urllib") as murl:
            rc = measure.run_completion(
                port=18179, model="/m.gguf",
                result_file=str(tmp_path / "out.json"),
                fixture=fixture,
            )
        assert rc == 1
        assert murl.request.urlopen.call_count == 0

    def test_fixture_gguf_mismatch_is_fail_closed(self, tmp_path):
        fixture = _fixture(n_tokens=1500)  # model_sha256 = "deadbeef" * 8
        with mock.patch.object(measure, "urllib") as murl:
            rc = measure.run_completion(
                port=18179, model="/m.gguf",
                result_file=str(tmp_path / "out.json"),
                fixture=fixture, model_sha256="f" * 64,  # wrong GGUF
            )
        assert rc == 1
        assert murl.request.urlopen.call_count == 0
        payload = json.loads((tmp_path / "out.json").read_text())
        assert "refusing to run with mismatched fixture" in str(payload["errors"])

    def test_result_payload_records_fixture_audit_fields(self, tmp_path):
        fixture = _fixture(n_tokens=1500)
        body = _fake_completion_body(prompt_n=1500)
        with mock.patch.object(measure, "urllib") as murl:
            murl.request.urlopen.return_value = _make_response_mock(body)
            rc = measure.run_completion(
                port=18179, model="/m.gguf",
                result_file=str(tmp_path / "out.json"),
                fixture=fixture, model_sha256="deadbeef" * 8,
            )
        assert rc == 0
        payload = json.loads((tmp_path / "out.json").read_text())
        # Audit fields that prove which exact-token workload ran.
        assert payload["fixture_prompt_tokens_len"] == 1500
        assert payload["fixture_n_predict"] == 128
        assert payload["fixture_seed"] == 0
        assert payload["fixture_temperature"] == 0.0
        assert payload["fixture_model_sha256"] == "deadbeef" * 8
        # fixture_hash exists and is a 64-char hex.
        assert len(payload["fixture_hash"]) == 64

    def test_http_failure_is_fail_closed(self, tmp_path):
        fixture = _fixture(n_tokens=1500)
        with mock.patch.object(measure, "urllib") as murl:
            murl.request.urlopen.side_effect = OSError("connection refused")
            rc = measure.run_completion(
                port=18179, model="/m.gguf",
                result_file=str(tmp_path / "out.json"),
                fixture=fixture,
            )
        assert rc == 1
        payload = json.loads((tmp_path / "out.json").read_text())
        assert "completion request failed" in str(payload["errors"])


class TestGenerateFixture:
    """Tokenize via a mocked llama-server, persist JSON, verify content."""

    def test_tokenize_and_persist(self, tmp_path, monkeypatch):
        tokenize_body = {"tokens": [1, 2, 3, 4, 5]}
        prompt_text = "The quick brown fox."
        out = tmp_path / "fixture.json"
        with mock.patch.object(measure, "urllib") as murl:
            murl.request.urlopen.return_value = _make_response_mock(tokenize_body)
            fixture = measure.generate_fixture(
                port=18179,
                prompt_text=prompt_text,
                model=str(tmp_path / "fake.gguf"),  # not a real file path; sha256 unknown
                output_path=str(out),
            )
        # Tokens came from the mocked /tokenize endpoint.
        assert fixture["prompt_tokens"] == [1, 2, 3, 4, 5]
        assert out.is_file()
        loaded = json.loads(out.read_text())
        assert loaded["prompt_tokens"] == [1, 2, 3, 4, 5]
        assert loaded["n_predict"] == 128
        # source_prompt_sha256 is the SHA of the input prompt text.
        assert len(loaded["source_prompt_sha256"]) == 64