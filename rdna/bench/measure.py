#!/usr/bin/env python3
"""Pure-python helpers for the rdna custom-framework seam.

No GPU, no llama.cpp, no Hyperloom imports: the bench entrypoint delegates
its measured work here so the semantics are unit-testable outside a run.

Three responsibilities this module extracts from the bash script:

* ``split_server_args`` uses shlex to honor quotes and escaping -- the old
  bash ``read -r -a $EXTRA`` split on whitespace and silently mis-split
  quoted arguments.
* ``port_in_use`` does a TCP ``connect_ex`` probe so a foreign listener on
  the experiment lane fail-closes the run before the server launches.
* ``build_result`` fixes the H0.0 result semantics: ``success`` follows
  ``quality_gate.passed`` (a failing gate is a failure), ``mean_e2el_ms``
  is the wall-clock request latency (``wall * 1000``), and the prompt
  phase is exported as ``prompt_eval_ms`` -- never labeled TTFT, because
  this workload has exactly one completion per request.

The CLI subcommands exist for the bash entrypoint to call; the functions
above are what the unit tests exercise.
"""

from __future__ import annotations

import json
import shlex
import socket
import sys
import time
import urllib.request

FRAMEWORK = "custom"
WORKLOAD_KIND = "scriptable"
THROUGHPUT_UNIT = "tok/s"


def split_server_args(extra: str) -> list[str]:
    """Split ``EXTRA_CUSTOM_ARGS`` with shell quoting rules, failing on garbage.

    Empty/whitespace input returns ``[]``. Posix shlex raises ``ValueError``
    on unterminated quotes instead of silently mis-splitting, which is what
    the bash ``read -r -a`` did.
    """
    if not extra or not extra.strip():
        return []
    return shlex.split(extra, posix=True)


def port_in_use(port: int) -> bool:
    """True when something is already listening on 127.0.0.1:port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            return s.connect_ex(("127.0.0.1", int(port))) == 0
    except OSError:
        return True


def parse_timings(t: dict) -> dict:
    """Normalize buun-llama-cpp timings, falling back to upstream keys."""
    prompt_n = int(t.get("prompt_n") or t.get("prompt_eval_count") or 0)
    prompt_ms = (t.get("prompt_ms") or (t.get("prompt_eval_duration") or 0) / 1e6) or 0.0
    eval_n = int(t.get("predicted_n") or t.get("eval_count") or 0)
    eval_ms = (t.get("predicted_ms") or (t.get("eval_duration") or 0) / 1e6) or 0.0
    pp_tok_s = float(
        t.get("prompt_per_second")
        or (prompt_n / prompt_ms * 1000.0 if prompt_ms > 0 else 0.0)
    )
    tg_tok_s = float(
        t.get("predicted_per_second")
        or (eval_n / eval_ms * 1000.0 if eval_ms > 0 else 0.0)
    )
    return {
        "prompt_n": prompt_n,
        "prompt_ms": float(prompt_ms),
        "eval_n": eval_n,
        "eval_ms": float(eval_ms),
        "pp_tok_s": pp_tok_s,
        "tg_tok_s": tg_tok_s,
        "draft_n": int(t.get("draft_n") or 0),
        "draft_n_accepted": int(t.get("draft_n_accepted") or 0),
    }


def build_result(
    model: str,
    wall: float,
    timings: dict,
    quality_ok: bool,
    context_size: int,
) -> dict:
    """InferenceX-shaped result with the corrected H0.0+ semantics.

    ``success`` follows ``quality_gate.passed``; ``mean_e2el_ms`` is the
    wall-clock request latency; ``prompt_eval_ms`` is the prompt phase,
    never labeled TTFT (this workload has exactly one completion).
    """
    eval_n = timings["eval_n"]
    return {
        "framework": FRAMEWORK,
        "model_id": model,
        "workload_kind": WORKLOAD_KIND,
        "throughput_unit": THROUGHPUT_UNIT,
        "success": bool(quality_ok),
        "quality_gate": {"passed": bool(quality_ok)},
        "output_throughput": round(timings["tg_tok_s"], 3),
        "request_throughput": round(1.0 / wall, 3),
        "total_token_throughput": round((timings["prompt_n"] + eval_n) / wall, 3),
        "completed": 1,
        "total_input_tokens": timings["prompt_n"],
        "total_output_tokens": eval_n,
        "duration": round(wall, 3),
        "prompt_eval_ms": round(timings["prompt_ms"], 3),
        "mean_tpot_ms": round(timings["eval_ms"] / eval_n, 3) if eval_n else None,
        "mean_e2el_ms": round(wall * 1000.0, 3),
        "prompt_eval_tok_s": round(timings["pp_tok_s"], 3),
        "draft_n": timings["draft_n"],
        "draft_n_accepted": timings["draft_n_accepted"],
        "context_size": int(context_size or 0),
    }


def fail_payload(model: str, errors: list[str]) -> dict:
    return {
        "framework": FRAMEWORK,
        "model_id": model,
        "workload_kind": WORKLOAD_KIND,
        "throughput_unit": THROUGHPUT_UNIT,
        "success": False,
        "quality_gate": {"passed": False},
        "errors": errors,
    }


def _persist(result_file: str, payload: dict) -> None:
    with open(result_file, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def run_completion(
    port: int,
    model: str,
    result_file: str,
    *,
    fixture: dict | None = None,
) -> int:
    """Drive one completion, persist ``inferencex_result.json``, gate.

    P1.10: ``fixture`` is an exact-token workload spec (prompt string +
    n_predict + seed + temperature). The bench script sends this exact
    payload to llama-server so re-runs on the same GGUF produce the
    same token sequence after tokenization. The fixture itself is
    hashed (see ``fixture_hash``) and recorded in the result payload so
    audit replays can verify which workload was used.

    Pinned to a 4K-character prompt and 128 n_predict by default. The
    prompt string is fixed; combined with the GGUF tokenizer it produces
    a deterministic token sequence per GGUF. Bench scripts may override
    by passing ``fixture=...``.
    """
    if fixture is None:
        fixture = DEFAULT_FIXTURE
    fixture_hash = hashlib.sha256(
        json.dumps(fixture, sort_keys=True).encode("utf-8")
    ).hexdigest()
    prompt = fixture["prompt"]
    n_predict = int(fixture.get("n_predict", 128))
    seed = int(fixture.get("seed", 0))
    temperature = float(fixture.get("temperature", 0.0))
    req = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "seed": seed,
        "cache_prompt": False,
    }
    start = time.monotonic()
    try:
        with urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:{port}/completion",
                data=json.dumps(req).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=600,
        ) as resp:
            body = json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001 - fail-closed gate
        _persist(result_file, fail_payload(model, [f"completion request failed: {exc!r}"]))
        return 1
    wall = time.monotonic() - start
    timings = parse_timings(body.get("timings", {}))
    quality_ok = bool(body.get("content")) and timings["eval_n"] > 0
    payload = build_result(model, wall, timings, quality_ok, body.get("context_size") or 0)
    # Audit fields: prove which exact-token workload produced these numbers.
    payload["fixture_hash"] = fixture_hash
    payload["fixture_n_predict"] = n_predict
    payload["fixture_seed"] = seed
    payload["fixture_temperature"] = temperature
    payload["fixture_prompt_chars"] = len(prompt)
    _persist(result_file, payload)
    if not quality_ok:
        print("BENCH_FAIL: empty completion or zero eval tokens", file=sys.stderr)
        return 1
    print(
        f"pp={timings['pp_tok_s']:.1f} tok/s tg={timings['tg_tok_s']:.1f} tok/s "
        f"prompt_eval={timings['prompt_ms']:.0f} ms prompt_n={timings['prompt_n']} "
        f"eval_n={timings['eval_n']} wall={wall:.1f}s "
        f"fixture={fixture_hash[:12]}"
    )
    return 0


#: Default exact-token fixture. The prompt is a fixed string; combined
#: with the GGUF tokenizer and the seed it produces a deterministic
#: token sequence for a given GGUF. Bench scripts may override by
#: passing ``fixture=...`` to ``run_completion``.
DEFAULT_FIXTURE: dict = {
    "prompt": (
        "The quick brown fox jumps over the lazy dog while the wise owl watches from "
        "the oak tree. "
    ) * 200,
    "n_predict": 128,
    "seed": 0,
    "temperature": 0.0,
}


def _usage(prog: str) -> None:
    print(
        f"usage: {prog} {{port-in-use <port> | split-args-file <extra> <out> | "
        f"bench <port> <model> <result_file> | fail-payload <model> <result_file> <msg>}}",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        _usage(sys.argv[0])
        return 2
    cmd, *rest = args
    if cmd == "port-in-use":
        port = int(rest[0])
        return 1 if port_in_use(port) else 0
    if cmd == "split-args-file":
        extra = rest[0] if rest else ""
        out = rest[1] if len(rest) > 1 else ""
        try:
            tokens = split_server_args(extra)
        except ValueError as exc:
            print(f"BENCH_FAIL: EXTRA_CUSTOM_ARGS not parseable: {exc}", file=sys.stderr)
            return 1
        with open(out, "wb") as fh:
            fh.write("\0".join(tokens).encode("utf-8"))
        return 0
    if cmd == "bench":
        port, model, result_file = rest[0], rest[1], rest[2]
        return run_completion(int(port), model, result_file)
    if cmd == "fail-payload":
        model, result_file, msg = rest[0], rest[1], rest[2]
        _persist(result_file, fail_payload(model, [msg]))
        return 1
    _usage(sys.argv[0])
    return 2


if __name__ == "__main__":
    sys.exit(main())
