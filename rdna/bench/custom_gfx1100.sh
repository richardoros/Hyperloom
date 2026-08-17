#!/usr/bin/env bash
# Hyperloom custom-framework benchmark entrypoint for RDNA3 gfx1100 (RX 7900 XTX).
#
# Contract (Hyperloom bypass scriptable path):
#   - invoked as: bash custom_gfx1100.sh  (cwd = per-run workspace)
#   - reads env: MODEL, RESULT_DIR, RESULT_FILENAME, RUNNER_TYPE,
#                EXTRA_CUSTOM_ARGS (operator-pinned server flags)
#   - writes: $RESULT_DIR/$RESULT_FILENAME.json  (flat InferenceX shape)
#   - MUST emit quality_gate.passed: a scriptable workload with a missing or
#     unparseable gate scores 0.0 and every candidate is rejected.
#
# Server defaults mirror the production unit qwen38-turboquant.service
# (port 18079) except the port, which stays on the experiment lane 18179.
# Never run this on the production port.

set -u

MODEL="${MODEL:?MODEL (GGUF path) required}"
RESULT_DIR="${RESULT_DIR:-.}"
RESULT_FILENAME="${RESULT_FILENAME:-inferencex_result}"
PORT="${PORT:-18179}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-/home/homelabserver/src/llama.cpp-turboquant/build/bin/llama-server}"
SERVER_LOG="$RESULT_DIR/server.log"

mkdir -p "$RESULT_DIR"

fail() { # $1 = error message
  python3 - "$RESULT_DIR/$RESULT_FILENAME.json" "$1" <<'PY'
import json, sys
payload = {
    "framework": "custom",
    "model_id": __import__("os").environ.get("MODEL", ""),
    "workload_kind": "scriptable",
    "throughput_unit": "tok/s",
    "success": False,
    "quality_gate": {"passed": False},
    "errors": [sys.argv[2]],
}
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
PY
  echo "BENCH_FAIL: $1" >&2
  exit 1
}

# Production-identical environment (qwen38-turboquant.service).
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export GPU_DEVICE_ORDINAL="${GPU_DEVICE_ORDINAL:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"
export GPU_MAX_HEAP_SIZE="${GPU_MAX_HEAP_SIZE:-100}"
export GPU_MAX_ALLOC_PERCENT="${GPU_MAX_ALLOC_PERCENT:-100}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-1}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm/lib}"
export TURBO_TCQ_CB="${TURBO_TCQ_CB:-/home/homelabserver/src/llama.cpp-turboquant/codebooks/3bit/cb_50iter_finetuned.bin}"
export TURBO_TCQ_CB2="${TURBO_TCQ_CB2:-/home/homelabserver/src/llama.cpp-turboquant/codebooks/2bit/tcq_2bit_100iter_s99.bin}"

# Production-identical server flags; EXTRA_CUSTOM_ARGS appends operator knobs.
SERVER_ARGS=(
  -m "$MODEL"
  -ngl 99
  -c 131072
  -np 1
  -kvu
  -b 512
  -ub 256
  -fa on
  -ctk turbo4
  -ctv turbo4
  --cache-ram 0
  -ctxcp 32
  -cms 256
  --fit off
  --no-logits-all
  --port "$PORT"
)
# shellcheck disable=SC2206
read -r -a EXTRA <<< "${EXTRA_CUSTOM_ARGS:-}"
SERVER_ARGS+=("${EXTRA[@]}")

[ -x "$LLAMA_SERVER_BIN" ] || fail "llama-server binary missing: $LLAMA_SERVER_BIN"
[ -f "$MODEL" ] || fail "model file missing: $MODEL"

"$LLAMA_SERVER_BIN" "${SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null' EXIT

# Wait for health (model load on 24 GB takes up to ~2 min).
for _ in $(seq 1 120); do
  if curl -sf -m 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    fail "llama-server died during startup (see $SERVER_LOG)"
  fi
  sleep 2
done
curl -sf -m 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 \
  || fail "server did not become healthy on port $PORT within 240s (see $SERVER_LOG)"

RESULT_FILE="$RESULT_DIR/$RESULT_FILENAME.json"
python3 - "$RESULT_FILE" "$PORT" "$MODEL" <<'PY'
import json
import os
import sys
import time
import urllib.request

result_file, port, model = sys.argv[1], int(sys.argv[2]), sys.argv[3]

# ~4K-token prompt: repeat a filler sentence until the char budget implies
# >= 4096 tokens (English ~4 chars/token; the server count is authoritative).
unit = "The quick brown fox jumps over the lazy dog while the wise owl watches from the oak tree. "
prompt = (unit * 200)[:16000]

req = {
    "prompt": prompt,
    "n_predict": 128,
    "temperature": 0.0,
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
    payload = {
        "framework": "custom",
        "model_id": model,
        "workload_kind": "scriptable",
        "throughput_unit": "tok/s",
        "success": False,
        "quality_gate": {"passed": False},
        "errors": [f"completion request failed: {exc!r}"],
    }
    with open(result_file, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    sys.exit(1)

wall = time.monotonic() - start
t = body.get("timings", {})
# buun-llama-cpp fork key names; upstream llama.cpp equivalents as fallback.
prompt_n = int(t.get("prompt_n") or t.get("prompt_eval_count") or 0)
prompt_ms = (t.get("prompt_ms") or (t.get("prompt_eval_duration") or 0) / 1e6) or 0.0
eval_n = int(t.get("predicted_n") or t.get("eval_count") or 0)
eval_ms = (t.get("predicted_ms") or (t.get("eval_duration") or 0) / 1e6) or 0.0
pp_tok_s = float(t.get("prompt_per_second") or (prompt_n / prompt_ms * 1000.0 if prompt_ms > 0 else 0.0))
tg_tok_s = float(t.get("predicted_per_second") or (eval_n / eval_ms * 1000.0 if eval_ms > 0 else 0.0))
ttft_ms = float(prompt_ms or 0.0)
quality_ok = bool(body.get("content")) and eval_n > 0

payload = {
    "framework": "custom",
    "model_id": model,
    "workload_kind": "scriptable",
    "throughput_unit": "tok/s",
    "success": True,
    "quality_gate": {"passed": quality_ok},
    "output_throughput": round(tg_tok_s, 3),
    "request_throughput": round(1.0 / wall, 3),
    "total_token_throughput": round((prompt_n + eval_n) / wall, 3),
    "completed": 1,
    "total_input_tokens": prompt_n,
    "total_output_tokens": eval_n,
    "duration": round(wall, 3),
    "mean_ttft_ms": round(ttft_ms, 3),
    "mean_tpot_ms": round(eval_ms / eval_n, 3) if eval_n else None,
    "mean_e2el_ms": round(eval_ms, 3),
    "prompt_eval_tok_s": round(pp_tok_s, 3),
    "draft_n": int(t.get("draft_n") or 0),
    "draft_n_accepted": int(t.get("draft_n_accepted") or 0),
    "context_size": int(body.get("context_size") or 0),
}
with open(result_file, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)

if not quality_ok:
    print("BENCH_FAIL: empty completion or zero eval tokens", file=sys.stderr)
    sys.exit(1)
print(f"pp={pp_tok_s:.1f} tok/s tg={tg_tok_s:.1f} tok/s ttft={ttft_ms:.0f} ms "
      f"prompt_n={prompt_n} eval_n={eval_n} wall={wall:.1f}s")
PY

exit $?