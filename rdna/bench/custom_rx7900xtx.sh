#!/usr/bin/env bash
# Hyperloom custom-framework benchmark entrypoint for the RX 7900 XTX
# (gfx1100, 96 CU) on the RDNA3 consumer lane.
#
# Contract (Hyperloom bypass scriptable path):
#   - invoked as: bash custom_rx7900xtx.sh  (cwd = per-run workspace)
#   - reads env: MODEL, RESULT_DIR, RESULT_FILENAME, RUNNER_TYPE,
#                EXTRA_CUSTOM_ARGS (operator-pinned server flags,
#                shlex-parsed by measure.py)
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
MEASURE_PY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/measure.py"
SERVER_LOG="$RESULT_DIR/server.log"
EXTRA_ARGS_FILE="$RESULT_DIR/extra_args.nul"
# H0.6.2: exact-token fixture path. Empty = DEFAULT_FIXTURE (back-compat),
# but the orchestrator MUST pass an explicit path for real measurements.
FIXTURE_PATH="${FIXTURE_PATH:-}"

mkdir -p "$RESULT_DIR"

fail() { # $1 = error message
  python3 "$MEASURE_PY" fail-payload "$MODEL" "$RESULT_DIR/$RESULT_FILENAME.json" "$1" || true
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

# Fail closed if the experiment port is already owned. Production (18079)
# is never touched here; a foreign process on 18179 means another agent or
# a leftover run, and a second server would make the numbers meaningless.
if ! python3 "$MEASURE_PY" port-in-use "$PORT" >/dev/null 2>&1; then
  fail "port $PORT is already in use; refusing to launch (foreign process)"
fi

# Production-identical server flags; EXTRA_CUSTOM_ARGS appends operator knobs
# via shlex (quotes and escapes honored; garbage rejected -- measure.py
# exits 1 on unparseable input, which fail()s the run).
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
python3 "$MEASURE_PY" split-args-file "${EXTRA_CUSTOM_ARGS:-}" "$EXTRA_ARGS_FILE" \
  || fail "EXTRA_CUSTOM_ARGS not parseable as shell words"
mapfile -d '' -t EXTRA < "$EXTRA_ARGS_FILE"
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

python3 "$MEASURE_PY" bench "$PORT" "$MODEL" "$RESULT_DIR/$RESULT_FILENAME.json" "$FIXTURE_PATH" \
  || fail "bench subcommand failed"
exit 0
