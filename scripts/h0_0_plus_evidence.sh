#!/usr/bin/env bash
# H0.0+ closure-pass evidence bundle. Run from the fork clone
# (/home/homelabserver/hyperloom-rdna). Records the full CLI smoke in a
# timestamped bundle dir for the 10/10 acceptance evidence.
#
# Hard external deadlines (the smoke is a 10/10 polish, not an optimization
# run):
#   - outer wrapper: 12m
#   - venv install:  5m
#   - CLI optimize:  8m
#   - tests:         3m
# Internal Hyperloom budget: --max-hours 0.10 (6 min) + --closing-grace-sec 30.
# On timeout: cleanup child processes, preserve logs, classify as TIMEOUT.
set -euo pipefail

REPO=$(pwd)
VENV=/tmp/h0-cli-venv
GGUF=/home/homelabserver/models/Qwen3.8-27B-UD-Q5_K_XL.gguf
FRAMEWORK_PATH=/home/homelabserver/src/llama.cpp-turboquant
BUNDLE="$REPO/h0_0_plus_evidence_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$BUNDLE"

SMOKE_PORT=18180
RESULT="TIMEOUT"

cleanup() {
  local rc=$?
  echo "cleanup: kill any llama-server bound to 127.0.0.1:$SMOKE_PORT"
  pkill -KILL -f "llama-server.*--port $SMOKE_PORT" 2>/dev/null || true
  exit "$rc"
}
trap cleanup EXIT INT TERM

# Stage 1: venv (5m). Reused across runs.
echo "=== stage 1: venv (5m) ==="
if ! [ -x "$VENV/bin/python" ] \
  || ! "$VENV/bin/python" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)" 2>/dev/null; then
  rm -rf "$VENV"
  uv venv --python 3.11 "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export VENV BUNDLE REPO
timeout --signal=TERM --kill-after=20s 5m bash -c '
  set -e
  uv pip install --python "$VENV/bin/python" -e ".[runtime]" packaging pytest "ray[default]==2.44.1" \
    > "$BUNDLE/venv_install.log" 2>&1
  # Force pin click<8.3.0 AFTER the editable install, which can pull a newer
  # click via transitive deps. The CLI preflight rejects click >= 8.3.0 and
  # would otherwise try to auto-install via python -m pip install - which
  # fails because the venv has no pip module.
  uv pip install --python "$VENV/bin/python" --quiet "click<8.3.0" \
    >> "$BUNDLE/venv_install.log" 2>&1
  uv pip install --python "$VENV/bin/python" --quiet \
    aiohttp tqdm numpy requests transformers huggingface_hub datasets pandas \
    >> "$BUNDLE/venv_install.log" 2>&1
  uv pip install --python "$VENV/bin/python" --quiet "click<8.3.0" \
    >> "$BUNDLE/venv_install.log" 2>&1
' || { echo "RESULT=STAGE1_VENV"; exit 1; }

"$VENV/bin/python" -c 'import sys; print(sys.executable)' | tee "$BUNDLE/python_executable.txt"
"$VENV/bin/python" --version | tee "$BUNDLE/python_version.txt"
uv pip freeze --python "$VENV/bin/python" | tee "$BUNDLE/pip_freeze.txt"

# install.sh would call the provider-credential preflight; the smoke is
# LLM-free (mock backends) and would not consume the credentials anyway.
ls -la src/hyperloom/inference_optimizer/assets/install.sh | tee "$BUNDLE/install_sh_present.txt"

# Stage 2: production/GPU fail-closed gate.
echo "=== stage 2: fail-closed gate ==="
systemctl is-active qwen38-turboquant.service | tee "$BUNDLE/qwen38_service_state_before.txt" || true
ss -ltnp | tee "$BUNDLE/listeners_before.txt"

if systemctl is-active --quiet qwen38-turboquant.service; then
  echo 'BLOCKED: production service active; do not stop it in H0.0+' | tee "$BUNDLE/blocked.txt"
  echo "BLOCKED=18079_SERVICE_ACTIVE" > "$BUNDLE/result.txt"
  exit 42
fi
if ss -ltnp | grep -E ':18079\b'; then
  echo 'BLOCKED: 18079 already listening; do not mutate production in H0.0+' | tee "$BUNDLE/blocked.txt"
  echo "BLOCKED=18079_LISTENING" > "$BUNDLE/result.txt"
  exit 42
fi

rocm-smi --showproductname --showuniqueid --showbus --showmeminfo vram --json \
  | tee "$BUNDLE/rocm_smi_before.json" || true
pgrep -af 'llama-server|llama.cpp|turboquant' \
  | tee "$BUNDLE/llama_processes_before.txt" || true

# Stage 3: prepare CLI env.
echo "=== stage 3: prepare CLI env ==="
SMOKE_USER_DATA="$BUNDLE/user_data"
mkdir -p "$SMOKE_USER_DATA/runtime"
cat > "$SMOKE_USER_DATA/runtime/kernel-agent.env.sh" <<'EOF'
# H0.0+ stub kernel-agent env. The smoke uses --no-kernel --no-explore
# --no-framework-agent --no-eval --critic-mock --robustness-mock, so the
# runtime never reads HYPERLOOM_KERNEL_AGENT_ROOT or TRACELENS_ROOT. This
# file exists only to satisfy the preflight fallback loader.
export HYPERLOOM_KERNEL_AGENT_ROOT="$USER_DATA_PATH/__kernel_agent_stub__"
export TRACELENS_ROOT="$USER_DATA_PATH/__tracelens_stub__"
EOF

export USER_DATA_PATH="$SMOKE_USER_DATA"
export KERNEL_AGENT_ENV="$SMOKE_USER_DATA/runtime/kernel-agent.env.sh"
export PORT="$SMOKE_PORT"
# Stub credentials (mock backends never call these).
export OPENAI_BASE_URL="http://localhost:0/v1"
export OPENAI_API_KEY="smoke-stub-key-not-used"
# The CLI's preflight requires HYPERLOOM_BENCHMARK_BACKEND=bypass for
# --framework=custom; otherwise it falls back to Magpie and chokes.
export HYPERLOOM_BENCHMARK_BACKEND=bypass

LAUNCH_INFO="$BUNDLE/launch-info.json"

# Stage 4: CLI optimize (8m hard deadline).
# PRELUDE plain-profile may execute. TraceLens roofline analysis is disabled
# by --no-enable-roofline.
echo "=== stage 4: CLI optimize (8m hard deadline) ==="
START_SEC=$(date +%s)
set +e
timeout --signal=TERM --kill-after=30s 8m \
  "$VENV/bin/python" -m hyperloom.inference_optimizer.cli -v optimize \
    --model "$GGUF" \
    --framework custom \
    --gpu-type rx7900xtx \
    --framework-path "$FRAMEWORK_PATH" \
    --benchmark-scripts-dir "$REPO/rdna/bench" \
    --tp 1 \
    --ep 1 \
    --conc 1 \
    --isl 1024 \
    --osl 1024 \
    --max-hours 0.10 \
    --closing-grace-sec 30 \
    --no-kernel \
    --no-explore \
    --no-framework-agent \
    --no-enable-conc-sweep \
    --no-enable-roofline \
    --no-eval \
    --no-research-scout \
    --no-static-recon \
    --research-lane-capacity 0 \
    --gpu-specialist-capacity 0 \
    --critic-mock \
    --robustness-mock \
    --enablement off \
    --degraded-pr \
    --launch-info-file "$LAUNCH_INFO" 2>&1 | tee "$BUNDLE/cli.log"
CLI_RC=${PIPESTATUS[0]}
set -e
END_SEC=$(date +%s)
WALL=$((END_SEC - START_SEC))
echo "cli wall-time: ${WALL}s rc=$CLI_RC"

echo "$CLI_RC" | tee "$BUNDLE/cli_exit_code.txt"
echo "$WALL" | tee "$BUNDLE/cli_wall_seconds.txt"

case "$CLI_RC" in
  124) RESULT="TIMEOUT_TIMEOUT"; REASON="outer 8m timeout hit" ;;
  137) RESULT="TIMEOUT_KILL"; REASON="SIGKILL after SIGTERM grace window" ;;
  0) RESULT="SUCCESS"; REASON="cli exit 0" ;;
  *) RESULT="FAILURE"; REASON="cli exit $CLI_RC" ;;
esac
echo "$RESULT" | tee "$BUNDLE/result.txt"
echo "reason: $REASON" | tee -a "$BUNDLE/result.txt"

# Stage 5: post-run capture.
echo "=== stage 5: post-run capture ==="
ss -ltnp | tee "$BUNDLE/listeners_after.txt"
systemctl is-active qwen38-turboquant.service | tee "$BUNDLE/qwen38_service_state_after.txt" || true
rocm-smi --showproductname --showuniqueid --showbus --showmeminfo vram --json \
  | tee "$BUNDLE/rocm_smi_after.json" || true
git status --short | tee "$BUNDLE/git_status_short.txt"

# Stage 6: tests (3m hard deadline).
echo "=== stage 6: tests (3m hard deadline) ==="
timeout --signal=TERM --kill-after=20s 3m \
  "$VENV/bin/python" -m pytest -q rdna/tests/test_rdna_seam.py \
    src/hyperloom/inference_optimizer/tests/test_gpu_board_list_single_source.py \
    src/hyperloom/inference_optimizer/tests/test_custom_framework.py \
    src/hyperloom/inference_optimizer/tests/test_gpu_pool_device_resolution.py \
    2>&1 | tee "$BUNDLE/pytest.log" || true

case "$RESULT" in
  SUCCESS)   bundle_rc=0 ;;
  TIMEOUT_*) bundle_rc=124 ;;
  FAILURE)   bundle_rc=1 ;;
  BLOCKED*)  bundle_rc=42 ;;
  *)         bundle_rc=1 ;;
esac
echo "bundle_rc=$bundle_rc result=$RESULT"
exit "$bundle_rc"
