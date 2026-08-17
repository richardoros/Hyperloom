# rdna/ - RDNA3 consumer-GPU seam for Hyperloom

Fork goal (H0.0, locked): prove the Hyperloom `custom` framework path can launch
the production llama.cpp workload on the RX 7900 XTX (gfx1100), measure it,
terminate it, and persist a result, without touching the production service
(`qwen38-turboquant.service`, port 18079).

## Seam contract (verified against upstream 8b2a912c0)

- Framework: `custom` (registered in `framework_registry.py`, kind=scriptable).
- `--framework-path` sets `FRAMEWORK_REPO_PATH` = the checkout on PolicyGate's
  patch allowlist (specialists may only edit source there).
- `--benchmark-scripts-dir` sets `HYPERLOOM_BYPASS_SCRIPTS_DIR`; the entrypoint
  is taken as `custom_<gpu-type>.sh`, or the single `.sh` in the directory.
- The entrypoint is invoked as `bash custom_gfx1100.sh` with env: `MODEL`,
  `RESULT_DIR`, `RESULT_FILENAME` (=`inferencex_result`), `RUNNER_TYPE`,
  plus anything `--extra-env` pins (pinned keys become part of the measurement
  contract; a variant may add keys but not overwrite pinned ones).
- It must write `$RESULT_DIR/inferencex_result.json` (flat InferenceX shape).
  `quality_gate.passed` is fail-closed: missing/unparseable gate = score 0.0
  and every candidate is rejected. `throughput_unit`/`workload_kind` are
  relayed verbatim; `output_throughput` and `mean_e2el_ms` feed the display.
- The bypass executor (`orchestrator/actions/executors/bypass_scriptable.py`)
  normalizes the raw result into a Magpie-compatible `benchmark_report.json`
  via `bypass_report.py`, so existing collectors consume it unchanged.

## Files

- `bench/custom_gfx1100.sh` - benchmark entrypoint. Server flags default to the
  production unit `qwen38-turboquant.service` (port 18079) but always bind the
  experiment lane 18179. Overrides: `MODEL`, `PORT`, `LLAMA_SERVER_BIN`,
  `EXTRA_CUSTOM_ARGS` (operator-pinned extra server flags).

## gfx1100 / gfx1150 identity

Added to the dispatch tables so `--gpu-type gfx1100` is accepted and the
runner label resolves for script lookup:

- `src/hyperloom/common/gpu_identity.py`: `gfx1100` (96 CU), `gfx1150` (16 CU).
- `src/hyperloom/inference_optimizer/gpu_types.py`: `_GFX_TO_RUNNER` entries.

## Run

```bash
cd /home/homelabserver/hyperloom-rdna
export MODEL=/home/homelabserver/models/Qwen3.8-27B-UD-Q5_K_XL.gguf
mkdir -p /tmp/h0-run && cd /tmp/h0-run
bash /home/homelabserver/hyperloom-rdna/rdna/bench/custom_gfx1100.sh
cat inferencex_result.json
```

Produces: server.log (llama-server stdout), inferencex_result.json (flat
report with quality_gate), and prints `pp=... tg=...` on success.

## Production map (recorded 2026-08-17, H0.0)

- Service: `qwen38-turboquant.service` (enabled; currently INACTIVE), port 18079.
- Model: `/home/homelabserver/models/Qwen3.8-27B-UD-Q5_K_XL.gguf`
  SHA-256 `176a6a3f034e9cdc447c10cd00329fc9b31002e6589b9295f2ad4f1eefe0f6ab`.
- Binary: `/home/homelabserver/src/llama.cpp-turboquant/build/bin/llama-server`
  (checkout = richardoros/buun-llama-cpp fork, branch
  `perf/rocm-fused-gdn-exp2-swa-deferred-recurrent`).
- Flags: `-ngl 99 -c 131072 -np 1 -kvu -b 512 -ub 256 -fa on -ctk turbo4
  -ctv turbo4 --cache-ram 0 -ctxcp 32 -cms 256 --fit off --no-logits-all`
- Env: `ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 HSA_ENABLE_SDMA=0
  GPU_MAX_HEAP_SIZE=100 GPU_MAX_ALLOC_PERCENT=100 GPU_MAX_HW_QUEUES=1
  LD_LIBRARY_PATH=/opt/rocm/lib TURBO_TCQ_CB=<3bit cb>
  TURBO_TCQ_CB2=<2bit cb>`.
- Prior measured result (Qwen3.6, 131K, turbo4): pp 285.9 tok/s, tg 10.33
  tok/s, 22.94 GiB peak VRAM (inference-bench/turboquant-result.md).