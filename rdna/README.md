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
- The entrypoint is invoked as `bash custom_rx7900xtx.sh` with env: `MODEL`,
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

- `bench/custom_rx7900xtx.sh` - benchmark entrypoint. Server flags default to
  the production unit `qwen38-turboquant.service` (port 18079) but always bind
  the experiment lane 18179. Overrides: `MODEL`, `PORT`, `LLAMA_SERVER_BIN`,
  `EXTRA_CUSTOM_ARGS` (operator-pinned extra server flags, shlex-parsed).
- `bench/measure.py` - pure-python helpers used by the entrypoint:
  `split_server_args` (shlex with quoted/escaped tokens and fail-closed on
  unparseable input), `port_in_use` (TCP probe, fail-closed on foreign
  listener before launch), `parse_timings` (buun-llama-cpp key names with
  upstream llama.cpp fallback), `build_result` (success=quality_ok,
  `mean_e2el_ms` = wall*1000, `prompt_eval_ms` separate, never labeled TTFT).
- `tools/h0_smoke_driver.py` - smoke driver that calls the public
  `_run_scriptable_benchmark` bypass executor. Run end-to-end through the
  Hyperloom code path with one completion.

## Board identity (rx7900xtx / radeon890m)

The board name is the gpu_type key; the gfx arch + CU count is the dispatch
tuple. Keeping the arch out of the key means the CLI `--gpu-type` names a
real product (the operator buys an RX 7900 XTX, not a gfx1100).

- `src/hyperloom/common/gpu_identity.py`:
  - `rx7900xtx -> (gfx1100, 96 CU)`
  - `radeon890m -> (gfx1150, 16 CU)`
- `src/hyperloom/inference_optimizer/gpu_types.py`:
  - `_GFX_TO_RUNNER`: `gfx1100 -> rx7900xtx`, `gfx1150 -> radeon890m`.
  - `_PRODUCT_ALIASES`: rocm-smi product names that differ from the
    uppercased gpu_type key (`RX 7900 XTX`, `RADEON 890M`); autodetect
    uses the joined `_TAG_TO_GPU_TYPE` map.

## Run

```bash
cd /home/homelabserver/hyperloom-rdna
export MODEL=/home/homelabserver/models/Qwen3.8-27B-UD-Q5_K_XL.gguf
mkdir -p /tmp/h0-run && cd /tmp/h0-run
bash /home/homelabserver/hyperloom-rdna/rdna/bench/custom_rx7900xtx.sh
cat inferencex_result.json
```

Produces: server.log (llama-server stdout), inferencex_result.json (flat
report with quality_gate, `prompt_eval_ms` separate, `mean_e2el_ms` =
wall*1000), `extra_args.nul` (evidence of shlex parsing), and prints
`pp=... tg=... prompt_eval=... ms` on success.

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
