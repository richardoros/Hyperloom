# Hyperloom-RDNA Locked Plan (2026-08-17)

Goal: autonomously optimize Qwen3.8-27B (Q5_K_XL, turbo4 KV) under buun-llama-cpp on
RX 7900 XTX (gfx1100, OCuLink eGPU) + Radeon 890M (gfx1150 iGPU).

## Verified upstream facts (2026-08-17)

- AMD-AGI/Hyperloom: MIT, v1.0.0b1 (pyproject), 3,042 commits, active. [confirmed]
- `framework=custom` EXISTS: repo ships examples/hyperloom-custom-advanced/ installed
  as a skill for Claude Code / Cursor / Codex. `--framework` is a CLI field. [confirmed]
- TraceLens is HARDWARE-AGNOSTIC (compat.rst component matrix). GEAK / IntelliKit /
  Magpie / AgentKernelArena are Instinct-bound (MI300X/325X/355X, ROCm 7.2, Ubuntu 22.04/24.04).
- Hyperloom invokes external coding agents (Claude Code / Codex / Cursor) under their
  own licenses - budget token cost.
- OS risk: validated matrix is Ubuntu + ROCm 7.2; host is Fedora. Install friction
  expected; fallback = docker mode with host llama.cpp mounted read-only.

## Local facts (verified 2026-08-17)

- PRODUCTION = qwen38-turboquant.service (enabled), port 18079, not 18080.
  Model: /home/homelabserver/models/Qwen3.8-27B-UD-Q5_K_XL.gguf, ROCm, turbo4 KV.
  Experiment lane: 18179. Older qwen36 service (18080) is disabled.
- Existing assets: /home/homelabserver/llama-turboquant/ (bench suite: suites/, tools/,
  DeepSeekAndDestroy/), /home/homelabserver/llama.cpp/ (build-tq3*, build-rocm,
  build-vulkan), /home/homelabserver/src/llama.cpp-turboquant/, inference-bench/,
  benchmarking/, models/. Workhorse-bench may half-exist in llama-turboquant/.
- GPUs: c8:00.0 Navi 31 (7900 XTX), c9:00.0 Strix (890M).

## Locked decisions

- Fork AMD-AGI/Hyperloom at PINNED SHA at fork time, never rebase during campaign.
- Repos: /home/homelabserver/hyperloom-rdna/ (branch feature/rdna-llamacpp),
  buun-llama-cpp (autoresearch/<experiment-id>), workhorse-bench/ (read-only judge).
- Executor: this session (opencode) executes H0.0.
- apply/ checkout must stay untouched by this effort.

## Milestones

### H0.0 - Fork + seam smoke test (NO optimization, NO evaluator)
1. Fork at pinned SHA, branch feature/rdna-llamacpp. First commit = this PLAN.md.
2. Inventory pass: llama-turboquant/, llama.cpp/, src/llama.cpp-turboquant/, models/.
3. Production map: confirm 18079 service, GGUF SHA-256, exact llama-server flags.
4. Wire framework=custom (per examples/hyperloom-custom-advanced/) to existing
   llama.cpp bench script. Add gfx1100 identity.
5. Lifecycle proof on 18179: launch -> measure -> terminate -> persist result.
EXIT GATE: Hyperloom launches workload, measures, terminates, persists a result.

### H0.5 - The trustworthy laboratory (MOST IMPORTANT)
- Evaluator owns: source checkout, build-from-SHA, GGUF hash verify, launch,
  measurement, statistics, DB writes, keep/reject. Agent owns only: hypothesis,
  sources, proposed patch/config. Harness is the ONLY DB writer.
- Identity block: both git SHAs, GGUF SHA-256, compiler, cmake flags, HIP/ROCm/
  Mesa/Vulkan/kernel versions. Evaluator builds candidate itself, verifies binary.
- Variance characterization BEFORE any search: 5x unchanged baseline under load
  gate; per-metric sigma; bootstrap rule max(2sigma, +3%), revised empirically.
- Load gate per repetition: no foreign GPU process, VRAM < threshold, idle util,
  clocks/temp/power in range, CPU/RAM headroom, no swap, no competing build/celery.
  Failure = CONTENDED, never LOSS.
- A/B/A ordering; void block if A_before vs A_after disagree materially.
- Capture production coding baseline (20 jobs: success, first-pass, pytest, lint,
  retries, tokens, wall-clock) before any search.
- Tiered gates: fast perf (pp4K + tg4K x N) -> perf matrix (pp 4/32/128K,
  tg @4/64/128K) -> product gate (20 coding jobs). Quality/test/OOM/crash
  regression = reject. Product metric (correct work / wall-clock hr) outranks tok/s.
- Three acts: RETAIN (autonomous, descendants allowed) / APPROVE (human) / PROMOTE
  (deterministic deploy). Touching qwen38-turboquant.service requires human merge
  -> deploy -> smoke -> rollback.

### H0.6 - Exclusive-XTX lifecycle
drain 18079 -> clean GPU gate -> candidate on 18179 -> kill -> restore -> verify 18079.

### H0.7+ 
Config search (ctx/batch/ubatch, MTP, KV, checkpoints, ROCm flags) -> H1 Vulkan
backend -> H2 Q6 + gfx1100/gfx1150 placement -> H3 profiling + kernel autoresearch
(TraceLens kept, hardware-agnostic; GEAK/Magpie stay disabled until H3) -> H4 novel
kernels.

## Experiment DB schema (locked)

experiments: id, timestamp, hyperloom_sha, llama_cpp_sha, gguf_sha256, model, quant,
backend, rocm_version, mesa_version, kernel_version, cmake_flags, runtime_flags,
gpu0_identity, gpu1_identity, gpu_topology, gpu_temp_start/end, gpu_clock,
vram_start/peak, gpu_utilization, gpu_power, cpu_load, ram_available, swap,
system_load, contention_gate_pass, context, batch, ubatch, mtp_depth, kv_k, kv_v,
gpu_split, hypothesis, sources, changed_files, parent_experiment, candidate_branch,
repeat_count, run_order, pp4k, pp32k, pp128k, tg4k, tg64k, tg128k, vram_peak,
mtp_acceptance, quality_score, wallclock_score, metrics, variance, confidence,
decision, decision_reason, promoted_sha, notes.

## Research discipline

Before any new optimization: search upstream work -> inspect PRs/issues -> compare
fork -> hypothesis -> test locally. Record every experiment incl. failures + sources.
