#!/usr/bin/env python3
"""H0.0 smoke driver: run the custom-framework seam through Hyperloom's own
bypass executor (the exact code path the orchestrator uses for a scriptable
custom baseline), then print the normalized benchmark_report.json.

Usage:
    MODEL=/path/to/model.gguf python3 rdna/tools/h0_smoke_driver.py [OUTPUT_DIR]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hyperloom.orchestrator.actions.executors.bypass_runner import _run_scriptable_benchmark  # noqa: E402


def main() -> int:
    output_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path("/tmp/h0-run")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = os.environ["MODEL"]
    # The CLI publishes --framework-path / --benchmark-scripts-dir as env
    # vars; mirror that here so the executor's resolve_scriptable_script
    # finds rdna/bench/custom_<gpu-type>.sh. The driver-only env override
    # keeps this script callable from a plain shell (no install.sh needed).
    rdna_bench = REPO_ROOT / "rdna" / "bench"
    os.environ.setdefault("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(rdna_bench))
    os.environ.setdefault("HYPERLOOM_BENCHMARK_BACKEND", "bypass")
    os.environ.setdefault("FRAMEWORK_REPO_PATH", "/home/homelabserver/src/llama.cpp-turboquant")
    bench = {
        "framework": "custom",
        "model": model,
        "runner_type": "rx7900xtx",
        "precision": "q5_k_xl",
        "profiler": {
            "torch_profiler": {"enabled": False},
            "system_profiler": {"enabled": False},
            "tracelens": {"enabled": False},
        },
        "envs": {},
        "timeout_seconds": 900,
    }
    # PORT override lets the smoke run when 18179 is already in use by a
    # unrelated development workload -- amendment 4 of the H0.0+ closure pass
    # says we never touch a foreign process, but the script reads PORT
    # from its env so an operator override is a legitimate experiment lane.
    # The executor's build_scriptable_env reads `bench["envs"]`, not the
    # bench_envs argument, so the override must land in the bench config.
    if "PORT" in os.environ:
        bench["envs"]["PORT"] = os.environ["PORT"]
    rc = _run_scriptable_benchmark(
        framework="custom",
        model=model,
        bench=bench,
        bench_envs={},
        timeout_s=900,
        output_dir=output_dir,
    )
    reports = sorted(output_dir.rglob("benchmark_report.json"))
    if not reports:
        print(f"FAIL: no benchmark_report.json under {output_dir}", file=sys.stderr)
        return rc or 1
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    print(json.dumps(report, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())