"""Measurement loop for H0.5 evaluator.

Wraps the lower-level ``_run_scriptable_benchmark`` executor from the
Hyperloom fork (the same seam the H0.0 smoke driver used) so the evaluator
is deterministic regardless of the Hyperloom CLI's PRELUDE behaviour
(see H0-GAP-001).

Each measurement is one complete server-launch + n-token completion
cycle. ``repeat`` runs the cycle N times and returns the list of
``inferencex_result.json`` payloads plus a bootstrap CI on the key
metrics (output_throughput, total_token_throughput, mean_tpot_ms,
mean_e2el_ms, prompt_eval_ms).

The gate runs before the FIRST launch; the evaluator launches and tears
down a fresh server per repetition so that warm caches / prefill state
do not cross-contaminate the variance baseline.
"""

from __future__ import annotations

import dataclasses
import json
import os
import statistics
import sys
from pathlib import Path

# Locate the fork's src so the scriptable executor is importable without
# the user having to pip-install anything.
_HERE = Path(__file__).resolve()
_FORK = _HERE.parents[2]  # rdna/h05/measure.py -> rdna/h05/ -> rdna/ -> fork-root
if str(_FORK / "src") not in sys.path:
    sys.path.insert(0, str(_FORK / "src"))


@dataclasses.dataclass(frozen=True)
class Measurement:
    """One InferenceX-shaped result, normalized to floats."""

    output_throughput: float
    total_token_throughput: float
    prompt_eval_tok_s: float
    prompt_eval_ms: float
    mean_tpot_ms: float
    mean_e2el_ms: float
    quality_ok: bool
    eval_n: int
    prompt_n: int

    @classmethod
    def from_inferencex(cls, payload: dict) -> "Measurement":
        return cls(
            output_throughput=float(payload.get("output_throughput", 0.0)),
            total_token_throughput=float(payload.get("total_token_throughput", 0.0)),
            prompt_eval_tok_s=float(payload.get("prompt_eval_tok_s", 0.0)),
            prompt_eval_ms=float(payload.get("prompt_eval_ms", 0.0)),
            mean_tpot_ms=float(payload.get("mean_tpot_ms", 0.0)),
            mean_e2el_ms=float(payload.get("mean_e2el_ms", 0.0)),
            quality_ok=bool(payload.get("quality_gate", {}).get("passed", False)),
            eval_n=int(payload.get("total_output_tokens", 0)),
            prompt_n=int(payload.get("total_input_tokens", 0)),
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class CIBounds:
    """Bootstrap 95 % CI on one metric."""

    low: float
    high: float
    median: float
    n: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class AggregateResult:
    """N measurements + per-metric CIs + a single median point."""

    measurements: list[Measurement]
    ci: dict[str, CIBounds]

    def median(self, field: str) -> float:
        vals = [float(getattr(m, field)) for m in self.measurements]
        return statistics.median(vals) if vals else 0.0

    def to_dict(self) -> dict:
        return {
            "measurements": [m.to_dict() for m in self.measurements],
            "ci": {k: v.to_dict() for k, v in self.ci.items()},
        }


# Metrics we track end-to-end. Order matches the DB column order.
METRIC_FIELDS: tuple[str, ...] = (
    "output_throughput",
    "total_token_throughput",
    "prompt_eval_tok_s",
    "prompt_eval_ms",
    "mean_tpot_ms",
    "mean_e2el_ms",
)


def bootstrap_ci(values: list[float], *, n_boot: int = 2000, alpha: float = 0.05) -> CIBounds:
    """Non-parametric bootstrap CI on a 1-D sample.

    The 2-sigma floor + max(2σ, +3%) promotion rule in the decision module
    relies on this being robust to small N (N=3 is the minimum we run).
    """
    n = len(values)
    if n == 0:
        return CIBounds(low=0.0, high=0.0, median=0.0, n=0)
    import random

    rng = random.Random(0xC0FFEE)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return CIBounds(low=lo, high=hi, median=statistics.median(values), n=n)


def aggregate(measurements: list[Measurement]) -> AggregateResult:
    ci: dict[str, CIBounds] = {}
    for field in METRIC_FIELDS:
        vals = [float(getattr(m, field)) for m in measurements]
        ci[field] = bootstrap_ci(vals)
    return AggregateResult(measurements=measurements, ci=ci)


# ---------------------------------------------------------------------------
# Wrapping the scriptable executor from the Hyperloom fork
# ---------------------------------------------------------------------------


def _run_one(
    *,
    model: str,
    framework_repo: str,
    bench_scripts_dir: str,
    runner_type: str,
    port: int,
    extra_envs: dict[str, str] | None = None,
    timeout_s: int = 900,
    output_root: Path,
) -> Measurement:
    """One full cycle: launch the bypass-scriptable executor, capture its
    ``inferencex_result.json``, and return a normalised Measurement.
    """
    # Import lazily so the import path is set up before the orchestrator
    # code touches its env-safety / kb / bootstrap machinery.
    from hyperloom.orchestrator.actions.executors.bypass_runner import (  # type: ignore
        _run_scriptable_benchmark,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    envs = dict(extra_envs or {})
    envs.setdefault("PORT", str(port))
    bench = {
        "framework": "custom",
        "model": model,
        "runner_type": runner_type,
        "precision": "q5_k_xl",
        "profiler": {
            "torch_profiler": {"enabled": False},
            "system_profiler": {"enabled": False},
            "tracelens": {"enabled": False},
        },
        "envs": envs,
        "timeout_seconds": timeout_s,
    }
    # Required so resolve_scriptable_script picks our bench dir / framework repo.
    os.environ.setdefault("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(bench_scripts_dir))
    os.environ.setdefault("HYPERLOOM_BENCHMARK_BACKEND", "bypass")
    os.environ.setdefault("FRAMEWORK_REPO_PATH", str(framework_repo))

    rc = _run_scriptable_benchmark(
        framework="custom",
        model=model,
        bench=bench,
        bench_envs={},
        timeout_s=timeout_s,
        output_dir=output_root,
    )
    if rc != 0:
        raise RuntimeError(f"scriptable benchmark exited with rc={rc}")
    # The executor writes <output_dir>/benchmark_custom_<ts>/inferencex_result.json
    candidates = sorted(output_root.glob("benchmark_custom_*/inferencex_result.json"))
    if not candidates:
        raise RuntimeError(f"no inferencex_result.json under {output_root}")
    payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
    return Measurement.from_inferencex(payload)


def run_repeat(
    *,
    model: str,
    framework_repo: str,
    bench_scripts_dir: str,
    runner_type: str,
    port: int,
    repeat: int,
    extra_envs: dict[str, str] | None = None,
    timeout_s: int = 900,
    work_root: Path,
) -> AggregateResult:
    """Run N independent measurements and aggregate."""
    measurements: list[Measurement] = []
    for i in range(repeat):
        per_run = work_root / f"run_{i:02d}"
        m = _run_one(
            model=model,
            framework_repo=framework_repo,
            bench_scripts_dir=bench_scripts_dir,
            runner_type=runner_type,
            port=port,
            extra_envs=extra_envs,
            timeout_s=timeout_s,
            output_root=per_run,
        )
        measurements.append(m)
    return aggregate(measurements)