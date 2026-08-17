"""Deterministic decision rule for H0.5 evaluator.

Inputs:
  - baseline:  AggregateResult from N_BASELINE (=5) measurements on the
               pinned identity.
  - candidate: AggregateResult from N_CANDIDATE measurements on the
               same identity (a variant build, a recipe change, a tuning
               knob). Must be ``identity_hash == baseline.identity_hash``
               for a meaningful compare (the CLI enforces this).

Output:
  - PROMOTE if candidate is better than baseline by max(2σ, +3%) on the
    primary metric (output_throughput tok/s).
  - APPROVE if candidate is better than baseline but within the tight
    margin (the evaluator learns the new recipe works, does not yet
    promote it).
  - RETAIN  otherwise (no improvement).

  The decision is symmetric for cost metrics (mean_tpot_ms,
  mean_e2el_ms, prompt_eval_ms) - lower is better.

A/B/A interleaving (mandatory before PROMOTE on tiered gates) is
provided as :func:`compare_aba`. Tiered gate enforcement is the caller's
responsibility; H0.5 first pass stops at the single-tier comparison.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from typing import Literal

from .measure import AggregateResult

Direction = Literal["higher", "lower"]


@dataclasses.dataclass(frozen=True)
class Decision:
    outcome: str  # "PROMOTE" | "APPROVE" | "RETAIN" | "INCONCLUSIVE"
    reason: str
    baseline_median: float
    candidate_median: float
    delta_pct: float
    sigma_floor_pct: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _delta_pct(baseline: float, candidate: float, *, direction: Direction) -> float:
    """Signed percent improvement. Positive = improvement."""
    if baseline == 0:
        return 0.0
    raw = (candidate - baseline) / baseline
    return -raw if direction == "lower" else raw


def _sigma_floor_pct(values: list[float], direction: Direction) -> float:
    """2-sigma floor in percent (positive)."""
    if len(values) < 2:
        return 0.0
    sd = statistics.stdev(values)
    median = statistics.median(values)
    if median == 0:
        return 0.0
    # 2-sigma floor in percent of the median. Conservative for tight runs.
    raw = (2 * sd) / median
    return -raw if direction == "lower" else raw


def compare(
    *,
    baseline: AggregateResult,
    candidate: AggregateResult,
    metric: str,
    direction: Direction = "higher",
    promote_floor_pct: float = 0.03,
    approve_margin_pct: float = 0.01,
) -> Decision:
    """Compare baseline vs candidate on one metric.

    Decision thresholds:
      - PROMOTE: improvement > max(2σ of baseline, +promote_floor_pct)
      - APPROVE: improvement > +approve_margin_pct AND <= promote_floor
      - RETAIN:  otherwise (no improvement, or candidate regressed)
      - INCONCLUSIVE: empty samples on either side (caller must re-run)
    """
    b_vals = [float(getattr(m, metric)) for m in baseline.measurements]
    c_vals = [float(getattr(m, metric)) for m in candidate.measurements]
    if not b_vals or not c_vals:
        return Decision(
            outcome="INCONCLUSIVE",
            reason=f"empty samples: baseline={len(b_vals)} candidate={len(c_vals)}",
            baseline_median=statistics.median(b_vals) if b_vals else 0.0,
            candidate_median=statistics.median(c_vals) if c_vals else 0.0,
            delta_pct=0.0,
            sigma_floor_pct=0.0,
        )
    b_med = statistics.median(b_vals)
    c_med = statistics.median(c_vals)
    delta = _delta_pct(b_med, c_med, direction=direction)
    sigma_floor = max(_sigma_floor_pct(b_vals, direction), promote_floor_pct)

    if delta > sigma_floor:
        return Decision(
            outcome="PROMOTE",
            reason=(
                f"{metric}: candidate {c_med:.3f} beats baseline {b_med:.3f} "
                f"by {delta:.3%} (> sigma_floor {sigma_floor:.3%})"
            ),
            baseline_median=b_med,
            candidate_median=c_med,
            delta_pct=delta,
            sigma_floor_pct=sigma_floor,
        )
    if delta > approve_margin_pct:
        return Decision(
            outcome="APPROVE",
            reason=(
                f"{metric}: candidate {c_med:.3f} beats baseline {b_med:.3f} "
                f"by {delta:.3%} (within {promote_floor_pct:.1%} promote floor)"
            ),
            baseline_median=b_med,
            candidate_median=c_med,
            delta_pct=delta,
            sigma_floor_pct=sigma_floor,
        )
    return Decision(
        outcome="RETAIN",
        reason=(
            f"{metric}: candidate {c_med:.3f} does not beat baseline {b_med:.3f} "
            f"(delta {delta:.3%} <= approve margin {approve_margin_pct:.1%})"
        ),
        baseline_median=b_med,
        candidate_median=c_med,
        delta_pct=delta,
        sigma_floor_pct=sigma_floor,
    )


def compare_aba(
    *,
    baseline: AggregateResult,
    candidate: AggregateResult,
    follow_up: AggregateResult,
    metric: str,
    direction: Direction = "higher",
) -> Decision:
    """A/B/A: confirm the candidate's improvement holds under a fresh
    re-measurement. A PROMOTE only sticks if the candidate's median also
    beats the follow-up baseline within the same margin. Used as the
    gate before any candidate becomes a new HEAD build.
    """
    primary = compare(
        baseline=baseline,
        candidate=candidate,
        metric=metric,
        direction=direction,
    )
    if primary.outcome != "PROMOTE":
        return primary
    follow_up_cmp = compare(
        baseline=follow_up,
        candidate=candidate,
        metric=metric,
        direction=direction,
    )
    if follow_up_cmp.outcome == "PROMOTE":
        return Decision(
            outcome="PROMOTE",
            reason=(
                primary.reason
                + " | A/B/A confirmed: candidate also beats follow-up "
                f"baseline ({follow_up_cmp.delta_pct:.3%} delta)"
            ),
            baseline_median=primary.baseline_median,
            candidate_median=primary.candidate_median,
            delta_pct=primary.delta_pct,
            sigma_floor_pct=primary.sigma_floor_pct,
        )
    return Decision(
        outcome="RETAIN",
        reason=(
            primary.reason
            + f" | A/B/A failed: follow-up delta only {follow_up_cmp.delta_pct:.3%}"
        ),
        baseline_median=primary.baseline_median,
        candidate_median=primary.candidate_median,
        delta_pct=primary.delta_pct,
        sigma_floor_pct=primary.sigma_floor_pct,
    )