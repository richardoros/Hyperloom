"""Unit tests for the deterministic decision rule."""

from __future__ import annotations

import pytest

from rdna.h05.decision import compare, compare_aba
from rdna.h05.measure import Measurement, aggregate


def _m(*, tg: float, e2el: float, tpot: float = 30.0, prompt_ms: float = 1000.0) -> Measurement:
    return Measurement(
        output_throughput=tg,
        total_token_throughput=tg * 10.0,
        prompt_eval_tok_s=tg * 100.0,
        prompt_eval_ms=prompt_ms,
        mean_tpot_ms=tpot,
        mean_e2el_ms=e2el,
        quality_ok=True,
        eval_n=128,
        prompt_n=4096,
    )


class TestCompare:
    def test_empty_samples_inconclusive(self):
        b = aggregate([])
        c = aggregate([_m(tg=25.0, e2el=4000.0)])
        d = compare(baseline=b, candidate=c, metric="output_throughput", direction="higher")
        assert d.outcome == "INCONCLUSIVE"

    def test_clear_promote(self):
        # Baseline: 25 tok/s with 1.0 sd. Candidate: 27 tok/s (+8%, > 2σ + 3% floor).
        b = aggregate([_m(tg=t, e2el=4000.0) for t in (25.0, 25.5, 24.5, 25.0, 25.0)])
        c = aggregate([_m(tg=t, e2el=3800.0) for t in (27.0, 27.5, 26.5, 27.0, 27.0)])
        d = compare(baseline=b, candidate=c, metric="output_throughput", direction="higher")
        assert d.outcome == "PROMOTE"
        assert d.delta_pct > 0.07

    def test_within_margin_approve(self):
        # Baseline: 25 tok/s. Candidate: 25.5 tok/s (+2%). Inside the
        # promote floor (3%) but above approve margin (1%).
        b = aggregate([_m(tg=t, e2el=4000.0) for t in (25.0, 25.0, 25.0, 25.0, 25.0)])
        c = aggregate([_m(tg=t, e2el=4000.0) for t in (25.5, 25.5, 25.5, 25.5, 25.5)])
        d = compare(baseline=b, candidate=c, metric="output_throughput", direction="higher")
        assert d.outcome == "APPROVE"

    def test_no_improvement_retain(self):
        b = aggregate([_m(tg=t, e2el=4000.0) for t in (25.0, 25.0, 25.0, 25.0, 25.0)])
        c = aggregate([_m(tg=t, e2el=4000.0) for t in (24.0, 24.0, 24.0, 24.0, 24.0)])
        d = compare(baseline=b, candidate=c, metric="output_throughput", direction="higher")
        assert d.outcome == "RETAIN"

    def test_lower_is_better_for_latency(self):
        # Lower e2el is better. baseline=5000ms, candidate=4500ms = -10% (improvement).
        b = aggregate([_m(tg=25.0, e2el=5000.0) for _ in range(5)])
        c = aggregate([_m(tg=25.0, e2el=4500.0) for _ in range(5)])
        d = compare(baseline=b, candidate=c, metric="mean_e2el_ms", direction="lower")
        assert d.outcome == "PROMOTE"
        assert d.delta_pct > 0.09

    def test_regression_with_lower_direction(self):
        # baseline=4500ms, candidate=5000ms = +10% (regression in lower-is-better).
        b = aggregate([_m(tg=25.0, e2el=4500.0) for _ in range(5)])
        c = aggregate([_m(tg=25.0, e2el=5000.0) for _ in range(5)])
        d = compare(baseline=b, candidate=c, metric="mean_e2el_ms", direction="lower")
        assert d.outcome == "RETAIN"
        assert d.delta_pct < 0


class TestCompareABA:
    def test_promote_holds_in_follow_up(self):
        b1 = aggregate([_m(tg=t, e2el=4000.0) for t in (25.0,) * 5])
        c = aggregate([_m(tg=t, e2el=3800.0) for t in (27.0,) * 5])
        b2 = aggregate([_m(tg=t, e2el=4000.0) for t in (25.0,) * 5])
        d = compare_aba(baseline=b1, candidate=c, follow_up=b2, metric="output_throughput")
        assert d.outcome == "PROMOTE"
        assert "A/B/A confirmed" in d.reason

    def test_promote_fails_when_follow_up_regresses(self):
        b1 = aggregate([_m(tg=t, e2el=4000.0) for t in (25.0,) * 5])
        c = aggregate([_m(tg=t, e2el=3800.0) for t in (27.0,) * 5])
        b2 = aggregate([_m(tg=t, e2el=4000.0) for t in (26.5,) * 5])  # follow-up re-measured 26.5
        d = compare_aba(baseline=b1, candidate=c, follow_up=b2, metric="output_throughput")
        assert d.outcome == "RETAIN"
        assert "A/B/A failed" in d.reason

    def test_non_promote_does_not_run_aba(self):
        b1 = aggregate([_m(tg=25.0, e2el=4000.0) for _ in range(5)])
        c = aggregate([_m(tg=24.0, e2el=4000.0) for _ in range(5)])
        b2 = aggregate([_m(tg=25.0, e2el=4000.0) for _ in range(5)])
        d = compare_aba(baseline=b1, candidate=c, follow_up=b2, metric="output_throughput")
        assert d.outcome == "RETAIN"