"""Unit tests for the experiment DB."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from rdna.h05.db import Decision, ExperimentDB, utc_now
from rdna.h05.identity import IdentityBlock
from rdna.h05.measure import Measurement


def _identity(label_seed: int) -> IdentityBlock:
    return IdentityBlock(
        source_repo="/src",
        source_sha=f"sha{label_seed}",
        binary_path="/bin",
        binary_sha256=f"bin{label_seed:064d}"[-64:],
        model_path="/m",
        model_sha256=f"mdl{label_seed:064d}"[-64:],
        model_size_bytes=1000,
        gpu_type="rx7900xtx",
        gfx_arch="gfx1100",
        rocm_version="7.2.3",
        kernel_release="6.16.5",
        cmake_flags_sha="cmake",
        compiler_sha="gcc",
        cpu_model="AMD",
    )


def _measurement(tg: float) -> Measurement:
    return Measurement(
        output_throughput=tg,
        total_token_throughput=tg * 10.0,
        prompt_eval_tok_s=tg * 100.0,
        prompt_eval_ms=1000.0,
        mean_tpot_ms=30.0,
        mean_e2el_ms=4000.0,
        quality_ok=True,
        eval_n=128,
        prompt_n=4096,
    )


class TestExperimentDB:
    def test_open_experiment_round_trips(self, tmp_path: Path):
        db_path = tmp_path / "experiments.db"
        with ExperimentDB(db_path) as db:
            ib = _identity(1)
            eid = db.open_experiment(label="exp1", identity=ib)
            assert eid >= 1
            row = db.experiment(eid)
            assert row is not None
            assert row["label"] == "exp1"
            assert row["identity_hash"] == ib.identity_hash
            assert json.loads(row["identity_json"])["source_sha"] == "sha1"

    def test_record_run_appends_per_variant(self, tmp_path: Path):
        """Two runs with the same (experiment, variant, iteration) is a
        HARD ERROR: evidence must not be silently overwritten."""
        import sqlite3
        db_path = tmp_path / "experiments.db"
        ib = _identity(2)
        with ExperimentDB(db_path) as db:
            eid = db.open_experiment(label="exp2", identity=ib)
            db.record_run(
                eid, variant="baseline", iteration=0,
                identity=ib, gate_json="{}",
                measurement_json=json.dumps(_measurement(25.0).to_dict()),
                success=True,
            )
            with pytest.raises(sqlite3.IntegrityError):
                db.record_run(
                    eid, variant="baseline", iteration=0,
                    identity=ib, gate_json="{}",
                    measurement_json=json.dumps(_measurement(27.0).to_dict()),
                    success=True,
                )
            # Different variant at the same iteration is allowed.
            db.record_run(
                eid, variant="candidate", iteration=0,
                identity=ib, gate_json="{}",
                measurement_json=json.dumps(_measurement(27.0).to_dict()),
                success=True,
            )
            runs = db.experiment_runs(eid)
            assert len(runs) == 2

    def test_per_run_identity_persisted(self, tmp_path: Path):
        """Each run row carries its own identity; the experiment identity
        is the baseline identity only."""
        db_path = tmp_path / "experiments.db"
        ib_baseline = _identity(10)
        ib_candidate = _identity(11)
        ib_candidate = dataclasses.replace(ib_candidate, binary_sha256="z" * 64)  # noqa: F821
        with ExperimentDB(db_path) as db:
            eid = db.open_experiment(label="exp_per_run", identity=ib_baseline)
            db.record_run(eid, variant="baseline", iteration=0, identity=ib_baseline,
                          gate_json="{}", measurement_json="{}", success=True)
            db.record_run(eid, variant="candidate", iteration=0, identity=ib_candidate,
                          gate_json="{}", measurement_json="{}", success=True)
            runs = db.experiment_runs(eid)
            assert len(runs) == 2
            baseline_run = next(r for r in runs if r["variant"] == "baseline")
            candidate_run = next(r for r in runs if r["variant"] == "candidate")
            assert baseline_run["identity_hash"] == ib_baseline.identity_hash
            assert candidate_run["identity_hash"] == ib_candidate.identity_hash
            assert baseline_run["identity_hash"] != candidate_run["identity_hash"]

    def test_record_decision(self, tmp_path: Path):
        db_path = tmp_path / "experiments.db"
        with ExperimentDB(db_path) as db:
            ib = _identity(3)
            eid = db.open_experiment(label="exp3", identity=ib)
            db.record_decision(eid, decision=Decision.PROMOTE, reason="test", metrics={"k": 1.0})
            db.close_experiment(eid, status="DECIDED", decision=Decision.PROMOTE, reason="test")
            row = db.experiment(eid)
            assert row["decision"] == "PROMOTE"
            assert row["status"] == "DECIDED"

    def test_list_experiments_orders_newest_first(self, tmp_path: Path):
        db_path = tmp_path / "experiments.db"
        with ExperimentDB(db_path) as db:
            db.open_experiment(label="a", identity=_identity(4))
            db.open_experiment(label="b", identity=_identity(5))
            rows = db.list_experiments()
            assert [r["label"] for r in rows] == ["b", "a"]