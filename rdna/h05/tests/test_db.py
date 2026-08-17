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

    def test_record_run_unique_on_iteration(self, tmp_path: Path):
        db_path = tmp_path / "experiments.db"
        with ExperimentDB(db_path) as db:
            ib = _identity(2)
            eid = db.open_experiment(label="exp2", identity=ib)
            db.record_run(eid, iteration=0, gate_json="{}", measurement_json=json.dumps(_measurement(25.0).to_dict()), success=True)
            # Re-record the same iteration should overwrite, not error.
            db.record_run(eid, iteration=0, gate_json="{}", measurement_json=json.dumps(_measurement(27.0).to_dict()), success=True)
            runs = db.experiment_runs(eid)
            assert len(runs) == 1
            assert json.loads(runs[0]["measurement_json"])["output_throughput"] == 27.0

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