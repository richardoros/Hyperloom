"""Experiment DB for H0.5 evaluator.

Four tables, one SQLite file, no migrations:

* ``experiments``: one row per evaluation. Identity triple + label +
  status + decision (RETAIN / APPROVE / PROMOTE / BLOCKED / ERROR).
* ``runs``:         one row per measurement repetition. Tied to an
  experiment; carries the iteration index, the gate report, and the
  raw measurement JSON.
* ``decisions``:   one row per decision event. Records the rationale and
  the metric values that triggered it (for audit replay).
* ``meta``:        free-form key-value (schema version, db version).

Why SQLite: the evaluator runs on one host, never networked, with serial
inserts. A single-process file DB gives us crash-safe atomic writes via
``isolation_level=EXCLUSIVE`` and zero ops burden.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import sqlite3
import time
from pathlib import Path

from .identity import IdentityBlock
from .measure import AggregateResult


SCHEMA_VERSION = 1


class Decision(str, enum.Enum):
    RETAIN = "RETAIN"
    APPROVE = "APPROVE"
    PROMOTE = "PROMOTE"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"
    PENDING = "PENDING"


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    label           TEXT NOT NULL UNIQUE,
    identity_json   TEXT NOT NULL,
    identity_hash   TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'PENDING',
    decision        TEXT NOT NULL DEFAULT 'PENDING',
    decision_reason TEXT NOT NULL DEFAULT '',
    created_utc     TEXT NOT NULL,
    closed_utc      TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(id),
    iteration       INTEGER NOT NULL,
    gate_json       TEXT NOT NULL,
    measurement_json TEXT NOT NULL,
    success         INTEGER NOT NULL,
    created_utc     TEXT NOT NULL,
    UNIQUE(experiment_id, iteration)
);
CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(id),
    decision        TEXT NOT NULL,
    reason          TEXT NOT NULL,
    metrics_json    TEXT NOT NULL,
    created_utc     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experiments_hash ON experiments(identity_hash);
CREATE INDEX IF NOT EXISTS idx_runs_experiment  ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_decisions_experiment ON decisions(experiment_id);
"""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class ExperimentDB:
    """Single-file SQLite wrapper. Use as a context manager for atomicity."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # EXCLUSIVE isolation -> writes serialize, no other writers can race.
        self._conn = sqlite3.connect(str(self.db_path), isolation_level="EXCLUSIVE")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ExperimentDB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- experiments ----------------------------------------------------------

    def open_experiment(self, *, label: str, identity: IdentityBlock) -> int:
        identity_json = identity.to_json()
        identity_hash = identity.identity_hash
        cur = self._conn.execute(
            "INSERT INTO experiments(label, identity_json, identity_hash, status, created_utc) "
            "VALUES (?, ?, ?, 'PENDING', ?)",
            (label, identity_json, identity_hash, utc_now()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def close_experiment(self, experiment_id: int, *, status: str, decision: Decision, reason: str) -> None:
        self._conn.execute(
            "UPDATE experiments SET status = ?, decision = ?, decision_reason = ?, closed_utc = ? "
            "WHERE id = ?",
            (status, decision.value, reason, utc_now(), experiment_id),
        )
        self._conn.commit()

    def record_decision(
        self,
        experiment_id: int,
        *,
        decision: Decision,
        reason: str,
        metrics: dict,
    ) -> None:
        self._conn.execute(
            "INSERT INTO decisions(experiment_id, decision, reason, metrics_json, created_utc) "
            "VALUES (?, ?, ?, ?, ?)",
            (experiment_id, decision.value, reason, json.dumps(metrics, sort_keys=True), utc_now()),
        )
        self._conn.commit()

    # -- runs -----------------------------------------------------------------

    def record_run(
        self,
        experiment_id: int,
        *,
        iteration: int,
        gate_json: str,
        measurement_json: str,
        success: bool,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO runs(experiment_id, iteration, gate_json, measurement_json, success, created_utc) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(experiment_id, iteration) DO UPDATE SET "
            "gate_json = excluded.gate_json, measurement_json = excluded.measurement_json, "
            "success = excluded.success, created_utc = excluded.created_utc",
            (
                experiment_id,
                iteration,
                gate_json,
                measurement_json,
                1 if success else 0,
                utc_now(),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    # -- queries --------------------------------------------------------------

    def list_experiments(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT id, label, identity_hash, status, decision, decision_reason, "
            "created_utc, closed_utc FROM experiments ORDER BY id DESC"
        ).fetchall()

    def experiment(self, experiment_id: int) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()

    def experiment_runs(self, experiment_id: int) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT iteration, gate_json, measurement_json, success, created_utc "
            "FROM runs WHERE experiment_id = ? ORDER BY iteration",
            (experiment_id,),
        ).fetchall()