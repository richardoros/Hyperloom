"""H0.5 evaluator CLI.

Subcommands:

* ``baseline``  : run N measurements on the current pinned identity and
                  record them as the experiment's baseline set.
* ``candidate`` : same, but as the candidate set on an experiment label.
* ``decide``    : read both sets from the DB and emit a Decision.
* ``list``      : list all experiments.
* ``show``      : show one experiment's identity + runs.

The CLI is deliberately small. Everything observable about a measurement
is in the DB; the CLI is the operator's window into it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .db import Decision, ExperimentDB
from .decision import compare
from .gate import gate
from .identity import IdentityBlock, collect_identity
from .measure import Measurement, METRIC_FIELDS, aggregate, run_repeat


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------


def _common_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--source-repo",
        default="/home/homelabserver/src/llama.cpp-turboquant",
        help="llama.cpp checkout path (source SHA from .git/HEAD).",
    )
    p.add_argument(
        "--binary",
        default="/home/homelabserver/src/llama.cpp-turboquant/build/bin/llama-server",
        help="llama-server binary path.",
    )
    p.add_argument(
        "--model",
        default="/home/homelabserver/models/Qwen3.8-27B-UD-Q5_K_XL.gguf",
        help="GGUF model path.",
    )
    p.add_argument(
        "--gpu-type",
        default="rx7900xtx",
        help="gpu_type key (rx7900xtx, radeon890m, mi300x, ...).",
    )
    p.add_argument(
        "--bench-scripts-dir",
        default=str(Path(__file__).resolve().parent.parent / "bench"),
        help="Custom framework benchmark scripts directory.",
    )
    p.add_argument(
        "--runner-type",
        default="rx7900xtx",
        help="Runner type for the bypass-scriptable executor (matches --gpu-type).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=18179,
        help="Experiment port (NOT 18079 - that is production).",
    )
    p.add_argument("--db", default="rdna/h05_results/experiments.db", help="Path to the SQLite experiment DB.")
    p.add_argument(
        "--extra-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Forward extra env vars to the bench script (e.g. TURBO_TCQ_CB=...). Repeatable.",
    )
    p.add_argument("--repeat", type=int, default=5, help="Number of independent measurements to run.")
    p.add_argument("--timeout-s", type=int, default=900, help="Per-measurement timeout (seconds).")
    return p


def _parse_extra_env(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--extra-env must be KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v
    return out


# ---------------------------------------------------------------------------
# Identity + gate (shared)
# ---------------------------------------------------------------------------


def _identity_and_gate(args: argparse.Namespace) -> tuple[IdentityBlock, "object"]:
    identity = collect_identity(
        source_repo=args.source_repo,
        binary_path=args.binary,
        model_path=args.model,
        gpu_type=args.gpu_type,
        gfx_arch=None,  # let detect infer from gpu_type
    )
    if identity.model_size_bytes is None:
        raise SystemExit(f"model file not found: {identity.model_path}")
    required = identity.model_size_bytes  # GGUF size is a safe lower bound
    report = gate(
        identity=identity,
        exp_port=args.port,
        required_bytes=required,
    )
    return identity, report


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_baseline(args: argparse.Namespace) -> int:
    identity, gate_report = _identity_and_gate(args)
    if not gate_report.ok:
        # Record BLOCKED in the DB so the operator sees the attempt in the
        # experiment list and the failure mode is auditable.
        try:
            with ExperimentDB(args.db) as db:
                eid = db.open_experiment(label=args.label, identity=identity)
                db.record_run(
                    eid,
                    variant="baseline",
                    iteration=0,
                    identity=identity,
                    gate_json=gate_report.to_json(),
                    measurement_json=json.dumps({"blocked": True}),
                    success=False,
                )
                db.close_experiment(
                    eid,
                    status="BLOCKED",
                    decision=Decision.BLOCKED,
                    reason=gate_report.reason,
                )
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"gate": "FAIL", "reason": gate_report.reason, "db_error": repr(exc)}))
            return 42
        print(json.dumps({"gate": "FAIL", "reason": gate_report.reason, "experiment_id": eid}))
        return 42
    extra = _parse_extra_env(args.extra_env)
    with ExperimentDB(args.db) as db:
        eid = db.open_experiment(label=args.label, identity=identity)
        agg = run_repeat(
            model=identity.model_path,
            framework_repo=args.source_repo,
            bench_scripts_dir=args.bench_scripts_dir,
            runner_type=args.runner_type,
            port=args.port,
            repeat=args.repeat,
            extra_envs=extra,
            timeout_s=args.timeout_s,
            work_root=Path(f"rdna/h05_results/{args.label}"),
        )
        for i, m in enumerate(agg.measurements):
            db.record_run(
                eid,
                variant="baseline",
                iteration=i,
                identity=identity,
                gate_json=gate_report.to_json(),
                measurement_json=json.dumps(m.to_dict()),
                success=m.quality_ok,
            )
        db.close_experiment(
            eid,
            status="COMPLETED",
            decision=Decision.PENDING,
            reason=f"{args.repeat} measurements, baseline established",
        )
        print(json.dumps({"experiment_id": eid, "label": args.label, "baseline": agg.to_dict()}))
    return 0


def cmd_candidate(args: argparse.Namespace) -> int:
    identity, gate_report = _identity_and_gate(args)
    if not gate_report.ok:
        # Same BLOCKED semantics as cmd_baseline.
        try:
            with ExperimentDB(args.db) as db:
                row = db._conn.execute(  # noqa: SLF001
                    "SELECT id FROM experiments WHERE label = ?", (args.label,)
                ).fetchone()
                if row is None:
                    raise SystemExit(
                        f"experiment not found: {args.label}; run `baseline` first"
                    )
                eid = int(row[0])
                db.record_run(
                    eid,
                    variant="candidate",
                    iteration=0,
                    identity=identity,
                    gate_json=gate_report.to_json(),
                    measurement_json=json.dumps({"blocked": True}),
                    success=False,
                )
                db.close_experiment(
                    eid,
                    status="BLOCKED",
                    decision=Decision.BLOCKED,
                    reason=gate_report.reason,
                )
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"gate": "FAIL", "reason": gate_report.reason, "db_error": repr(exc)}))
            return 42
        print(json.dumps({"gate": "FAIL", "reason": gate_report.reason, "experiment_id": eid}))
        return 42
    extra = _parse_extra_env(args.extra_env)
    with ExperimentDB(args.db) as db:
        # Find the experiment by label; reuse it for candidate runs.
        row = db._conn.execute(  # noqa: SLF001 - CLI helper
            "SELECT id FROM experiments WHERE label = ?", (args.label,)
        ).fetchone()
        if row is None:
            raise SystemExit(f"experiment not found: {args.label}; run `baseline` first")
        eid = int(row[0])
        agg = run_repeat(
            model=identity.model_path,
            framework_repo=args.source_repo,
            bench_scripts_dir=args.bench_scripts_dir,
            runner_type=args.runner_type,
            port=args.port,
            repeat=args.repeat,
            extra_envs=extra,
            timeout_s=args.timeout_s,
            work_root=Path(f"rdna/h05_results/{args.label}_candidate"),
        )
        # Candidate runs append within (experiment_id, variant='candidate').
        for i, m in enumerate(agg.measurements):
            db.record_run(
                eid,
                variant="candidate",
                iteration=i,
                identity=identity,
                gate_json=gate_report.to_json(),
                measurement_json=json.dumps(m.to_dict()),
                success=m.quality_ok,
            )
        db.close_experiment(
            eid,
            status="COMPLETED",
            decision=Decision.PENDING,
            reason=f"candidate: {args.repeat} measurements appended",
        )
        print(json.dumps({"experiment_id": eid, "label": args.label, "candidate": agg.to_dict()}))
    return 0


def _load_aggregate(
    db: ExperimentDB,
    experiment_id: int,
    *,
    variant: str,
    count: int,
) -> "object":
    """Load the first ``count`` non-blocked runs for one variant of an
    experiment. Blocked rows (recorded as evidence of a failed gate)
    are skipped — they do not contribute to the variance baseline."""
    rows = db.experiment_runs(experiment_id, variant=variant)
    measurements: list[Measurement] = []
    for r in rows:
        payload = json.loads(r["measurement_json"])
        if payload.get("blocked"):
            continue
        measurements.append(Measurement.from_inferencex(payload))
        if len(measurements) >= count:
            break
    if len(measurements) < count:
        raise SystemExit(
            f"experiment {experiment_id}: variant={variant} expected at least {count} "
            f"non-blocked runs, have {len(measurements)}"
        )
    return aggregate(measurements)


def cmd_decide(args: argparse.Namespace) -> int:
    with ExperimentDB(args.db) as db:
        rows = db.list_experiments()
        row = next((r for r in rows if r["label"] == args.label), None)
        if row is None:
            raise SystemExit(f"experiment not found: {args.label}")
        eid = int(row["id"])
        # Per-variant counts. cmd_baseline appends variant='baseline',
        # cmd_candidate appends variant='candidate'. BLOCKED rows are
        # recorded as evidence but do not contribute to variance.
        baseline = _load_aggregate(db, eid, variant="baseline", count=args.repeat)
        candidate = _load_aggregate(db, eid, variant="candidate", count=args.repeat)
        per_metric: dict[str, dict] = {}
        outcomes: list[str] = []
        for metric in METRIC_FIELDS:
            direction = "lower" if metric in ("mean_tpot_ms", "mean_e2el_ms", "prompt_eval_ms") else "higher"
            # P0.4: do NOT pass allow_promote=True here. PROMOTE stays
            # gated on A/B/A + product-quality gate (not yet wired). The
            # CLI's plain compare path emits APPROVE / RETAIN only; the
            # A/B/A gate will be a separate ``decide-aba`` subcommand in
            # H0.7 once the agent loop drives A/B/A experiments.
            decision = compare(
                baseline=baseline,
                candidate=candidate,
                metric=metric,
                direction=direction,
            )
            per_metric[metric] = decision.to_dict()
            outcomes.append(decision.outcome)
        # Aggregate rule. PROMOTE never appears in this path (allow_promote
        # default False). APPROVE if any metric approved and none regressed;
        # RETAIN if anything regressed or inconclusive.
        if all(o == "APPROVE" for o in outcomes):
            overall = "APPROVE"
        elif "RETAIN" in outcomes or "INCONCLUSIVE" in outcomes:
            overall = "RETAIN"
        else:
            # Mixed: some APPROVE, some unchanged. Demote to RETAIN until
            # A/B/A confirms.
            overall = "RETAIN"
        reason = (
            f"{overall}: "
            + ", ".join(f"{m}={d}" for m, d in per_metric.items())
        )
        db.record_decision(
            eid,
            decision=Decision(overall),
            reason=reason,
            metrics=per_metric,
        )
        db.close_experiment(eid, status="DECIDED", decision=Decision(overall), reason=reason)
        print(json.dumps({"experiment_id": eid, "label": args.label, "decision": overall, "per_metric": per_metric, "reason": reason}))
    return 0 if overall == "APPROVE" else 1


def cmd_list(args: argparse.Namespace) -> int:
    with ExperimentDB(args.db) as db:
        for r in db.list_experiments():
            print(f"{r['id']:>4}  {r['created_utc']}  {r['status']:<10}  {r['decision']:<8}  {r['label']}  hash={r['identity_hash'][:12]}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    with ExperimentDB(args.db) as db:
        row = db._conn.execute(  # noqa: SLF001
            "SELECT * FROM experiments WHERE label = ?", (args.label,)
        ).fetchone()
        if row is None:
            raise SystemExit(f"experiment not found: {args.label}")
        print(json.dumps(json.loads(row["identity_json"]), indent=2))
        print(f"\nstatus={row['status']} decision={row['decision']} reason={row['decision_reason']}")
        runs = db.experiment_runs(int(row["id"]))
        current_variant = None
        for r in runs:
            if r["variant"] != current_variant:
                current_variant = r["variant"]
                print(f"\n  --- variant={r['variant']} ---")
            m = json.loads(r["measurement_json"])
            if m.get("blocked"):
                print(
                    f"  iter={r['iteration']:>2} BLOCKED (gate fail) "
                    f"identity={r['identity_hash'][:12]}"
                )
                continue
            print(
                f"  iter={r['iteration']:>2} success={bool(r['success'])} "
                f"output={m['output_throughput']:.2f} tok/s "
                f"e2el={m['mean_e2el_ms']:.1f}ms "
                f"tpot={m['mean_tpot_ms']:.2f}ms "
                f"identity={r['identity_hash'][:12]}"
            )
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    common = _common_parser()
    parser = argparse.ArgumentParser(
        prog="h05",
        description="H0.5 evaluator for the RDNA3 lane.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_baseline = sub.add_parser("baseline", help="Run a baseline set.", parents=[common])
    p_baseline.add_argument("--label", required=True, help="Experiment label.")
    p_baseline.set_defaults(func=cmd_baseline)

    p_candidate = sub.add_parser("candidate", help="Run a candidate set (after baseline).", parents=[common])
    p_candidate.add_argument("--label", required=True, help="Experiment label (must already exist).")
    p_candidate.set_defaults(func=cmd_candidate)

    p_decide = sub.add_parser("decide", help="Compare baseline vs candidate for an experiment.", parents=[common])
    p_decide.add_argument("--label", required=True, help="Experiment label.")
    p_decide.set_defaults(func=cmd_decide)

    p_list = sub.add_parser("list", help="List experiments.", parents=[common])
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Show one experiment.", parents=[common])
    p_show.add_argument("--label", required=True, help="Experiment label.")
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())