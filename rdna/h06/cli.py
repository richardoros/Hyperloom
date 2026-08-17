"""H0.6 exclusive-XTX lifecycle CLI.

Subcommand:

    python -m rdna.h06.cli measure \\
        --experiment-id 1 \\
        --measurement echo_success \\
        --artifact-dir rdna/h06_results/artifacts

The ``measurement`` callable is resolved by name. Built-ins:

    echo_success  → returns 0 (no real measurement; for dry runs)
    echo_fail     → raises RuntimeError (tests the trap)

Real measurement callables live in H0.7 (the agent loop). For H0.6
the dry runs are enough to prove the lifecycle is correct.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .orchestrator import Outcome, run_lifecycle


_BUILTIN_MEASUREMENTS = {
    "echo_success": lambda _ctx: 0,
    "echo_fail": lambda _ctx: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="h06",
        description="H0.6 exclusive-XTX lifecycle orchestrator.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_measure = sub.add_parser("measure", help="Run the lifecycle around a measurement.")
    p_measure.add_argument("--experiment-id", type=int, required=True)
    p_measure.add_argument(
        "--measurement", default="echo_success",
        help="Measurement callable name (built-in: echo_success, echo_fail).",
    )
    p_measure.add_argument(
        "--artifact-dir", default="rdna/h06_results/artifacts",
        help="Directory for the lifecycle audit artifact.",
    )
    p_measure.add_argument(
        "--allow-service", action="append", default=[],
        help="Service name the orchestrator may stop (repeatable).",
    )
    p_measure.add_argument(
        "--allowed-pid", action="append", type=int, default=[],
        help="PID that may legitimately use the lab GPU (repeatable).",
    )
    p_measure.add_argument(
        "--snapshot-override", default=None,
        help="JSON file with a PreStateSnapshot to use instead of take_snapshot.",
    )
    p_measure.set_defaults(func=cmd_measure)
    args = parser.parse_args(argv)

    if args.cmd == "measure":
        return cmd_measure(args)
    parser.error(f"unknown subcommand: {args.cmd}")  # noqa: SLF001 - argparse
    return 2  # unreachable


def cmd_measure(args: argparse.Namespace) -> int:
    measurement_name = args.measurement
    if measurement_name not in _BUILTIN_MEASUREMENTS:
        print(json.dumps({
            "error": "unknown measurement",
            "measurement": measurement_name,
            "available": list(_BUILTIN_MEASUREMENTS),
        }))
        return 2
    measurement = _BUILTIN_MEASUREMENTS[measurement_name]
    snapshot_override = None
    if args.snapshot_override:
        from .snapshot import PreStateSnapshot
        snapshot_override = PreStateSnapshot(**json.loads(Path(args.snapshot_override).read_text()))
    audit = run_lifecycle(
        measurement=measurement,
        experiment_id=args.experiment_id,
        artifact_dir=args.artifact_dir,
        allowlist=tuple(args.allow_service),
        allowed_pids=tuple(args.allowed_pid),
        snapshot_override=snapshot_override,
    )
    print(json.dumps({"outcome": audit.outcome.value, "reason": audit.reason}))
    return 0 if audit.outcome == Outcome.PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())