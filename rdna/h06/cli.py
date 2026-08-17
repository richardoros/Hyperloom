"""H0.6 exclusive-XTX lifecycle CLI.

Subcommand:

    python -m rdna.h06.cli measure \\
        --experiment-id N \\
        --measurement echo_success | h05_baseline \\
        --timeout-seconds 600 \\
        --artifact-dir rdna/h06_results/artifacts

Measurements:

    echo_success     → returns 0 immediately (dry-run success).
    echo_fail        → raises RuntimeError (tests the trap path).
    echo_nonzero     → returns 1 (exercises the FAIL-on-non-zero path).
    h05_baseline     → runs the real H0.5 baseline measurement via
                        rdna/h05/measure.py. Spawns the custom-framework
                        scriptable executor under a NEW process group
                        (start_new_session=True). The orchestrator's
                        trap kills the entire PGID on exit / signal /
                        timeout / exception. The bench script reads
                        HYPERLOOM_FIXTURE_PATH and replays the pinned
                        numeric exact-token fixture.

The lifecycle audit artifact lands in
``rdna/h06_results/lifecycle_<N>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

from .orchestrator import (
    MeasurementContext,
    MeasurementResult,
    Outcome,
    run_lifecycle,
)


def _echo_success(ctx: MeasurementContext) -> MeasurementResult:
    return MeasurementResult(exit_code=0, candidate_pid=None)


def _echo_fail(ctx: MeasurementContext) -> None:  # never returns normally
    raise RuntimeError("synthetic failure from echo_fail")


def _echo_nonzero(ctx: MeasurementContext) -> MeasurementResult:
    return MeasurementResult(exit_code=1, candidate_pid=None)


def _echo_slow(ctx: MeasurementContext) -> MeasurementResult:
    """Sleep through the deadline; SIGKILL via the trap path on exit.

    This routine is used by the timeout classification test. It MUST
    NOT register the orchestrator's own PID/PGID as the candidate (the
    trap would then kill the orchestrator itself, recursing into the
    signal handler). Instead it spawns a CHILD subprocess with its own
    session and registers the child's PID. The trap kills the child
    on exit.
    """
    import subprocess
    proc = subprocess.Popen(
        ["sleep", "60"], start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    ctx.register_candidate_pid(proc.pid)
    try:
        ctx.register_candidate_pgid(os.getpgid(proc.pid))
    except (OSError, ProcessLookupError):
        pass
    try:
        stdout, stderr = proc.communicate(timeout=30)
        return MeasurementResult(
            exit_code=proc.returncode, candidate_pid=proc.pid,
        )
    except subprocess.TimeoutExpired:
        # Hard-kill the whole session so the sleep dies.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        return MeasurementResult(
            exit_code=137, candidate_pid=proc.pid, timed_out=True,
        )


def _h05_baseline(ctx: MeasurementContext) -> MeasurementResult:
    """Real H0.5 baseline measurement callable.

    Spawns the bypass-scriptable executor via rdna/h05/measure.py
    inside a NEW process group (start_new_session=True). The candidate
    llama-server + its driver threads are descendants of that group;
    the orchestrator's RestoreTrap will SIGTERM/SIGKILL the entire
    PGID on exit / signal / timeout / exception. With an explicit
    NUMERIC-TOKEN fixture path the measurement is replayable.

    The H0.5 measurement record is persisted into the per-run
    workspace under rdna/h05_results/<experiment_id>/.
    """
    import os
    import shutil
    import subprocess
    import sys
    import time
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    work_dir = repo_root / "rdna" / "h05_results" / f"experiment_{ctx.experiment_id}_h06_baseline"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Fixture: explicit path first; otherwise DEFAULT_FIXTURE. The
    # orchestrator MUST pass an explicit path for H0.5 to be
    # scientifically repeatable; falling back to DEFAULT_FIXTURE is
    # a back-compat path that the bench CLI flags as not-a-real-baseline.
    fixture_path = os.environ.get("HYPERLOOM_FIXTURE_PATH", "").strip()
    if not fixture_path:
        print(
            json.dumps({
                "warning": (
                    "no --fixture-path supplied; bench CLI will fall back "
                    "to DEFAULT_FIXTURE (string prompt). For an H0.5 "
                    "reproducible baseline, run 'generate_fixture' against "
                    "the production GGUF and pass --fixture-path."
                ),
            })
        )
        fixture_path = ""
    env = dict(os.environ)
    env.setdefault("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(repo_root / "rdna" / "bench"))
    env.setdefault("FRAMEWORK_REPO_PATH", "/home/homelabserver/src/llama.cpp-turboquant")
    env["HYPERLOOM_BENCHMARK_BACKEND"] = "bypass"
    env["PORT"] = "18180"
    if fixture_path:
        env["FIXTURE_PATH"] = fixture_path

    # H0.6.2: start a brand new process group so the orchestrator's
    # trap can os.killpg the whole tree on exit (wrapper + bypass +
    # llama-server + its driver threads). Detach from the controlling
    # terminal.
    deadline_s = ctx._deadline_seconds if hasattr(ctx, "_deadline_seconds") else None
    if deadline_s is None:
        deadline_s = 900.0

    cmd = [
        sys.executable, "-c",
        "import sys; sys.path.insert(0, 'rdna');"
        "from h05.measure import run_repeat;"
        "from h05.identity import collect_identity;"
        f"import json, pathlib;"
        f"out = pathlib.Path('rdna/h05_results/experiment_{ctx.experiment_id}_h06_baseline');"
        "out.mkdir(parents=True, exist_ok=True);"
        "ident = collect_identity("
        "  source_repo='/home/homelabserver/src/llama.cpp-turboquant',"
        "  binary_path='/home/homelabserver/src/llama.cpp-turboquant/build/bin/llama-server',"
        "  model_path='/home/homelabserver/models/Qwen3.8-27B-UD-Q5_K_XL.gguf',"
        "  gpu_type='rx7900xtx',"
        ");"
        f"agg = run_repeat("
        "  model=ident.model_path,"
        "  framework_repo='/home/homelabserver/src/llama.cpp-turboquant',"
        "  bench_scripts_dir='rdna/bench',"
        "  runner_type='rx7900xtx',"
        "  port=18180,"
        "  repeat=5,"
        "  identity_provider=lambda: ident,"
        f"  work_root=out,"
        ");"
        "print(json.dumps({'aggregated': agg.to_dict(), 'identity': {'sha': ident.identity_hash}}))"
    ]

    proc = subprocess.Popen(
        cmd, cwd=str(repo_root), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,  # NEW PGID for the whole tree
    )
    # Register PID + PGID so the trap kills the entire subtree on exit
    # (not just the wrapper).
    ctx.register_candidate_pid(proc.pid)
    try:
        ctx.register_candidate_pgid(os.getpgid(proc.pid))
    except (OSError, ProcessLookupError):
        pass

    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=deadline_s)
    except subprocess.TimeoutExpired:
        # H0.6.2: SIGKILL the whole process group (catches wrapper +
        # bypass + llama-server descendants even if the wrapper is
        # wedged). `Popen.pid` is also the PGID when start_new_session=True.
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        return MeasurementResult(
            exit_code=137,  # SIGKILL = 128 + 9
            candidate_pid=proc.pid,
            timed_out=timed_out,
        )

    if stderr:
        print(f"[h06 h05_baseline] stderr: {stderr[-2000:]}")
    return MeasurementResult(
        exit_code=proc.returncode,
        candidate_pid=proc.pid,
        timed_out=timed_out,
    )


_BUILTIN_MEASUREMENTS: dict[str, callable] = {
    "echo_success": _echo_success,
    "echo_fail": _echo_fail,
    "echo_nonzero": _echo_nonzero,
    "echo_slow": _echo_slow,
    "h05_baseline": _h05_baseline,
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
        help="Measurement callable name (built-in: echo_success, echo_fail, echo_nonzero, h05_baseline).",
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
    p_measure.add_argument(
        "--timeout-seconds", type=float, default=900.0,
        help="Hard wall-clock deadline for the measurement callable.",
    )
    p_measure.add_argument(
        "--fixture-path", default="",
        help="Path to a persisted NUMERIC exact-token fixture JSON. "
             "Empty = back-compat with DEFAULT_FIXTURE; real H0.5 "
             "measurements MUST pass an explicit path so the audit "
             "records the fixture's SHA.",
    )
    p_measure.set_defaults(func=cmd_measure)
    args = parser.parse_args(argv)

    if args.cmd == "measure":
        return cmd_measure(args)
    parser.error(f"unknown subcommand: {args.cmd}")  # noqa: SLF001 - argparse
    return 2


def cmd_measure(args: argparse.Namespace) -> int:
    measurement_name = args.measurement
    if measurement_name not in _BUILTIN_MEASUREMENTS:
        print(json.dumps({
            "error": "unknown measurement",
            "measurement": measurement_name,
            "available": list(_BUILTIN_MEASUREMENTS),
        }))
        return 2
    # H0.6.2: fixture path may propagate to the child subprocess via env.
    if args.fixture_path:
        os.environ["HYPERLOOM_FIXTURE_PATH"] = args.fixture_path

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
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps({
        "outcome": audit.outcome.value,
        "reason": audit.reason,
        "candidate_pid": audit.candidate_pid,
        "candidate_pgid": audit.candidate_pgid,
        "candidate_exit": audit.candidate_exit,
        "timed_out": audit.timed_out,
    }))
    return 0 if audit.outcome == Outcome.PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())