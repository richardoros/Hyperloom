"""Build-from-SHA provenance for H0.5.

The evaluator owns the candidate build. Building into the production
build directory is forbidden: H0.5 must not mutate
``src/llama.cpp-turboquant/build/``. Instead the candidate is built into
``rdna/h05_results/builds/<source-sha>/<build-config-hash>/`` (or
``/tmp/hyperloom-evaluator-build-<short-id>``), and the resulting binary
SHA-256 is recorded so the build identity is provably linked to the
source SHA.

The contract:

* The user supplies ``--build-config`` (a free-form text string that
  influences the cmake invocation, e.g. cmake args or ccache state).
* ``build_config_hash`` is the SHA-256 of the canonicalized
  ``build_config`` string.
* The evaluator runs ``cmake -B build && cmake --build build`` in an
  isolated checkout directory (typically a clone / worktree of the
  source repo). The resulting ``llama-server`` binary is SHA-256-ed and
  compared to the observed identity block's binary_sha256.
* If they match, the build is provenance-verified: the recorded binary
  came from this source SHA with this build config.
* If they don't match (the recorded binary was built elsewhere), the
  evaluator emits BLOCKED with reason ``build provenance mismatch``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from .identity import _sha256_file, _sha256_text


@dataclasses.dataclass(frozen=True)
class BuildProvenance:
    """Proof that a binary was built from a specific source SHA + config."""

    source_repo: str
    source_sha: Optional[str]
    build_dir: str
    build_config: str
    build_config_hash: str
    binary_path: str
    binary_sha256: str
    build_log_path: Optional[str]
    build_log_sha256: Optional[str]
    built_at_utc: str

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def hash_build_config(build_config: str) -> str:
    """SHA-256 of the canonicalized build config string."""
    return _sha256_text(" ".join(build_config.split()))


def _run_capture(cmd: list[str], cwd: Path, timeout: float) -> tuple[int, str, str]:
    try:
        out = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return out.returncode, out.stdout, out.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, "", f"{exc!r}"


def _git_run(args: list[str], cwd: Path, timeout: float = 30.0) -> tuple[int, str, str]:
    try:
        out = subprocess.run(
            args, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, check=False
        )
        return out.returncode, out.stdout.strip(), out.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, "", f"{exc!r}"


def _resolve_clean_worktree(
    source_repo: Path,
    source_sha: Optional[str],
    build_root: Path,
) -> Path:
    """Create a detached, CLEAN worktree at exactly ``source_sha``.

    Raises on:
      * missing source_sha argument (refuse to build "whatever HEAD is")
      * dirty working tree in the source checkout
      * git worktree creation failure
      * HEAD != source_sha verification failure

    Returns the worktree path; the caller MUST configure/build from
    this path, not the original checkout.
    """
    if not source_sha:
        raise RuntimeError(
            "build_provenance requires source_sha; refusing to build "
            "whatever the working tree happens to be"
        )
    if shutil.which("git") is None:
        raise RuntimeError("git is required for build provenance")
    rc, _, err = _git_run(["git", "rev-parse", "--is-inside-work-tree"], cwd=source_repo)
    if rc != 0:
        raise RuntimeError(f"{source_repo} is not a git work tree: {err}")
    # Working tree must be CLEAN: no uncommitted, no staged. Otherwise
    # we are not really building source_sha.
    rc, _, _ = _git_run(["git", "diff", "--quiet"], cwd=source_repo)
    if rc != 0:
        raise RuntimeError(
            f"{source_repo} has uncommitted changes; refusing to build (clean the tree first)"
        )
    rc, _, _ = _git_run(["git", "diff", "--cached", "--quiet"], cwd=source_repo)
    if rc != 0:
        raise RuntimeError(
            f"{source_repo} has staged changes; refusing to build (clean the tree first)"
        )
    # Verify the source_repo HEAD is exactly source_sha.
    rc, head, _ = _git_run(["git", "rev-parse", "HEAD"], cwd=source_repo)
    if rc != 0 or head != source_sha:
        raise RuntimeError(
            f"{source_repo} HEAD is {head}, requested source_sha is {source_sha}"
        )
    # Create a detached worktree at source_sha. The worktree lives
    # under build_root/ so production checkouts are not mutated.
    worktree_path = build_root / "worktrees" / source_sha
    rc, _, err = _git_run(
        ["git", "worktree", "add", "--detach", str(worktree_path), source_sha],
        cwd=source_repo,
    )
    if rc != 0:
        raise RuntimeError(f"git worktree add failed: {err}")
    # Verify HEAD in the worktree is exactly source_sha.
    rc, wt_head, _ = _git_run(["git", "rev-parse", "HEAD"], cwd=worktree_path)
    if rc != 0 or wt_head != source_sha:
        raise RuntimeError(
            f"worktree HEAD is {wt_head}, expected {source_sha}; refusing to build"
        )
    return worktree_path


def build_candidate(
    *,
    source_repo: str,
    source_sha: Optional[str],
    build_config: str,
    build_root: Path,
    binary_name: str = "llama-server",
    timeout_seconds: int = 3600,
) -> BuildProvenance:
    """Build llama.cpp from ``source_repo`` into an isolated directory.

    Layout:

        build_root/worktrees/<source-sha>/                  (detached worktree)
        build_root/<source-sha>/<build_config_hash>/build/bin/<binary_name>
        build_root/<source-sha>/<build_config_hash>/build.log

    Provenance contract:
      * The build is performed in an evaluator-owned detached worktree
        at exactly ``source_sha``; the production checkout is never
        modified.
      * ``source_repo`` HEAD must equal ``source_sha`` AND the working
        tree must be clean (no uncommitted, no staged). Refuses to
        build otherwise — provenance would be ambiguous.
      * ``build_config`` is parsed via ``shlex.split`` so quoted strings
        and escapes are honoured.
      * ``build_config_hash`` is the SHA-256 of the canonicalized config
        so two builds with semantically-equal configs hash equal.

    Raises ``RuntimeError`` on git / cmake / build failure.
    """
    source_repo = Path(source_repo).resolve()
    if not source_repo.is_dir():
        raise RuntimeError(f"source repo not found: {source_repo}")
    if not (source_repo / "CMakeLists.txt").is_file():
        raise RuntimeError(f"not a llama.cpp checkout (no CMakeLists.txt): {source_repo}")

    build_root.mkdir(parents=True, exist_ok=True)
    worktree = _resolve_clean_worktree(source_repo, source_sha, build_root)
    build_config_hash = hash_build_config(build_config)
    target_dir = build_root / source_sha / build_config_hash
    target_dir.mkdir(parents=True, exist_ok=True)
    build_dir = target_dir / "build"
    binary_path = build_dir / "bin" / binary_name
    log_path = target_dir / "build.log"

    cmake_args = ["cmake", "-B", str(build_dir), "-S", str(worktree)]
    if build_config:
        cmake_args.extend(shlex.split(build_config))

    # Phase 1: cmake configure (from the worktree, NEVER the production tree).
    rc, cmake_stdout, cmake_stderr = _run_capture(
        cmake_args, cwd=worktree, timeout=600.0,
    )
    if rc != 0:
        raise RuntimeError(f"cmake configure failed (rc={rc}): {cmake_stderr[-1000:]}")

    # Phase 2: cmake build.
    rc, build_stdout, build_stderr = _run_capture(
        ["cmake", "--build", str(build_dir), "--parallel"],
        cwd=worktree,
        timeout=float(timeout_seconds),
    )
    log = (
        "### worktree (detached) ###\n"
        f"worktree: {worktree}\n"
        f"source_sha: {source_sha}\n"
        f"build_config_hash: {build_config_hash}\n\n"
        "### cmake configure ###\n"
        f"$ {' '.join(cmake_args)}\n"
        f"{cmake_stdout}\n{cmake_stderr}\n\n"
        "### cmake build ###\n"
        f"$ cmake --build {build_dir} --parallel\n"
        f"{build_stdout}\n{build_stderr}\n"
    )
    log_path.write_text(log, encoding="utf-8")
    if rc != 0:
        raise RuntimeError(f"cmake build failed (rc={rc}): see {log_path}")

    if not binary_path.is_file():
        raise RuntimeError(f"build succeeded but binary not found: {binary_path}")

    binary_sha = _sha256_file(binary_path)
    log_sha = _sha256_file(log_path)

    return BuildProvenance(
        source_repo=str(source_repo),
        source_sha=source_sha,
        build_dir=str(build_dir),
        build_config=build_config,
        build_config_hash=build_config_hash,
        binary_path=str(binary_path),
        binary_sha256=binary_sha,
        build_log_path=str(log_path),
        build_log_sha256=log_sha,
        built_at_utc=utc_now(),
    )


def verify_provenance(
    *,
    expected_binary_sha256: str,
    source_repo: str,
    source_sha: Optional[str],
    build_config: str,
    build_root: Path,
    binary_name: str = "llama-server",
    timeout_seconds: int = 3600,
) -> BuildProvenance:
    """Build from source and verify the resulting binary matches the
    expected SHA-256. Returns the BuildProvenance on success; raises on
    mismatch or build failure.
    """
    provenance = build_candidate(
        source_repo=source_repo,
        source_sha=source_sha,
        build_config=build_config,
        build_root=build_root,
        binary_name=binary_name,
        timeout_seconds=timeout_seconds,
    )
    if provenance.binary_sha256 != expected_binary_sha256:
        raise RuntimeError(
            f"build provenance mismatch: built binary SHA={provenance.binary_sha256} "
            f"!= expected SHA={expected_binary_sha256} "
            f"(build_config={provenance.build_config_hash[:12]})"
        )
    return provenance