# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pytest hooks and shared helpers for the inference_optimizer tests package."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.session.paths import make_session_dir


@pytest.fixture(autouse=True)
def _isolate_session_layout_env(monkeypatch, tmp_path_factory):
    """Drop the session-dir pin and point MULTI_NODE_STATE_FILE at a missing sentinel so tests run single-node."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", raising=False)
    mn_state_sentinel = tmp_path_factory.mktemp("mn_state") / "missing_state.json"
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(mn_state_sentinel))
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)


@pytest.fixture(autouse=True)
def _clear_kernel_request_handler_caches():
    """Clear ``lru_cache`` state on env-bound helpers between tests."""
    from hyperloom.orchestrator.kernel import request_handlers as krh

    krh._default_kernel_batch_parallel.cache_clear()


def _bootstrap_kernel_agent_env() -> None:
    """Point HYPERLOOM_KERNEL_AGENT_ROOT at the in-repo kernel-agent checkout."""
    if os.environ.get("HYPERLOOM_KERNEL_AGENT_ROOT"):
        return
    repo = Path(__file__).resolve().parents[4]
    kernel_agent = repo / "src" / "hyperloom" / "agents" / "kernel"
    if kernel_agent.is_dir():
        os.environ["HYPERLOOM_KERNEL_AGENT_ROOT"] = str(kernel_agent)


_bootstrap_kernel_agent_env()


def seed_target_analysis_marker(session_dir: Path) -> Path:
    """Write a ``no_target_gpu_configured`` marker JSON at the session dir."""
    from hyperloom.inference_optimizer.session.session_paths import target_baseline_json

    path = target_baseline_json(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "skipped",
                "reason": "no_target_gpu_configured",
                "warning": "compare_against_gpu is empty",
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    """A fresh session dir under an isolated ``USER_DATA_PATH``, seeded with the
    ``no_target_gpu_configured`` target-analysis marker.
    """
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    sd = make_session_dir()
    seed_target_analysis_marker(sd)
    return sd


def init_git_repo(
    path: Path,
    *,
    seed_file: str = "src.py",
    seed_text: str = "def f():\n    return 1\n",
) -> None:
    """Initialise a minimal git repo with one commit under ``path``.

    Seeds a single tracked file and commits it so ``git worktree add`` and
    patch application have a base commit to branch from.
    """
    path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "Hyperloom Test"
    env["GIT_AUTHOR_EMAIL"] = "hyperloom@test.local"
    env["GIT_COMMITTER_NAME"] = env["GIT_AUTHOR_NAME"]
    env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        check=True,
        capture_output=True,
        env=env,
    )
    (path / seed_file).write_text(seed_text, encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(path), "add", "."],
        check=True,
        capture_output=True,
        env=env,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "init"],
        check=True,
        capture_output=True,
        env=env,
    )


def git_commit_all(path: Path, message: str) -> None:
    """Stage everything under ``path`` and commit with a fixed non-interactive identity."""
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "Hyperloom Test"
    env["GIT_AUTHOR_EMAIL"] = "hyperloom@test.local"
    env["GIT_COMMITTER_NAME"] = env["GIT_AUTHOR_NAME"]
    env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]
    subprocess.run(
        ["git", "-C", str(path), "add", "."],
        check=True,
        capture_output=True,
        env=env,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", message],
        check=True,
        capture_output=True,
        env=env,
    )


class _BuildFakeCoordinator:
    """Minimal coordinator surface for off-loop targeted-build tests.

    Wires a real ``TaskRegistry`` + ``ResourceLockManager`` + ``SharedState``
    against a temp SQLite DB so the build pump/reaper can run without a full
    Coordinator.
    """

    def __init__(self, session_dir: Path, db) -> None:
        from hyperloom.orchestrator.bus.resource_lock import (
            ResourceLockManager,
            SqliteLeaseBackend,
        )
        from hyperloom.orchestrator.state.shared_state import SharedState
        from hyperloom.orchestrator.state.task_registry import TaskRegistry

        self.session_dir = session_dir
        self.tasks = TaskRegistry(db)
        self.locks = ResourceLockManager(SqliteLeaseBackend(db))
        self.shared_state = SharedState()


@pytest.fixture
def build_coord(tmp_path):
    """Fake coordinator backed by a temp DB for targeted-build lifecycle tests."""
    from hyperloom.orchestrator.bus.storage import SqliteConnection
    from hyperloom.orchestrator.bus.storage.schema import ensure_schema

    db = SqliteConnection(tmp_path / "coordinator.db")
    ensure_schema(db.raw)
    fc = _BuildFakeCoordinator(tmp_path, db)
    yield fc
    db.close()


@pytest.fixture
def build_lifecycle(build_coord):
    """``BuildLifecycleCollaborator`` bound to the ``build_coord`` fixture."""
    from hyperloom.orchestrator.loop.build_lifecycle import BuildLifecycleCollaborator

    return BuildLifecycleCollaborator(build_coord)


def patch_integrate_patch_allowlist(monkeypatch, tmp_path: Path) -> None:
    """Register common tmp_path framework repos for integrate_patch allowlist tests."""
    from hyperloom.orchestrator.actions.executors import integrate_patch as ip
    from hyperloom.orchestrator.framework import paths as fp

    real = fp.resolve_source_file_allowlist

    def _merged() -> tuple[str, ...]:
        extras: list[str] = []
        for name in ("fw", "repo", "framework"):
            cand = tmp_path / name
            if cand.is_dir():
                extras.append(str(cand.resolve()))
        merged = list(real())
        for extra in extras:
            if extra not in merged:
                merged.append(extra)
        return tuple(merged)

    monkeypatch.setattr(ip, "resolve_source_file_allowlist", _merged)


class _RayDoubleActorClass:
    """The ``@ray.remote`` class: ``.options(...)`` then ``.remote()`` for a handle."""

    def __init__(self, cls: type) -> None:
        self._cls = cls
        self._options: dict = {}

    def options(self, **opts):
        self._options = dict(opts)
        return self

    def remote(self, *args, **kwargs) -> "_RayDoubleActorHandle":
        return _RayDoubleActorHandle(self._cls(*args, **kwargs), self._options)


class _RayDoubleActorHandle:
    """An actor handle whose methods run in a pool sized like the real actor's.

    ``max_concurrency`` is read from the options the production code passed, not
    assumed: an actor left at Ray's single method slot gets a single-worker pool
    here too, so a method that has to reach a call already running blocks behind
    it exactly as it would on a real cluster.
    """

    def __init__(self, obj, options: dict) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self._obj = obj
        self._pool = ThreadPoolExecutor(max_workers=int(options.get("max_concurrency", 1) or 1))
        self.killed = False

    def __getattr__(self, name: str):
        from types import SimpleNamespace

        method = getattr(self._obj, name)
        return SimpleNamespace(remote=lambda *a, **kw: self._pool.submit(method, *a, **kw))


class RayDouble:
    """A ``ray`` module stand-in that runs actor methods in real threads.

    Ray is not a test dependency, and what these tests are about is what a lease
    and its actor do to each other while work is in flight. So the transport is
    the only thing faked: the actor is the real class, running real subprocesses,
    and an ``ObjectRef`` is a :class:`~concurrent.futures.Future`.
    """

    class exceptions:  # noqa: N801 — mirrors the ray.exceptions namespace
        class RayActorError(Exception):
            pass

        class RayTaskError(Exception):
            pass

        class GetTimeoutError(Exception):
            pass

    def __init__(self) -> None:
        self.killed: list = []

    def remote(self, cls: type) -> _RayDoubleActorClass:
        return _RayDoubleActorClass(cls)

    def cluster_resources(self) -> dict:
        return {"CPU": 8.0, "GPU": 8.0, "serving_slot": 1.0}

    def get(self, ref, timeout: float | None = None):
        return ref.result(timeout)

    def wait(self, refs: list, *, num_returns: int = 1, timeout: float | None = None):
        from concurrent.futures import FIRST_COMPLETED
        from concurrent.futures import wait as futures_wait

        done, not_done = futures_wait(refs, timeout=timeout, return_when=FIRST_COMPLETED)
        return list(done)[:num_returns], list(not_done)

    def kill(self, actor) -> None:
        actor.killed = True
        self.killed.append(actor)


@pytest.fixture
def serving_lease_on_a_ray_double(monkeypatch):
    """A real :class:`ServingLease` over :class:`RayDouble`, closed on teardown."""
    import sys
    from types import SimpleNamespace

    from hyperloom.orchestrator.actions.executors import _ray_backend as rb
    from hyperloom.orchestrator.actions.executors import _ray_serving as rs

    monkeypatch.setitem(sys.modules, "ray", RayDouble())
    monkeypatch.setattr(rb, "get_ray_backend", lambda: SimpleNamespace(ensure=lambda **_kw: None))
    lease = rs.ServingLease(num_gpus=1)
    try:
        yield lease
    finally:
        lease.close()
