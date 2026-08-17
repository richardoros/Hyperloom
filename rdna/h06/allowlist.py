"""Allowlisted service stop / start for H0.6.

H0.6 NEVER does ``pkill llama-server``. The orchestrator may only:

  * stop systemd services explicitly listed in ``allowlist`` (default:
    the production ``qwen38-turboquant.service`` is NOT in the
    allowlist — touching it is a trust violation; the operator may
    add it later via config),
  * start those services back when restoring.

The orchestrator records every stop / start call with timestamps so
the lifecycle audit artifact is unambiguous.

If a foreign PID on the lab GPU is NOT in the allowlist, the
orchestrator reports BLOCKED (operator must kill it manually or add it
to the allowlist with an explicit comment).
"""

from __future__ import annotations

import dataclasses
import subprocess
import time
from typing import Optional


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclasses.dataclass(frozen=True)
class ServiceEvent:
    service: str
    action: str  # "stop" | "start"
    started_utc: str
    finished_utc: Optional[str]
    returncode: Optional[int]


def _systemctl(*args: list[str], timeout: float = 60.0) -> tuple[int, str]:
    try:
        out = subprocess.run(
            ["systemctl", *args],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        return out.returncode, (out.stdout + out.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, f"{exc!r}"


def stop_service(name: str, *, timeout: float = 60.0) -> ServiceEvent:
    started = utc_now()
    rc, _ = _systemctl(["stop", name], timeout=timeout)
    return ServiceEvent(
        service=name, action="stop",
        started_utc=started, finished_utc=utc_now(), returncode=rc,
    )


def start_service(name: str, *, timeout: float = 60.0) -> ServiceEvent:
    started = utc_now()
    rc, _ = _systemctl(["start", name], timeout=timeout)
    return ServiceEvent(
        service=name, action="start",
        started_utc=started, finished_utc=utc_now(), returncode=rc,
    )


def is_active(name: str, *, timeout: float = 5.0) -> bool:
    rc, _ = _systemctl(["is-active", name], timeout=timeout)
    return rc == 0


def DEFAULT_ALLOWLIST() -> tuple[str, ...]:
    """Services the orchestrator may stop to free the XTX.

    Empty by default: stopping a service requires explicit operator
    acknowledgement (this tuple is the allowlist; the operator
    populates it deliberately).

    Note: ``qwen38-turboquant.service`` is intentionally NOT in the
    default allowlist. Stopping production is a trust violation; the
    operator must add it explicitly with a one-line comment if they
    intend to drain it for an exclusive-XTX window.
    """
    return ()