"""Load / contention gate for H0.5 evaluator.

Three checks, fail-closed:

1. Production is untouched:
   - ``qwen38-turboquant.service`` is not active.
   - 18079 (the production port) is not listening.
2. The experiment lane is empty:
   - The chosen ``--port`` is not in use.
   - No other ``llama-server`` is bound to the GPU's VRAM.
3. The GPU has enough headroom to load the GGUF at the requested layers.
   Threshold is the GGUF size on disk as a conservative lower bound.

The gate runs BEFORE the binary launches. A failed gate means no
measurement is taken and the run is recorded as BLOCKED in the DB.
"""

from __future__ import annotations

import dataclasses
import json
import socket
import subprocess
from pathlib import Path

from .identity import IdentityBlock, detect_gpu_type


@dataclasses.dataclass(frozen=True)
class GateReport:
    """Outcome of one gate check pass. Always populated for traceability."""

    ok: bool
    reason: str
    qwen38_state: str  # "active" | "inactive" | "unknown"
    prod_listening: bool
    exp_port_listening: bool
    foreign_llama_servers: list[str]  # cmds of foreign llama-servers on the GPU
    vram_total_bytes: int
    vram_used_bytes: int
    vram_free_bytes: int
    headroom_bytes: int
    headroom_fraction: float
    required_bytes: int

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)


def _qwen38_state() -> str:
    """Return systemd state string for the production service."""
    try:
        out = subprocess.run(
            ["systemctl", "is-active", "qwen38-turboquant.service"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _is_listening(port: int, host: str = "127.0.0.1") -> bool:
    """TCP connect-ex probe. Returns True if something is listening."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            return s.connect_ex((host, port)) == 0
    except OSError:
        return True


def _foreign_llama_servers(gpu_type: str) -> list[str]:
    """Return the cmdlines of foreign llama-server processes on this GPU.

    The evaluator considers any other ``llama-server`` invocation with a
    different ``--port`` foreign. The current run's PID is excluded by the
    caller via :func:`gate` (it has not launched yet at gate time, so this
    is empty in practice for the gate).
    """
    out: list[str] = []
    try:
        ps = subprocess.run(
            ["pgrep", "-af", "llama-server"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return out
    for line in ps.stdout.splitlines():
        # pgrep -af prints "<pid> <cmdline>"
        if "/llama-server" not in line:
            continue
        out.append(line.strip())
    return out


def _vram_rocm_smi() -> tuple[int, int]:
    """Return (used_bytes, total_bytes) on the active GPU via rocm-smi."""
    try:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0, 0
    if out.returncode != 0 or not out.stdout.strip():
        return 0, 0
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return 0, 0
    for card in data.values():
        if isinstance(card, dict) and "VRAM Total Memory (B)" in card:
            try:
                total = int(card["VRAM Total Memory (B)"])
                used = int(card.get("VRAM Total Used Memory (B)", "0"))
                return used, total
            except (ValueError, TypeError):
                continue
    return 0, 0


def gate(
    *,
    identity: IdentityBlock,
    exp_port: int,
    required_bytes: int,
    prod_port: int = 18079,
    prod_service: str = "qwen38-turboquant.service",
    min_headroom_fraction: float = 0.10,
) -> GateReport:
    """Run the fail-closed gate.

    Args:
        identity:        The identity block (used for context only).
        exp_port:        Experiment port the evaluator intends to bind.
        required_bytes:  Minimum VRAM needed (GGUF size is a safe lower bound).
        prod_port:       Production port (default 18079).
        prod_service:    Production systemd unit (default qwen38-turboquant.service).
        min_headroom_fraction: Minimum free fraction required even if
            ``required_bytes`` is met (e.g. we want at least 10 % free after
            load for the kernel/cache + temperature headroom).
    """
    used, total = _vram_rocm_smi()
    free = max(0, total - used)
    headroom_bytes = max(0, free - required_bytes)
    headroom_fraction = (free / total) if total else 0.0

    qwen38 = _qwen38_state()
    prod_listening = _is_listening(prod_port)
    exp_listening = _is_listening(exp_port)
    foreign = _foreign_llama_servers(identity.gpu_type)

    ok = True
    reasons: list[str] = []

    if qwen38 == "active":
        ok = False
        reasons.append(f"production service {prod_service} is active; do not stop it in H0.5")
    if prod_listening:
        ok = False
        reasons.append(f"production port {prod_port} is listening; H0.5 does not touch production")
    if exp_listening:
        ok = False
        reasons.append(f"experiment port {exp_port} already in use")
    if foreign:
        ok = False
        reasons.append(
            f"{len(foreign)} foreign llama-server process(es) on this GPU; not killed"
        )
    if required_bytes > free:
        ok = False
        reasons.append(
            f"VRAM headroom insufficient: need {required_bytes} bytes, have {free}"
        )
    elif headroom_fraction < min_headroom_fraction:
        ok = False
        reasons.append(
            f"VRAM free fraction {headroom_fraction:.3f} < required {min_headroom_fraction:.3f}"
        )

    return GateReport(
        ok=ok,
        reason="; ".join(reasons) if reasons else "ok",
        qwen38_state=qwen38,
        prod_listening=prod_listening,
        exp_port_listening=exp_listening,
        foreign_llama_servers=foreign,
        vram_total_bytes=total,
        vram_used_bytes=used,
        vram_free_bytes=free,
        headroom_bytes=headroom_bytes,
        headroom_fraction=headroom_fraction,
        required_bytes=required_bytes,
    )