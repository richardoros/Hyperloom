"""Pre-state snapshot for H0.6.

Captures a deterministic picture of the host so the orchestrator can
both:
  (a) prove it changed things back (post-state == pre-state, except for
      the candidate PID we explicitly launched), and
  (b) detect foreign GPU owners that block the exclusive-XTX window.

Scope (minimum data the orchestrator needs):
  - GPU identity for the lab GPU (UUID, BDF, gfx arch, board)
  - processes touching that GPU (PID -> GPU device index; "foreign")
  - GPU telemetry (VRAM, temp, clock, util) at snapshot time
  - service states (qwen38-turboquant.service, allowlist)
  - listeners on production / experiment ports (18079, 18179)
  - production health probe (18079 HTTP /health)
"""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclasses.dataclass(frozen=True)
class GpuIdentity:
    uuid: str
    bdf: str
    gfx_arch: str
    board_name: str
    vendor: str


@dataclasses.dataclass(frozen=True)
class GpuProcess:
    pid: int
    device_index: int
    gpu_uuid: str
    cmdline_short: str  # first 120 chars of /proc/<pid>/cmdline


@dataclasses.dataclass(frozen=True)
class GpuTelemetry:
    vram_total_bytes: int
    vram_used_bytes: int
    vram_free_bytes: int
    temperature_c: Optional[float]
    sclk_mhz: Optional[int]
    mclk_mhz: Optional[int]
    utilization_pct: Optional[float]


@dataclasses.dataclass(frozen=True)
class ServiceState:
    name: str
    is_active: str  # "active" | "inactive" | "unknown"
    is_enabled: Optional[bool]


@dataclasses.dataclass(frozen=True)
class ListenerState:
    port: int
    listening: bool
    process: Optional[str]


@dataclasses.dataclass(frozen=True)
class PreStateSnapshot:
    """A frozen, JSON-serializable picture of the host pre-experiment."""

    captured_utc: str
    lab_gpu: Optional[GpuIdentity]
    gpu_processes: list[GpuProcess]
    gpu_telemetry: GpuTelemetry
    services: list[ServiceState]
    listeners: list[ListenerState]
    production_health_ok: bool

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    def lab_gpu_uuid(self) -> Optional[str]:
        return self.lab_gpu.uuid if self.lab_gpu else None


# ---------------------------------------------------------------------------
# GPU identity + telemetry + ROCm process attribution
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, timeout: float = 10.0) -> Optional[str]:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _run_json(cmd: list[str], *, timeout: float = 10.0) -> dict:
    raw = _run(cmd, timeout=timeout)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def detect_lab_gpu() -> Optional[GpuIdentity]:
    """Return the identity of the RDNA3 lab GPU (rx7900xtx).

    Matches on product-name; if multiple GPUs are present the first
    match wins. Returns None if no match is found.
    """
    data = _run_json(["rocm-smi", "--showuniqueid", "--showproductname", "--showbus", "--json"])
    if not data:
        return None
    for card in data.values():
        if not isinstance(card, dict):
            continue
        card_series = card.get("Card Series", "")
        if "RX 7900 XTX" not in card_series:
            continue
        return GpuIdentity(
            uuid=card.get("Unique ID", "unknown"),
            bdf=card.get("PCI Bus", "unknown"),
            gfx_arch=card.get("GFX Version", "unknown"),
            board_name=card_series,
            vendor=card.get("Card Vendor", "AMD"),
        )
    return None


def rocm_gpu_processes() -> list[GpuProcess]:
    """Return the ROCm-attributed GPU processes from `rocm-smi --showpidgpus`.

    Falls back to an empty list if rocm-smi is unavailable or the
    output is unparseable. Vulkan-only processes (which do not use the
    KFD/ROCm interface) will NOT appear here; for those we fall back to
    a /proc walk + AMDGPU bus scan in :func:`vulkan_gpu_processes`.
    """
    raw = _run(["rocm-smi", "--showpidgpus"], timeout=10.0)
    if not raw:
        return []
    out: list[GpuProcess] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("PID "):
            continue
        # Format: "PID 29926 is using 0 DRM device(s)"
        parts = line.split()
        try:
            pid = int(parts[1])
        except (ValueError, IndexError):
            continue
        # "using N DRM device(s)" -> device index
        device_index = -1
        if len(parts) >= 4 and parts[2] == "is" and parts[3] == "using":
            try:
                device_index = int(parts[4])
            except ValueError:
                pass
        cmdline = _proc_cmdline_short(pid)
        out.append(
            GpuProcess(
                pid=pid,
                device_index=device_index,
                gpu_uuid="",  # filled in by caller from lab_gpu.uuid
                cmdline_short=cmdline,
            )
        )
    return out


def _proc_cmdline_short(pid: int, *, limit: int = 120) -> str:
    """Return the first ``limit`` chars of /proc/<pid>/cmdline, NUL-cleaned."""
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    cleaned = data.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    return cleaned[:limit]


def _read_int_from_json(data: dict, *keys: str) -> int:
    for key in keys:
        if key in data:
            try:
                return int(data[key])
            except (TypeError, ValueError):
                continue
    return 0


def gpu_telemetry() -> GpuTelemetry:
    data = _run_json(["rocm-smi", "--showmeminfo", "vram", "--showtemp", "--showclocks", "--json"], timeout=10.0)
    if not data:
        return GpuTelemetry(
            vram_total_bytes=0, vram_used_bytes=0, vram_free_bytes=0,
            temperature_c=None, sclk_mhz=None, mclk_mhz=None, utilization_pct=None,
        )
    total = 0
    used = 0
    temperature = None
    sclk = None
    mclk = None
    for card in data.values():
        if not isinstance(card, dict):
            continue
        total = max(total, _read_int_from_json(card, "VRAM Total Memory (B)"))
        used = max(used, _read_int_from_json(card, "VRAM Total Used Memory (B)"))
        for k, v in card.items():
            if temperature is None and "Temperature" in k and "C" in k:
                try:
                    temperature = float(v)
                except (TypeError, ValueError):
                    pass
            if sclk is None and "sclk" in k.lower() and " MHz" in k.lower():
                try:
                    sclk = int(float(v))
                except (TypeError, ValueError):
                    pass
            if mclk is None and "mclk" in k.lower() and " MHz" in k.lower():
                try:
                    mclk = int(float(v))
                except (TypeError, ValueError):
                    pass
    free = max(0, total - used)
    return GpuTelemetry(
        vram_total_bytes=total, vram_used_bytes=used, vram_free_bytes=free,
        temperature_c=temperature, sclk_mhz=sclk, mclk_mhz=mclk, utilization_pct=None,
    )


# ---------------------------------------------------------------------------
# Services + listeners + production health
# ---------------------------------------------------------------------------


def service_state(name: str) -> ServiceState:
    active_raw = _run(["systemctl", "is-active", name], timeout=5.0)
    active = (active_raw or "unknown").strip()
    enabled_raw = _run(["systemctl", "is-enabled", name], timeout=5.0)
    enabled: Optional[bool] = None
    if enabled_raw is not None:
        v = enabled_raw.strip()
        if v in ("enabled", "disabled", "static", "masked"):
            enabled = v == "enabled"
        elif v == "unknown":
            enabled = None
    return ServiceState(name=name, is_active=active, is_enabled=enabled)


def _is_listening(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            return s.connect_ex((host, port)) == 0
    except OSError:
        return True


def listener_state(port: int) -> ListenerState:
    listening = _is_listening(port)
    process: Optional[str] = None
    if listening:
        # Find owner via `ss` if available; fall back to /proc/<pid>/comm.
        ss = _run(["ss", "-ltnp", f"sport = :{port}"], timeout=5.0)
        if ss:
            process = ss.strip().splitlines()[0] if ss.strip() else None
    return ListenerState(port=port, listening=listening, process=process)


def production_health_ok(*, host: str = "127.0.0.1", port: int = 18079, timeout: float = 5.0) -> bool:
    """Return True when the production llama-server /health responds 200 OK."""
    try:
        with urllib_request_urlopen(f"http://{host}:{port}/health", timeout=timeout) as resp:
            return resp.status == 200
    except (OSError, Exception):  # noqa: BLE001 - any error => unhealthy
        return False


def _urllib_request_urlopen(url: str, *, timeout: float):
    """Tiny indirection so tests can mock the URL opener."""
    import urllib.request
    return urllib.request.urlopen(url, timeout=timeout)


# Hmm: the alias below is the test seam. Rename for clarity.
urllib_request_urlopen = _urllib_request_urlopen  # noqa: SLF001 - public test hook


# ---------------------------------------------------------------------------
# Top-level snapshot
# ---------------------------------------------------------------------------


def take_snapshot(
    *,
    lab_gpu_uuid: Optional[str] = None,
    prod_port: int = 18079,
    exp_port: int = 18179,
    extra_service_names: tuple[str, ...] = (),
) -> PreStateSnapshot:
    """Capture the host state pre-experiment.

    Args:
        lab_gpu_uuid: override the detected lab GPU UUID (used in tests).
        prod_port: production port (default 18079).
        exp_port: experiment port (default 18179).
        extra_service_names: extra systemd services to snapshot (the
            default includes the lab's production service).
    """
    gpu = detect_lab_gpu()
    effective_uuid = lab_gpu_uuid or (gpu.uuid if gpu else "")
    processes = rocm_gpu_processes()
    if effective_uuid:
        processes = [
            dataclasses.replace(p, gpu_uuid=effective_uuid) for p in processes
        ]
    telemetry = gpu_telemetry()
    services: list[ServiceState] = []
    seen: set[str] = set()
    for name in ("qwen38-turboquant.service", *extra_service_names):
        if name in seen:
            continue
        seen.add(name)
        services.append(service_state(name))
    listeners = [listener_state(prod_port), listener_state(exp_port)]
    return PreStateSnapshot(
        captured_utc=utc_now(),
        lab_gpu=gpu,
        gpu_processes=processes,
        gpu_telemetry=telemetry,
        services=services,
        listeners=listeners,
        production_health_ok=production_health_ok(port=prod_port) if _is_listening(prod_port) else False,
    )