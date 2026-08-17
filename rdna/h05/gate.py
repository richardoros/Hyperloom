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
    # P1.6 extra load signals.
    cpu_load_per_core: float | None
    ram_free_bytes: int | None
    swap_used_bytes: int | None
    gpu_temp_c: float | None
    gpu_clock_mhz: int | None
    contended_build: list[str]

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


def _process_uses_gpu(cmdline: str) -> bool:
    """Heuristic: does this llama-server actually use GPU layers?

    CPU-only embedding servers (``-ngl 0``, ``--embedding``, ``--no-op-offload``)
    are excluded from the foreign-on-GPU list because they do not hold
    VRAM and do not contend for the experiment lane. The evaluator only
    cares about processes that are actually on the GPU.
    """
    if "llama-server" not in cmdline:
        return False
    tokens = cmdline.split()
    # CPU-only markers (any one suffices).
    if "--embedding" in tokens:
        return False
    if "--no-op-offload" in tokens:
        return False
    if "--no-gpu" in tokens:
        return False
    # Explicit GPU device flag (ROCm: -dev / --device; Vulkan: --device or -dev).
    if "-dev" in tokens or "--device" in tokens:
        return True
    # `-ngl N` (with N > 0) OR `--n-gpu-layers N` implies GPU. Both are
    # token-pairs in llama.cpp's CLI; we look at the arg + next token.
    for i, tok in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if tok in ("-ngl", "--n-gpu-layers") and nxt is not None:
            try:
                if int(nxt) > 0:
                    return True
            except ValueError:
                pass
    # Default: assume CPU if no GPU-positive marker is found.
    return False


def _foreign_llama_servers(
    *,
    gpu_type: str,
    foreign_owners_on_gpu: set[int] | None = None,
    gpu_uuid: str | None = None,
) -> list[str]:
    """Return cmdlines of foreign llama-server processes that hold VRAM.

    CPU-only embedding servers (and other llama-server invocations that
    did not request GPU layers) are excluded: they don't contend for the
    GPU and must not block the XTX laboratory merely because their
    executable is named ``llama-server``.

    Args:
        gpu_type: passed for context (logging); the gate does not yet
            have a per-process GPU UUID map so it cannot route processes
            to specific GPU types. The caller can pass ``gpu_uuid`` in a
            future revision.
        foreign_owners_on_gpu: PIDs the caller has already attributed to
            this GPU (e.g. via ``rocm-smi --showpidgpus``); accepted as
            legitimate users.
        gpu_uuid: this GPU's UUID (future use; not yet acted on).
    """
    out: list[str] = []
    foreign_owners_on_gpu = foreign_owners_on_gpu or set()
    _ = gpu_type, gpu_uuid  # accepted for API compatibility; not yet used
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
        if "/llama-server" not in line:
            continue
        # pgrep -af prints "<pid> <cmdline>"; pid is the first token.
        parts = line.strip().split(None, 1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmdline = parts[1] if len(parts) > 1 else ""
        if pid in foreign_owners_on_gpu:
            continue  # legitimately owned by us
        if not _process_uses_gpu(cmdline):
            continue  # CPU-only; not on this GPU
        out.append(line.strip())
    return out


# ---------------------------------------------------------------------------
# Extra load signals (P1.6)
# ---------------------------------------------------------------------------


def _cpu_load_per_core() -> float | None:
    """1-minute load average per CPU core.

    Returns ``None`` when /proc/loadavg is unreadable.
    """
    try:
        text = Path("/proc/loadavg").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        load1 = float(text.split()[0])
    except (ValueError, IndexError):
        return None
    try:
        ncpu = len(Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").split("processor"))
    except OSError:
        ncpu = 1
    return load1 / max(1, ncpu)


def _ram_free_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _swap_used_bytes() -> int | None:
    try:
        total = used = 0
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("SwapTotal:"):
                total = int(line.split()[1]) * 1024
            elif line.startswith("SwapFree:"):
                used = total - int(line.split()[1]) * 1024
        return used
    except (OSError, ValueError):
        return None


def _gpu_temp_c(gpu_type: str) -> float | None:
    """Return current GPU temperature in °C, or None."""
    try:
        out = subprocess.run(
            ["rocm-smi", "--showtemp", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None
    for card in data.values():
        if isinstance(card, dict):
            for k, v in card.items():
                if "Temperature" in k and "C" in k:
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        continue
        elif "Temperature (Sensor edge) (C)" in card:
            try:
                return float(card["Temperature (Sensor edge) (C)"])
            except (TypeError, ValueError):
                continue
    return None


def _gpu_clock_mhz(gpu_type: str) -> int | None:
    """Return current GPU sclk in MHz, or None."""
    try:
        out = subprocess.run(
            ["rocm-smi", "--showclocks", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None
    for card in data.values():
        if isinstance(card, dict):
            for k, v in card.items():
                if "sclk" in k.lower() and " MHz" in k.lower():
                    try:
                        return int(float(v))
                    except (TypeError, ValueError):
                        continue
    return None


def _build_contention() -> list[str]:
    """Detect build/Celery contention: heavy CPU users in build dirs OR celery workers.

    Returns a list of short reasons for the gate's failure message.
    """
    reasons: list[str] = []
    try:
        ps = subprocess.run(
            ["ps", "-eo", "pid,pcpu,comm,args", "--no-headers"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return reasons
    heavy_build = 0
    celery = 0
    for line in ps.stdout.splitlines():
        if "/build" in line and "/compile" in line:
            heavy_build += 1
        if "celery" in line and "worker" in line:
            celery += 1
    if heavy_build > 0:
        reasons.append(f"{heavy_build} active compile process(es)")
    if celery > 0:
        reasons.append(f"{celery} celery worker(s) running")
    return reasons


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
    foreign_owners_on_gpu: set[int] | None = None,
    gpu_uuid: str | None = None,
    max_cpu_load_per_core: float = 0.85,
    min_free_ram_bytes: int = 4 * 1024 * 1024 * 1024,
    max_swap_used_bytes: int = 64 * 1024 * 1024,
    max_gpu_temp_c: float = 90.0,
    min_gpu_clock_mhz: int = 500,
) -> GateReport:
    """Run the fail-closed gate.

    Args:
        identity:        The identity block (used for context only).
        exp_port:        Experiment port the evaluator intends to bind.
        required_bytes:  Minimum VRAM needed (GGUF size is a safe lower bound).
        prod_port:       Production port (default 18079).
        prod_service:    Production systemd unit (default qwen38-turboquant.service).
        min_headroom_fraction: Minimum free fraction required AFTER load
            for the kernel/cache + temperature headroom. ``headroom_fraction``
            is ``(free - required) / total`` (post-load free fraction), not
            ``free / total`` (pre-load free fraction).
        foreign_owners_on_gpu: PIDs the caller has already attributed to
            this GPU (e.g. via ``rocm-smi --showpidgpus``); they are
            accepted as legitimate users and NOT counted as foreign.
        gpu_uuid:        This GPU's UUID (used to attribute foreign procs).
        max_cpu_load_per_core: Per-core load above which the host is
            considered build/Celery-contended and BLOCKED.
        min_free_ram_bytes: Free RAM floor for system-side headroom.
        max_swap_used_bytes: Swap activity ceiling.
        max_gpu_temp_c:   GPU temperature ceiling (clocks throttle above this).
        min_gpu_clock_mhz: Minimum GPU clock (lower = power-saving = noise).
    """
    used, total = _vram_rocm_smi()
    free = max(0, total - used)
    headroom_bytes = max(0, free - required_bytes)
    # P1.7: headroom_fraction is POST-load free fraction, not pre-load.
    headroom_fraction = (headroom_bytes / total) if total else 0.0

    qwen38 = _qwen38_state()
    prod_listening = _is_listening(prod_port)
    exp_listening = _is_listening(exp_port)
    foreign = _foreign_llama_servers(
        gpu_type=identity.gpu_type,
        foreign_owners_on_gpu=foreign_owners_on_gpu,
        gpu_uuid=gpu_uuid,
    )
    cpu_load = _cpu_load_per_core()
    ram_free = _ram_free_bytes()
    swap_used = _swap_used_bytes()
    gpu_temp = _gpu_temp_c(identity.gpu_type)
    gpu_clock = _gpu_clock_mhz(identity.gpu_type)
    contended_build = _build_contention()

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
            f"VRAM post-load free fraction {headroom_fraction:.3f} < required {min_headroom_fraction:.3f}"
        )
    if cpu_load is not None and cpu_load > max_cpu_load_per_core:
        ok = False
        reasons.append(
            f"CPU load per-core {cpu_load:.2f} > {max_cpu_load_per_core:.2f} (build/Celery contention)"
        )
    if ram_free is not None and ram_free < min_free_ram_bytes:
        ok = False
        reasons.append(
            f"RAM free {ram_free // (1024*1024)} MiB < required {min_free_ram_bytes // (1024*1024)} MiB"
        )
    if swap_used is not None and swap_used > max_swap_used_bytes:
        ok = False
        reasons.append(
            f"swap activity {swap_used // (1024*1024)} MiB > {max_swap_used_bytes // (1024*1024)} MiB"
        )
    if gpu_temp is not None and gpu_temp > max_gpu_temp_c:
        ok = False
        reasons.append(
            f"GPU temperature {gpu_temp:.0f}°C > {max_gpu_temp_c:.0f}°C (thermal throttling likely)"
        )
    if gpu_clock is not None and gpu_clock < min_gpu_clock_mhz:
        ok = False
        reasons.append(
            f"GPU clock {gpu_clock} MHz < {min_gpu_clock_mhz} MHz (governor=power-saving)"
        )
    if contended_build:
        ok = False
        reasons.append(f"build/Celery contention detected: {', '.join(contended_build)}")

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
        cpu_load_per_core=cpu_load,
        ram_free_bytes=ram_free,
        swap_used_bytes=swap_used,
        gpu_temp_c=gpu_temp,
        gpu_clock_mhz=gpu_clock,
        contended_build=contended_build,
    )