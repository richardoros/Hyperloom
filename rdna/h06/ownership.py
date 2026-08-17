"""Real PID -> GPU attribution for H0.6.

Replaces the H0.5 heuristic (``-ngl``, ``--embedding`` etc) with a
deterministic lookup that consults the kernel:

  1. ROCm (KFD): ``rocm-smi --showpidgpus`` returns PIDs registered with
     the AMD kernel driver. The lab GPU's UUID identifies our board;
     KFD PIDs whose device index matches the lab GPU are "owned by us".

  2. Vulkan / direct amdgpu DRM: ``/proc/<pid>/maps`` exposes the FD
     of an opened DRM device. Any process with a mapping under
     ``/dev/dri/`` or ``/sys/class/drm/card*/`` is treated as a GPU
     owner. If the lab GPU's BDF matches the device in the mapping, we
     attribute ownership. This catches the Vulkan turboquant workload
     that H0.5's heuristic missed.

  3. Foreign = owned - allowed. PIDs in the operator allowlist
     (e.g. an embedding server the operator knows is OK to keep alive
     but that still uses GPU) are NOT considered foreign.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class ProcessGpuOwner:
    pid: int
    gpu_uuid: str
    backend: str  # "rocm" or "vulkan" or "drm"
    detail: str


def _rocm_pid_gpu_map() -> dict[int, int]:
    """Return {pid: device_index} from ``rocm-smi --showpidgpus``.

    Empty dict if rocm-smi is unavailable or unparseable. PIDs whose
    device_index == -1 (the "using N DRM device(s)" parse failed) are
    still in the dict; the caller can decide what to do.
    """
    try:
        out = subprocess.run(
            ["rocm-smi", "--showpidgpus"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if out.returncode != 0:
        return {}
    mapping: dict[int, int] = {}
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line.startswith("PID "):
            continue
        parts = line.split()
        try:
            pid = int(parts[1])
        except (ValueError, IndexError):
            continue
        device_index = -1
        if len(parts) >= 4 and parts[2] == "is" and parts[3] == "using":
            try:
                device_index = int(parts[4])
            except ValueError:
                pass
        mapping[pid] = device_index
    return mapping


def _drm_pid_gpu_map() -> dict[int, str]:
    """Return {pid: drm_card_path} for PIDs with an open DRM FD.

    Walk /proc/<pid>/maps and look for ``/dev/dri/`` mappings. PIDs
    without a DRM mapping are NOT in the map. The card path is e.g.
    ``/dev/dri/card0`` (the drm fd target).
    """
    out: dict[int, str] = {}
    pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    for pid in pids:
        try:
            text = Path(f"/proc/{pid}/maps").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if "/dev/dri/" not in line:
                continue
            # line format: addr perms offset dev inode pathname
            parts = line.split(None, 5)
            if len(parts) < 6:
                continue
            path = parts[5]
            # Path may be /dev/dri/card0 or /dev/dri/renderD128.
            if not (path.startswith("/dev/dri/card") or path.startswith("/dev/dri/renderD")):
                continue
            out[pid] = path
            break  # one mapping per pid is enough
    return out


def _drm_card_to_bdf() -> dict[str, str]:
    """Return {card_path: BDF} by reading /sys/class/drm/cardN/device/uevent.

    Empty dict if /sys is unavailable.
    """
    out: dict[str, str] = {}
    drm_root = Path("/sys/class/drm")
    if not drm_root.is_dir():
        return out
    for entry in drm_root.iterdir():
        if not entry.name.startswith("card"):
            continue
        device = entry / "device"
        if not device.is_dir():
            continue
        # Resolve the canonical PCI BDF via /sys/bus/pci/devices.
        try:
            real = device.resolve()
            bdf = real.name  # e.g. 0000:c8:00.0
        except (OSError, ValueError):
            continue
        out[f"/dev/dri/{entry.name}"] = bdf
        # Render nodes also map to the same card.
        render = entry.with_name(f"renderD{entry.name[len('card'):]}")
        if render.exists() or (entry.parent / render.name).exists():
            out[f"/dev/dri/{render.name}"] = bdf
    return out


def attribute_to_lab_gpu(
    *,
    lab_gpu_bdf: str,
    lab_gpu_uuid: str,
    allowed_pids: Iterable[int] = (),
    drm_only: bool = False,
) -> list[ProcessGpuOwner]:
    """Return every PID that owns the lab GPU, minus the allowlist.

    Args:
        lab_gpu_bdf: the lab GPU's PCI BDF (e.g. "0000:c8:00.0").
        lab_gpu_uuid: the lab GPU's UUID (informational; included in the
            owner so the audit log is unambiguous).
        allowed_pids: PIDs that are explicitly allowed to be on the GPU
            (e.g. an embedding server the operator pinned). These are
            filtered out of the returned list.
        drm_only: if True, ignore rocm-smi and use only the /proc/<pid>/maps
            DRM scan. Useful when rocm-smi is unavailable or for tests.
    """
    allowed = set(int(p) for p in allowed_pids)
    out: list[ProcessGpuOwner] = []
    if not drm_only:
        rocm_map = _rocm_pid_gpu_map()
        for pid, device_index in rocm_map.items():
            if pid in allowed:
                continue
            out.append(
                ProcessGpuOwner(
                    pid=pid,
                    gpu_uuid=lab_gpu_uuid,
                    backend="rocm",
                    detail=f"device_index={device_index}",
                )
            )
    drm_map = _drm_pid_gpu_map()
    card_to_bdf = _drm_card_to_bdf()
    for pid, card_path in drm_map.items():
        if pid in allowed:
            continue
        # Already attributed via rocm? skip.
        if any(o.pid == pid and o.backend == "rocm" for o in out):
            continue
        bdf = card_to_bdf.get(card_path, "")
        if bdf and bdf != lab_gpu_bdf:
            continue  # belongs to a different GPU
        out.append(
            ProcessGpuOwner(
                pid=pid,
                gpu_uuid=lab_gpu_uuid,
                backend="vulkan" if "renderD" in card_path else "drm",
                detail=f"{card_path} (bdf={bdf or 'unknown'})",
            )
        )
    return out