"""Real PID -> GPU attribution for H0.6.

Replaces the H0.5 heuristic (``-ngl``, ``--embedding`` etc) with a
deterministic lookup that consults the kernel:

  1. ROCm (KFD): ``rocm-smi --showpidgpus`` returns PIDs registered with
     the AMD kernel driver together with their DRM device index.

     Attribution rule (H0.6.1): a ROCm PID is owned by the lab GPU
     only when ``device_index == lab_gpu_device_index``. Without the
     lab GPU's index, the lookup cannot distinguish which KFD device a
     PID is on and the entry is reported with ``backend='unknown'``
     (caller decides: BLOCK, allowlist, or retry).

  2. Vulkan / direct amdgpu DRM: ``/proc/<pid>/maps`` exposes the FD
     of an opened DRM device. The mapping's card path
     (``/dev/dri/cardN`` or ``/dev/dri/renderD128``) is resolved to a
     PCI BDF via ``/sys/class/drm/cardN/device/uevent``. The lab GPU's
     BDF (normalized to lower-case) MUST match the mapping's BDF for
     the PID to count as an owner.

     Unresolved DRM devices (BDF not retrievable, or BDF not matching
     the lab) are NEVER positively attributed to the lab GPU; they are
     either filtered (different GPU) or marked ``backend='unknown'``.
     UNKNOWN is a BLOCK signal at the lifecycle gate.

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

from .snapshot import _run


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

    BDFs are case-normalized to lower-case so equality comparisons are
    meaningful across sources.

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
        out[f"/dev/dri/{entry.name}"] = _normalize_bdf(bdf)
        # Render nodes also map to the same card.
        render = entry.with_name(f"renderD{entry.name[len('card'):]}")
        if render.exists() or (entry.parent / render.name).exists():
            out[f"/dev/dri/{render.name}"] = _normalize_bdf(bdf)
    return out


def _normalize_bdf(bdf: str) -> str:
    """Return the canonical lower-case PCI BDF (``0000:c8:00.0``).

    Different rocm-smi / sysfs sources return different casings and
    zero-padded forms; we normalize once here so equality comparisons
    across sources are meaningful.
    """
    if not bdf:
        return ""
    bdf = bdf.strip().lower()
    # Some sources omit the 0000: domain prefix.
    if ":" not in bdf:
        bdf = "0000:" + bdf
    return bdf


def _rocm_device_index_to_bdf() -> dict[int, str]:
    """Build a {device_index: PCI BDF} map from rocm-smi ``--showbus``.

    rocm-smi prints one line per GPU with ``PCI Bus: <BDF>``. We
    enumerate GPUs in the same order as ``rocm-smi --showpidgpus``
    (which gives device_index per PID) so the join is unambiguous.
    """
    raw = _run(["rocm-smi", "--showbus", "--json"], timeout=10.0)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    out: dict[int, str] = {}
    for idx, key in enumerate(sorted(data.keys())):
        card = data[key]
        if not isinstance(card, dict):
            continue
        bdf = card.get("PCI Bus") or card.get("PCI Bus ID") or ""
        out[idx] = _normalize_bdf(bdf)
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

    Attribution rules (H0.6.1):
      * ROCm PIDs only count when their KFD ``device_index`` maps to
        the same PCI BDF as the lab GPU. PIDs whose KFD device
        resolves to a DIFFERENT GPU are not ours; PIDs whose device
        we cannot BDF-resolve are reported with ``backend='unknown'``
        (the lifecycle orchestrator treats UNKNOWN as BLOCK).
      * DRM PIDs only count when their card path resolves to the lab
        GPU's BDF. Unresolvable BDFs are reported with
        ``backend='unknown'``; BDFs that resolve to another GPU are
        filtered.

    BDFs are case-normalized (``normalize_bdf``) so ``0000:C8:00.0`` and
    ``0000:c8:00.0`` compare equal.
    """
    allowed = set(int(p) for p in allowed_pids)
    target_bdf = _normalize_bdf(lab_gpu_bdf)
    if not target_bdf:
        return [
            ProcessGpuOwner(
                pid=0, gpu_uuid=lab_gpu_uuid, backend="unknown",
                detail="lab GPU BDF is empty; cannot attribute",
            ),
        ]
    out: list[ProcessGpuOwner] = []
    if not drm_only:
        rocm_map = _rocm_pid_gpu_map()
        bdf_per_index = _rocm_device_index_to_bdf()
        for pid, device_index in rocm_map.items():
            if pid in allowed:
                continue
            pid_bdf = bdf_per_index.get(device_index, "")
            if pid_bdf == target_bdf:
                out.append(
                    ProcessGpuOwner(
                        pid=pid, gpu_uuid=lab_gpu_uuid, backend="rocm",
                        detail=f"device_index={device_index} bdf={pid_bdf}",
                    )
                )
            elif pid_bdf == "":
                out.append(
                    ProcessGpuOwner(
                        pid=pid, gpu_uuid=lab_gpu_uuid, backend="unknown",
                        detail=f"device_index={device_index} bdf=<unparsed>; do not assume",
                    )
                )
            # else: PID is on a different GPU; not ours.
    drm_map = _drm_pid_gpu_map()
    card_to_bdf = _drm_card_to_bdf()
    for pid, card_path in drm_map.items():
        if pid in allowed:
            continue
        # Already attributed via rocm? skip.
        if any(o.pid == pid and o.backend == "rocm" for o in out):
            continue
        bdf = _normalize_bdf(card_to_bdf.get(card_path, ""))
        if bdf == target_bdf:
            out.append(
                ProcessGpuOwner(
                    pid=pid, gpu_uuid=lab_gpu_uuid,
                    backend="vulkan" if "renderD" in card_path else "drm",
                    detail=f"{card_path} bdf={bdf}",
                )
            )
        elif bdf == "":
            out.append(
                ProcessGpuOwner(
                    pid=pid, gpu_uuid=lab_gpu_uuid, backend="unknown",
                    detail=f"{card_path} bdf=<unresolvable>; do not assume",
                )
            )
        # else: PID is on a different GPU; not ours.
    return out