"""Identity block for H0.5 evaluator.

Every measurement is attributed to a frozen identity triple:
  1. Build identity: the llama.cpp SOURCE SHA + build command + binary SHA.
  2. Model identity:  the GGUF SHA + size.
  3. Host identity:   the GPU + ROCm + kernel + cmake flags + compiler.

This triple is a single JSON object the DB stores once per experiment and
the evaluator hashes to a stable ``identity_hash`` so two measurements on
the same triple are unambiguously comparable and a measurement on a
different triple is unambiguously NOT comparable (without explicit
re-baselining).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def _sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, streaming for large GGUF / binary inputs."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_git_head(repo: str | os.PathLike) -> Optional[str]:
    """Return the current commit SHA of a git checkout, or None if not a repo."""
    repo = Path(repo)
    head = repo / ".git" / "HEAD"
    if not head.is_file():
        return None
    try:
        ref = head.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if ref.startswith("ref: "):
        ref_path = repo / ".git" / ref.removeprefix("ref: ")
        if ref_path.is_file():
            try:
                return ref_path.read_text(encoding="utf-8").strip()
            except OSError:
                return None
    return ref  # detached HEAD


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: float = 10.0) -> Optional[str]:
    """Run a short command, return stripped stdout or None on any failure."""
    try:
        out = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


@dataclasses.dataclass(frozen=True)
class IdentityBlock:
    """Frozen triple describing one reproducible measurement context."""

    # Build
    source_repo: str
    source_sha: Optional[str]
    binary_path: str
    binary_sha256: Optional[str]
    # Model
    model_path: str
    model_sha256: Optional[str]
    model_size_bytes: Optional[int]
    # Host
    gpu_type: str
    gfx_arch: str
    rocm_version: Optional[str]
    kernel_release: Optional[str]
    cmake_flags_sha: str  # sha256 of the canonicalized cmake invocation
    compiler_sha: Optional[str]  # gcc/clang version
    cpu_model: Optional[str]

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True, indent=2)

    @property
    def identity_hash(self) -> str:
        """Stable hash of the identity block. Two measurements on the same
        identity compare; two measurements on different identities do NOT,
        unless the operator explicitly tags the experiment as cross-identity.
        """
        # Drop volatile fields (size_bytes) from the canonical hash so the
        # identity stays stable across re-runs of the same inputs.
        canonical = dataclasses.asdict(self)
        canonical.pop("model_size_bytes")
        return _sha256_text(json.dumps(canonical, sort_keys=True))


# Allowlist of GPU product names the evaluator considers RDNA3-friendly.
_GPU_PRODUCT_ALIASES = {
    "RX 7900 XTX": "rx7900xtx",
    "RADEON 890M": "radeon890m",
}


def detect_gpu_type() -> str:
    """Return the gpu_type key for the active GPU (rx7900xtx / radeon890m /
    mi300x / mi355x / unknown) using rocm-smi --showproductname.

    Mirrors the gpu_types._PRODUCT_ALIASES bijection in the Hyperloom
    fork; the evaluator reads it independently rather than importing
    Hyperloom, so the load gate can run before the CLI is reachable.
    """
    smi = shutil.which("rocm-smi")
    if not smi:
        return "unknown"
    out = _run([smi, "--showproductname"])
    if not out:
        return "unknown"
    upper = out.upper()
    for tag, key in sorted(_GPU_PRODUCT_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if tag.upper() in upper:
            return key
    return "unknown"


def _cmake_flags_sha(repo: Path) -> str:
    """Hash the canonicalized cmake flags the build was configured with.

    Reads the ``CMakeCache.txt`` in ``build/`` when present (returns the
    joined CMake cache variables) so two builds with the same flags hash
    equal even when the source SHA differs.
    """
    cache = repo / "build" / "CMakeCache.txt"
    if not cache.is_file():
        return _sha256_text("<no CMakeCache.txt>")
    relevant: list[str] = []
    for raw in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        if "CMAKE_CXX_FLAGS" in line or "GGML_" in line or "LLAMA_" in line:
            relevant.append(line)
    return _sha256_text("\n".join(sorted(relevant)))


def _compiler_version() -> Optional[str]:
    for name in ("gcc", "clang", "cc"):
        path = shutil.which(name)
        if not path:
            continue
        out = _run([path, "--version"])
        if out:
            return out.splitlines()[0]
    return None


def _rocm_version() -> Optional[str]:
    out = _run(["hipconfig", "--version"]) if shutil.which("hipconfig") else None
    if out:
        return out
    out = _run(["rocminfo"]) if shutil.which("rocminfo") else None
    if out and "Runtime Version" in out:
        for ln in out.splitlines():
            if "Runtime Version" in ln:
                return ln.strip()
    return None


def _kernel_release() -> Optional[str]:
    return _run(["uname", "-r"])


def _cpu_model() -> Optional[str]:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return None
    for ln in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
        if ln.lower().startswith("model name"):
            return ln.split(":", 1)[1].strip()
    return None


def collect_identity(
    *,
    source_repo: str | os.PathLike,
    binary_path: str | os.PathLike,
    model_path: str | os.PathLike,
    gpu_type: str | None = None,
    gfx_arch: str | None = None,
) -> IdentityBlock:
    """Build an IdentityBlock from the current host state.

    Args:
        source_repo: llama.cpp checkout path (source SHA from .git/HEAD).
        binary_path: llama-server binary path (sha256 if present).
        model_path: GGUF model file path (sha256 if present).
        gpu_type: override autodetect.
        gfx_arch: override autodetect.
    """
    src = Path(source_repo)
    bin_path = Path(binary_path)
    model = Path(model_path)

    detected_gpu = gpu_type or detect_gpu_type()
    arch = gfx_arch or _infer_gfx_arch(detected_gpu)

    return IdentityBlock(
        source_repo=str(src.resolve()) if src.exists() else str(src),
        source_sha=_read_git_head(src),
        binary_path=str(bin_path),
        binary_sha256=_sha256_file(bin_path) if bin_path.is_file() else None,
        model_path=str(model),
        model_sha256=_sha256_file(model) if model.is_file() else None,
        model_size_bytes=model.stat().st_size if model.is_file() else None,
        gpu_type=detected_gpu,
        gfx_arch=arch,
        rocm_version=_rocm_version(),
        kernel_release=_kernel_release(),
        cmake_flags_sha=_cmake_flags_sha(src),
        compiler_sha=_compiler_version(),
        cpu_model=_cpu_model(),
    )


#: Static gfx-arch table mirroring Hyperloom's ``_GFX_TO_RUNNER`` mapping
#: for the boards the evaluator supports on day one.
_GPU_TO_GFX: dict[str, str] = {
    "rx7900xtx": "gfx1100",
    "radeon890m": "gfx1150",
    "mi300x": "gfx942",
    "mi355x": "gfx950",
}


def _infer_gfx_arch(gpu_type: str) -> str:
    return _GPU_TO_GFX.get(gpu_type, "unknown")