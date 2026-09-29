"""Which GPU this is, from ``nvidia-smi``: model, family, memory and driver.

``inferlint gpu-inspect`` prints these next to the free-card check (T2), so a test can
record the card it ran on. Reading them works wherever ``nvidia-smi`` does (Linux, WSL,
Windows); the free-card check itself needs Linux.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["GpuInfo", "family", "nvidia_smi_path", "parse", "read_gpus"]

_BASE = ("index", "name", "memory.total", "memory.used", "memory.free", "driver_version")
_CAP = "compute_cap"  # needs a 2022 or newer driver

# Compute capability to architecture family. More specific prefixes come first.
_FAMILIES: tuple[tuple[tuple[int, ...], str], ...] = (
    ((6,), "Pascal"),
    ((7, 0), "Volta"),
    ((7, 2), "Volta"),
    ((7, 5), "Turing"),
    ((8, 9), "Ada Lovelace"),
    ((8,), "Ampere"),
    ((9,), "Hopper"),
    ((10,), "Blackwell"),
    ((11,), "Blackwell"),
    ((12,), "Blackwell"),
)


def family(compute_cap: str | None) -> str | None:
    """``"8.9"`` -> ``"Ada Lovelace"``; None when the capability is missing or unknown."""
    if not compute_cap:
        return None
    try:
        parts = tuple(int(x) for x in compute_cap.split("."))
    except ValueError:
        return None
    for prefix, name in _FAMILIES:
        if parts[: len(prefix)] == prefix:
            return name
    return None


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    memory_total_mib: int | None
    memory_used_mib: int | None
    memory_free_mib: int | None
    driver: str | None
    compute_cap: str | None

    @property
    def family(self) -> str | None:
        return family(self.compute_cap)

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "family": self.family}


def _value(s: str) -> str | None:
    s = s.strip()
    return None if not s or s.startswith("[") else s  # "[N/A]", "[Not Supported]"


def _mib(s: str) -> int | None:
    v = _value(s)
    try:
        return None if v is None else int(float(v))
    except ValueError:
        return None


def parse(text: str, *, with_cap: bool = True) -> list[GpuInfo]:
    """Parse ``--format=csv,noheader,nounits`` output, one line per GPU."""
    n = len(_BASE) + (1 if with_cap else 0)
    gpus: list[GpuInfo] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < n:
            continue
        # A model name could contain a comma: everything between the index and the
        # fixed trailing fields belongs to it.
        tail = parts[len(parts) - (n - 2) :]
        name = ", ".join(parts[1 : len(parts) - (n - 2)])
        try:
            index = int(parts[0])
        except ValueError:
            continue
        gpus.append(
            GpuInfo(
                index=index,
                name=name,
                memory_total_mib=_mib(tail[0]),
                memory_used_mib=_mib(tail[1]),
                memory_free_mib=_mib(tail[2]),
                driver=_value(tail[3]),
                compute_cap=_value(tail[4]) if with_cap else None,
            )
        )
    return gpus


def nvidia_smi_path() -> str:
    return shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"


def _run(args: list[str]) -> str:
    return subprocess.run(
        [nvidia_smi_path(), *args], check=True, capture_output=True, text=True, timeout=30
    ).stdout


def read_gpus(query: Callable[[list[str]], str] = _run) -> list[GpuInfo]:
    """Every GPU ``nvidia-smi`` reports. Raises OSError or SubprocessError without one."""
    fmt = "--format=csv,noheader,nounits"
    try:
        return parse(query([f"--query-gpu={','.join((*_BASE, _CAP))}", fmt]))
    except subprocess.CalledProcessError:
        # An older driver rejects the compute_cap field; ask again without it.
        return parse(query([f"--query-gpu={','.join(_BASE)}", fmt]), with_cap=False)
