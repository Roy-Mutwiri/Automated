"""What this machine can actually do.

Every field here exists because something downstream makes a decision from it.
Nothing is collected for information, and nothing that identifies the machine
or its owner is collected at all -- no MAC address, no serial number, no
installed-software inventory.

Two rules the whole module follows:

  A probe never raises.   A machine that reports "unknown GPU" still provisions
                          correctly; a probe that throws during startup does not.
  A probe never hangs.    Everything is bounded, because this runs before the
                          user sees anything at all.

Milestone 1 is CPU-first by decision. GPU fields are recorded when they are
cheap to get, but model selection does not read them: Windows reports VRAM
through Win32_VideoController.AdapterRAM, which is a uint32 and therefore caps
at 4 GB and misreports on plenty of cards. Sizing a multi-gigabyte download
from a number that is wrong on modern hardware is worse than not trying.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from shared.paths import data_root

log = logging.getLogger(__name__)

UNKNOWN = "unknown"

#: Host the gold feed actually connects to. Reachability here is the single
#: most product-specific probe: a corporate, captive or geo-filtered network
#: fails at this socket and nowhere else, and today that surfaces only as an
#: endless reconnect loop long after the user thinks everything is fine.
MARKET_HOST = "data-stream.binance.vision"
MARKET_PORT = 443

#: Cheap generic reachability, used only to tell "no internet" apart from
#: "internet, but this one host is blocked".
INTERNET_HOST = "1.1.1.1"
INTERNET_PORT = 443

PROBE_TIMEOUT_S = 2.0


@dataclass
class HardwareProfile:
    os_name: str = UNKNOWN
    os_build: str = UNKNOWN
    architecture: str = UNKNOWN
    python_version: str = UNKNOWN

    cpu_model: str = UNKNOWN
    cpu_logical_cores: int = 0

    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0

    gpu_vendor: str = UNKNOWN
    gpu_model: str = UNKNOWN
    #: Recorded, never used for selection in milestone 1. See module docstring.
    gpu_usable_for_inference: bool = False
    gpu_unusable_reason: str = "GPU inference not enabled in this milestone"

    disk_data_total_gb: float = 0.0
    disk_data_free_gb: float = 0.0
    data_root: str = ""

    audio_output_devices: list[str] = field(default_factory=list)
    virtual_cable: str | None = None

    network_online: bool = False
    market_feed_reachable: bool = False
    market_feed_latency_ms: int | None = None

    model_server_url: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    # -- the stable subset ------------------------------------------------

    def stable_fields(self) -> dict[str, Any]:
        """Fields the profile hash is computed over.

        Free disk, available RAM, network state and latency are deliberately
        excluded: they change between any two launches, and a hash that never
        matches means the fast path never triggers and the machine re-probes
        and re-evaluates forever.

        Total RAM is rounded to the nearest GB for the same reason -- the raw
        byte count moves by a few megabytes across reboots.

        The GPU is excluded too, for a subtler reason: its probe is optional,
        so including it would make the hash depend on *whether the probe ran*.
        `doctor` probes the GPU and the launch path does not, so every doctor
        run would look like a hardware change and force a re-evaluation.
        """
        return {
            "os_name": self.os_name,
            "architecture": self.architecture,
            "cpu_model": _normalise(self.cpu_model),
            "cpu_logical_cores": self.cpu_logical_cores,
            "ram_total_gb": round(self.ram_total_gb),
            "virtual_cable": self.virtual_cable is not None,
        }

    def profile_hash(self) -> str:
        blob = json.dumps(self.stable_fields(), sort_keys=True)
        return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()[:32]

    # -- human summary ----------------------------------------------------

    def describe(self) -> str:
        gpu = self.gpu_model if self.gpu_model != UNKNOWN else "no GPU detected"
        cable = self.virtual_cable or "no virtual cable"
        devices = len(self.audio_output_devices)
        feed = (
            f"gold feed reachable ({self.market_feed_latency_ms} ms)"
            if self.market_feed_reachable
            else "gold feed NOT reachable"
        )
        net = "internet OK" if self.network_online else "no internet"
        return (
            f"{self.os_name} ({self.architecture}) | {self.cpu_logical_cores} cores | "
            f"{self.ram_total_gb:.1f} GB RAM ({self.ram_available_gb:.1f} free)\n"
            f"{gpu} | will use the processor for AI\n"
            f"{self.disk_data_free_gb:.1f} GB free at {self.data_root}\n"
            f"{devices} audio output(s) | {cable}\n"
            f"{net} | {feed}"
        )


def _normalise(text: str) -> str:
    return " ".join((text or UNKNOWN).split()).lower()


# -- individual probes -----------------------------------------------------


def _probe_os(p: HardwareProfile) -> None:
    p.os_name = platform.system() or UNKNOWN
    p.architecture = platform.machine() or UNKNOWN
    p.python_version = platform.python_version()
    if sys.platform == "win32":
        try:
            release, version, csd, _ptype = platform.win32_ver()
            edition = platform.win32_edition() if hasattr(platform, "win32_edition") else ""
            p.os_name = f"Windows {release} {edition}".strip()
            p.os_build = version or csd or UNKNOWN
        except Exception:
            p.os_build = UNKNOWN
    else:
        p.os_build = platform.release() or UNKNOWN


def _probe_cpu(p: HardwareProfile) -> None:
    p.cpu_logical_cores = os.cpu_count() or 0
    # Physical cores would need psutil or a wmic shell-out; wmic is deprecated
    # and absent on newer Windows builds, and nothing in milestone 1 makes a
    # decision from it, so it is not collected.
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            with key:
                p.cpu_model = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            return
        except Exception:
            pass
    p.cpu_model = platform.processor() or UNKNOWN


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _probe_ram(p: HardwareProfile) -> None:
    if sys.platform == "win32":
        try:
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                p.ram_total_gb = round(status.ullTotalPhys / 1e9, 2)
                p.ram_available_gb = round(status.ullAvailPhys / 1e9, 2)
                return
        except Exception:
            pass
    try:  # POSIX fallback; keeps the module usable in CI on Linux
        p.ram_total_gb = round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 2
        )
        p.ram_available_gb = round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") / 1e9, 2
        )
    except (ValueError, OSError, AttributeError):
        pass


def _probe_gpu(p: HardwareProfile) -> None:
    """Best effort, and explicitly not a selection input.

    A PowerShell CIM query costs several hundred milliseconds. It is worth that
    to put the GPU name in a support report and to explain to the user why the
    processor is being used, but nothing branches on the result.
    """
    if sys.platform != "win32":
        return
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "(Get-CimInstance Win32_VideoController | Select-Object -First 1"
             " -ExpandProperty Name)"],
            capture_output=True, text=True, timeout=6.0,
        )
        name = (out.stdout or "").strip()
        if name:
            p.gpu_model = name
            lowered = name.lower()
            for vendor in ("nvidia", "amd", "radeon", "intel"):
                if vendor in lowered:
                    p.gpu_vendor = "AMD" if vendor == "radeon" else vendor.upper()
                    break
    except Exception:
        pass


def _probe_disk(p: HardwareProfile) -> None:
    root = data_root()
    p.data_root = str(root)
    try:
        root.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(root)
        p.disk_data_total_gb = round(usage.total / 1e9, 1)
        p.disk_data_free_gb = round(usage.free / 1e9, 1)
    except OSError as exc:
        log.debug("disk probe failed: %s", exc)


def _probe_audio(p: HardwareProfile) -> None:
    try:
        from platform_.audio.devices import find_virtual_cable, list_output_devices

        p.audio_output_devices = [d.name for d in list_output_devices()]
        cable = find_virtual_cable()
        p.virtual_cable = cable.name if cable else None
    except Exception as exc:
        log.debug("audio probe failed: %s", exc)


def _reachable(host: str, port: int, timeout: float = PROBE_TIMEOUT_S) -> tuple[bool, int | None]:
    import time

    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, int((time.perf_counter() - started) * 1000)
    except OSError:
        return False, None


def _probe_network(p: HardwareProfile) -> None:
    p.network_online, _ = _reachable(INTERNET_HOST, INTERNET_PORT)
    p.market_feed_reachable, p.market_feed_latency_ms = _reachable(MARKET_HOST, MARKET_PORT)
    # A blocked market host on an otherwise working connection is the case
    # worth distinguishing, so record both rather than collapsing them.
    if p.market_feed_reachable:
        p.network_online = True


def _probe_model_server(p: HardwareProfile) -> None:
    """Is anything already serving? A plain TCP check, not a health request.

    The real health and model checks belong to readiness; this only answers
    "do we need to walk the user through installing Ollama".
    """
    try:
        from platform_.llm.discovery import KNOWN_ENDPOINTS
    except Exception:
        return
    for _label, url in KNOWN_ENDPOINTS:
        try:
            hostport = url.split("//", 1)[1].split("/", 1)[0]
            host, _, port = hostport.partition(":")
            ok, _ = _reachable(host, int(port or 80), timeout=0.4)
            if ok:
                p.model_server_url = url
                return
        except (ValueError, IndexError):
            continue


# -- the probe -------------------------------------------------------------


def probe(include_gpu: bool = False) -> HardwareProfile:
    """Measure the machine.

    GPU identification is off by default: a PowerShell CIM query costs about
    five seconds of process startup, and nothing in this milestone makes a
    decision from the answer. `doctor` turns it on, because a support report
    is worth the wait and nobody is waiting on a session to start.
    """
    p = HardwareProfile()
    probes = [
        ("os", _probe_os),
        ("cpu", _probe_cpu),
        ("ram", _probe_ram),
        ("disk", _probe_disk),
        ("audio", _probe_audio),
        ("network", _probe_network),
        ("model_server", _probe_model_server),
    ]
    if include_gpu:
        probes.append(("gpu", _probe_gpu))
    for name, fn in probes:
        try:
            fn(p)
        except Exception as exc:  # a probe must never take down the launch
            log.warning("capability probe %r failed: %s", name, exc)
    return p
