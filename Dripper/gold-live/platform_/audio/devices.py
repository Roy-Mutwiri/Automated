"""Audio output device discovery.

Each session's TTS must go to a virtual audio cable that LIVE Studio picks up
as a MICROPHONE. Do not let LIVE Studio capture desktop audio -- that sends
every system sound to the audience: notification chimes, alerts, a stray
browser tab.

A reassigned or missing audio device is a silent failure. The stream stays up,
the host appears to be talking, and nobody hears anything. It can run for hours
before someone notices, so it is a health check rather than a startup
assumption.

Install one virtual cable per session on each device:
  VB-CABLE          one cable, free              https://vb-audio.com/Cable/
  VoiceMeeter       several cables plus a mixer
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Names these products expose on Windows and Linux.
VIRTUAL_CABLE_PATTERNS = [
    re.compile(r"CABLE Input", re.I),        # VB-CABLE
    re.compile(r"VoiceMeeter (Aux )?Input", re.I),
    re.compile(r"VB-Audio", re.I),
    re.compile(r"pulse|pipewire.*sink", re.I),  # Linux virtual sinks
]


@dataclass(slots=True)
class AudioDevice:
    index: int
    name: str
    channels: int
    sample_rate: int
    is_default: bool = False

    @property
    def looks_virtual(self) -> bool:
        return any(p.search(self.name) for p in VIRTUAL_CABLE_PATTERNS)


def list_output_devices() -> list[AudioDevice]:
    """Enumerate output devices. Empty list if no audio backend is installed."""
    try:
        import sounddevice as sd
    except ImportError:
        log.warning("sounddevice not installed; audio output unavailable")
        return []

    devices: list[AudioDevice] = []
    try:
        default_out = sd.default.device[1]
    except Exception:  # noqa: BLE001
        default_out = None

    for index, info in enumerate(sd.query_devices()):
        if info.get("max_output_channels", 0) <= 0:
            continue
        devices.append(
            AudioDevice(
                index=index,
                name=str(info.get("name", f"device {index}")),
                channels=int(info["max_output_channels"]),
                sample_rate=int(info.get("default_samplerate", 48000)),
                is_default=(index == default_out),
            )
        )
    return devices


def find_virtual_cable(preferred_name: str | None = None) -> AudioDevice | None:
    """Locate the virtual cable this session should output to.

    `preferred_name` is a substring match, so seven cables on one machine can
    be told apart by configuring each session with its own.
    """
    devices = list_output_devices()
    if preferred_name:
        for d in devices:
            if preferred_name.lower() in d.name.lower():
                return d
        log.warning("no output device matching %r", preferred_name)
        return None

    for d in devices:
        if d.looks_virtual:
            return d
    return None


def describe_audio_setup() -> str:
    """Human-readable summary for the setup runbook and health output."""
    devices = list_output_devices()
    if not devices:
        return "No audio output devices found (is sounddevice installed?)"

    lines = ["Output devices:"]
    for d in devices:
        tags = []
        if d.is_default:
            tags.append("default")
        if d.looks_virtual:
            tags.append("VIRTUAL CABLE")
        suffix = f"  [{', '.join(tags)}]" if tags else ""
        lines.append(f"  {d.index:>3}  {d.name}{suffix}")

    if not any(d.looks_virtual for d in devices):
        lines += [
            "",
            "No virtual cable detected. Install VB-CABLE and point LIVE Studio's",
            "MICROPHONE input at it -- never at desktop audio, or every system",
            "sound on this machine goes out to the audience.",
        ]
    return "\n".join(lines)
