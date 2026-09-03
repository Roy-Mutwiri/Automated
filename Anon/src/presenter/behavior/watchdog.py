"""Standing checks on whether the presenter is still behaving like a person.

Every failure this looks for has actually happened in this project, and none of
them raised an exception when it did. The engine ran, the tests passed, the
video rendered, and the man in it was subtly wrong:

* the head ratcheted to 11.5 degrees and stayed there, because it took a share
  of every *shift* and was never corrected;
* the pitch drifted to -15 degrees for the last two minutes of a five-minute
  clip and stayed there;
* attention spent 75% of its time between two targets 5.5 degrees apart, which
  is what "an image that sometimes moves" looks like from the inside;
* the head's mean yaw was 0.91 degrees while every metric on the dashboard was
  green.

A watchdog is the right shape for this because these are all *statements about
a long window*, not about a frame. Nothing is wrong with any single frame of a
presenter who has been staring at one spot for four minutes.

Each check appends a plain-language warning, and `check()` returns the list. An
empty list is the only passing result.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

__all__ = ["Watchdog", "WatchdogLimits"]


@dataclass
class WatchdogLimits:
    """Thresholds, in the units a person would use to describe the problem."""

    # Attention
    max_fixation: float = 45.0          # s on one target before it is a stare
    min_camera_share: float = 0.06      # fraction of time toward the lens
    max_camera_gap: float = 150.0       # s without looking at the camera
    max_thinking_share: float = 0.12    # fraction of time gazing up and away
    min_families: int = 4               # distinct regions visited

    # Head
    min_yaw_sd: float = 2.5             # deg; below this the head is a photo
    min_yaw_p95: float = 8.0            # deg; the big turns must exist
    pitch_low: float = -14.0
    pitch_high: float = 12.0
    max_pitch_excursion: float = 60.0   # s continuously outside that band

    # Posture
    max_posture_mean: float = 0.45      # abs mean engagement: must revert
    min_posture_sd: float = 0.05        # but must not be frozen either


@dataclass
class Watchdog:
    """Accumulates state over a run and reports what looks inhuman."""

    limits: WatchdogLimits = field(default_factory=WatchdogLimits)

    _t: float = 0.0
    _yaw: list = field(default_factory=list)
    _pitch: list = field(default_factory=list)
    _eng: list = field(default_factory=list)
    _target: str = ""
    _target_since: float = 0.0
    _longest_fixation: float = 0.0
    _longest_target: str = ""
    _fam_time: dict = field(default_factory=dict)
    _last_camera: float = 0.0
    _max_camera_gap: float = 0.0
    _out_of_band_since: float = -1.0
    _max_out_of_band: float = 0.0

    def update(self, dt: float, motion, attention) -> None:
        self._t += dt
        self._yaw.append(motion.head_world_yaw())
        pitch = motion.head_world_pitch()
        self._pitch.append(pitch)
        self._eng.append(motion.posture.engagement)

        name = motion.attention.target
        if name != self._target:
            if self._target:
                held = self._t - self._target_since
                if held > self._longest_fixation:
                    self._longest_fixation = held
                    self._longest_target = self._target
            self._target = name
            self._target_since = self._t

        fam = attention.targets[attention.current].family
        self._fam_time[fam] = self._fam_time.get(fam, 0.0) + dt
        if fam == "CAMERA":
            self._last_camera = self._t
        else:
            gap = self._t - self._last_camera
            if gap > self._max_camera_gap:
                self._max_camera_gap = gap

        if self.limits.pitch_low <= pitch <= self.limits.pitch_high:
            self._out_of_band_since = -1.0
        else:
            if self._out_of_band_since < 0.0:
                self._out_of_band_since = self._t
            span = self._t - self._out_of_band_since
            if span > self._max_out_of_band:
                self._max_out_of_band = span

    # -- reporting ---------------------------------------------------------
    def check(self) -> list[str]:
        lim = self.limits
        out: list[str] = []
        if self._t < 30.0 or not self._yaw:
            return out

        held = self._t - self._target_since
        longest = max(self._longest_fixation, held)
        who = self._longest_target if self._longest_fixation >= held else self._target
        if longest > lim.max_fixation:
            out.append(f"stuck fixation: {longest:.0f}s on {who} "
                       f"(limit {lim.max_fixation:.0f}s)")

        cam = self._fam_time.get("CAMERA", 0.0) / self._t
        if cam < lim.min_camera_share:
            out.append(f"camera contact absent: {100 * cam:.1f}% of the time "
                       f"(floor {100 * lim.min_camera_share:.0f}%)")
        if self._max_camera_gap > lim.max_camera_gap:
            out.append(f"camera contact gap: {self._max_camera_gap:.0f}s without "
                       f"looking at the lens (limit {lim.max_camera_gap:.0f}s)")

        think = self._fam_time.get("THINKING", 0.0) / self._t
        if think > lim.max_thinking_share:
            out.append(f"thinking too often: {100 * think:.1f}% gazing away "
                       f"(ceiling {100 * lim.max_thinking_share:.0f}%)")

        fams = sum(1 for v in self._fam_time.values() if v > 1.0)
        if fams < lim.min_families:
            out.append(f"attention too narrow: {fams} regions visited "
                       f"(floor {lim.min_families})")

        sd = statistics.pstdev(self._yaw)
        if sd < lim.min_yaw_sd:
            out.append(f"head barely turns: yaw sd {sd:.2f} deg "
                       f"(floor {lim.min_yaw_sd:.1f}) - this is the "
                       f"photo-that-sometimes-moves failure")
        p95 = _quantile([abs(v) for v in self._yaw], 0.95)
        if p95 < lim.min_yaw_p95:
            out.append(f"no large head turns: abs yaw p95 {p95:.1f} deg "
                       f"(floor {lim.min_yaw_p95:.1f})")

        if self._max_out_of_band > lim.max_pitch_excursion:
            out.append(f"pitch parked outside comfort: {self._max_out_of_band:.0f}s "
                       f"outside [{lim.pitch_low:.0f}, {lim.pitch_high:.0f}] "
                       f"(limit {lim.max_pitch_excursion:.0f}s)")

        mean_eng = statistics.fmean(self._eng)
        if abs(mean_eng) > lim.max_posture_mean:
            out.append(f"posture not mean-reverting: mean engagement "
                       f"{mean_eng:+.2f} (limit {lim.max_posture_mean:.2f})")
        sd_eng = statistics.pstdev(self._eng)
        if sd_eng < lim.min_posture_sd:
            out.append(f"posture frozen: engagement sd {sd_eng:.3f} "
                       f"(floor {lim.min_posture_sd:.2f})")
        return out

    def summary(self) -> dict:
        yaw = self._yaw or [0.0]
        return dict(
            seconds=self._t,
            yaw_sd=statistics.pstdev(yaw),
            yaw_p95=_quantile([abs(v) for v in yaw], 0.95),
            pitch_mean=statistics.fmean(self._pitch or [0.0]),
            longest_fixation=max(self._longest_fixation,
                                 self._t - self._target_since),
            max_camera_gap=self._max_camera_gap,
            family_share={k: v / max(self._t, 1e-6)
                          for k, v in sorted(self._fam_time.items(),
                                             key=lambda kv: -kv[1])},
            engagement_mean=statistics.fmean(self._eng or [0.0]),
            engagement_sd=statistics.pstdev(self._eng or [0.0]),
        )


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    i = min(int(q * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[i]
