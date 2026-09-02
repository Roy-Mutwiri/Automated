"""Draw the canonical room from above, from config, not by hand.

The floorplan is a *derived view* of `config/room_geometry.yaml` and
`config/cameras.yaml`, never an independently drawn picture. If someone edits
the geometry and the plan does not move, the plan is lying - so it is generated
every time rather than saved as art.

It is also the validation tool for Camera 6: an overhead render and this plan
must agree about where the desk, chair and monitors are.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import yaml                              # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def rect(ax, centre, size, colour, label=None, alpha=0.85):
    x, y = centre[0] - size[0] / 2, centre[1] - size[1] / 2
    ax.add_patch(Rectangle((x, y), size[0], size[1], facecolor=colour,
                           edgecolor="#101010", linewidth=0.8, alpha=alpha))
    if label:
        ax.annotate(label, (centre[0], centre[1]), color="#f0f0f0", fontsize=6.5,
                    ha="center", va="center")


def main() -> int:
    g = yaml.safe_load((ROOT / "config/room_geometry.yaml").read_text(encoding="utf-8"))
    c = yaml.safe_load((ROOT / "config/cameras.yaml").read_text(encoding="utf-8"))

    r = g["room"]
    w, d = r["width"], r["depth"]
    fig, ax = plt.subplots(figsize=(9, 10), dpi=140)
    fig.patch.set_facecolor("#16151a")
    ax.set_facecolor("#1d1c22")

    ax.add_patch(Rectangle((-w / 2, 0), w, d, facecolor="#232128",
                           edgecolor="#6a6a72", linewidth=1.6))

    # The slat wall at y = 0, drawn at true pitch so the plan carries the same
    # global phase the 3D scene does.
    wc = g["wall_walnut"]
    x = wc["slat_phase_x"]
    while x <= wc["x_to"]:
        ax.add_patch(Rectangle((x - wc["slat_width"] / 2, 0), wc["slat_width"],
                               wc["slat_depth"], facecolor="#6b4a2f",
                               edgecolor="none"))
        x += wc["slat_pitch"]
    ax.annotate("walnut_wall_01", (0, -0.16), color="#c99a63", fontsize=8,
                ha="center")

    for m in g["monitors"]:
        rect(ax, (m["centre"][0], m["centre"][1]), (m["size"][0], m["size"][1] + 0.06),
             "#3b6ea5" if m.get("powered") else "#3a3a40", m["id"])
    for s in g.get("speakers", []):
        rect(ax, (s["centre"][0], s["centre"][1]), (s["size"][0], s["size"][1]),
             "#2f2f36", s["id"])

    dk = g["desk"]
    rect(ax, (dk["centre"][0], dk["centre"][1]), dk["top_size"][:2], "#2b2b31",
         "desk_main")

    ch = g["chair"]
    rect(ax, (ch["base_centre"][0], ch["base_centre"][1]),
         (ch["seat_size"][0], ch["seat_size"][1]), "#26262b", "chair_main")

    h = g["human"]
    ax.add_patch(Circle((h["hip"][0], h["hip"][1]), 0.20, facecolor="#c98d63",
                        edgecolor="#101010", linewidth=0.8))
    ax.annotate("streamer", (h["hip"][0], h["hip"][1] - 0.30), color="#f0c9a5",
                fontsize=8, ha="center")

    mic = g["microphone"]
    js = mic["joints"]
    ax.plot([p[0] for p in js], [p[1] for p in js], color="#8a8a92", linewidth=2)
    ax.annotate("mic_main", (js[-1][0], js[-1][1] - 0.12), color="#9a9aa2",
                fontsize=7, ha="center")

    for lm in g.get("landmarks", []):
        rect(ax, (lm["centre"][0], lm["centre"][1]),
             (lm["size"][0], lm["size"][1]), "#3a3340", lm["id"])

    for lspec in ("key", "fill", "rim"):
        p = g["lighting"][lspec]["position"]
        ax.plot(p[0], p[1], marker="*", markersize=11, color="#ffd479")
        ax.annotate(lspec, (p[0], p[1] + 0.12), color="#ffd479", fontsize=7,
                    ha="center")

    # Cameras with their actual aim.
    for spec in c["cameras"]:
        p, t = spec["position"], spec["look_at"]
        on = spec.get("enabled", True)
        colour = "#5ee08a" if on else "#5a6a60"
        ax.plot(p[0], p[1], marker="o", markersize=7, color=colour)
        ax.annotate(f"{spec['id']} {spec['focal_length_mm']:.0f}mm",
                    (p[0], p[1] + 0.14), color=colour, fontsize=7.5, ha="center")
        ax.annotate("", xy=(t[0], t[1]), xytext=(p[0], p[1]),
                    arrowprops=dict(arrowstyle="->", color=colour, alpha=0.55,
                                    linewidth=1.0))

    ax.set_xlim(-w / 2 - 0.3, w / 2 + 0.3)
    ax.set_ylim(-0.4, d + 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("X  (metres)", color="#c8c8d0")
    ax.set_ylabel("Y  depth from slat wall  (metres)", color="#c8c8d0")
    ax.tick_params(colors="#8a8a92")
    for s in ax.spines.values():
        s.set_color("#3a3a42")
    ax.set_title(f"Canonical streaming room - {g['scene_version']} - top view",
                 color="#f0f0f4", fontsize=11)
    ax.grid(color="#2e2e36", linewidth=0.5)

    out = ROOT / "docs/room_floorplan.png"
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"[floorplan] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
