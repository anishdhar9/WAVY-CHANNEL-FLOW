"""
geometry/channel_gen.py

Parametric 2-D channel geometry defined by a bottom and a top wall curve.

The default constructor builds the classic wavy channel with sinusoidal
walls; the phase difference phi between them controls the channel mode:
  phi=0°   → in-phase   (constant gap, channel translates bodily)
  phi=180° → anti-phase (converging-diverging, maximum gap oscillation)

Arbitrary 2-D channel shapes are supported through two factory methods:
  ChannelGeometry.from_profiles(x, y_bottom, y_top)   — tabulated walls
  ChannelGeometry.from_functions(L, H, f_bot, f_top)  — callable walls

All computations are pure NumPy vectorised operations over the x-array.
No Python loops over cells.
"""

from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


# Muted publication palette shared by geometry and mesh plots.
STYLE = {
    "wall_bottom": "#1f4e79",   # dark steel blue
    "wall_top":    "#7b1f1f",   # dark crimson
    "inlet":       "#1c5e38",   # dark forest green
    "outlet":      "#8b4a00",   # dark burnt orange
    "fluid_fill":  "#4a7fa5",
    "grid":        "#b8b8b8",
    "axes_bg":     "#f7f7f4",
    "spine":       "#b0b0b0",
    "text":        "#333333",
    "tick":        "#555555",
}


def apply_axes_style(ax: plt.Axes) -> None:
    """Apply the shared muted figure style to an Axes."""
    ax.set_facecolor(STYLE["axes_bg"])
    for spine in ax.spines.values():
        spine.set_color(STYLE["spine"])
        spine.set_linewidth(0.8)
    ax.tick_params(colors=STYLE["tick"], labelsize=9)
    ax.xaxis.label.set_color(STYLE["text"])
    ax.yaxis.label.set_color(STYLE["text"])
    ax.title.set_color(STYLE["text"])


class ChannelGeometry:
    """Parametric wavy-channel geometry.

    Parameters
    ----------
    L       : channel length [m]
    H       : mean channel height (wall-to-wall at x=0 for AR=0) [m]
    lam     : corrugation wavelength [m]
    AR      : amplitude ratio A/lambda  (dimensionless)
    phi_deg : phase shift of top wall relative to bottom wall [degrees]
    n_pts   : number of sample points along x (default 300)

    Attributes
    ----------
    A        : corrugation amplitude [m]  (= AR * lam)
    phi_rad  : phase shift [radians]
    x        : (n_pts,) x-coordinate array [m]
    y_bottom : (n_pts,) bottom wall y-coordinates [m]
    y_top    : (n_pts,) top wall y-coordinates [m]
    """

    def __init__(self, L: float, H: float, lam: float,
                 AR: float, phi_deg: float, n_pts: int = 300):
        self.L       = float(L)
        self.H       = float(H)
        self.lam     = float(lam)
        self.AR      = float(AR)
        self.phi_deg = float(phi_deg)
        self.n_pts   = int(n_pts)

        self.A       = AR * lam
        self.phi_rad = np.radians(phi_deg)
        self.x       = np.linspace(0.0, L, self.n_pts)

        if AR == 0.0:
            self.y_bottom = np.zeros(self.n_pts)
            self.y_top    = np.full(self.n_pts, H)
        else:
            k = 2.0 * np.pi / lam
            self.y_bottom = self.A * np.sin(k * self.x)
            self.y_top    = H + self.A * np.sin(k * self.x + self.phi_rad)

    # ------------------------------------------------------------------
    # General-geometry factories
    # ------------------------------------------------------------------

    @classmethod
    def from_profiles(cls, x: np.ndarray,
                      y_bottom: np.ndarray,
                      y_top: np.ndarray) -> "ChannelGeometry":
        """Build a channel from tabulated wall profiles.

        Accepts arbitrary single-valued wall curves y_bottom(x), y_top(x)
        sampled on a common, strictly increasing x-array. The mesh
        generator interpolates these profiles, so any 2-D channel whose
        walls are functions of x can be meshed this way.

        Parameters
        ----------
        x        : (n,) strictly increasing streamwise coordinates [m]
        y_bottom : (n,) bottom wall y-coordinates [m]
        y_top    : (n,) top wall y-coordinates [m]
        """
        x        = np.asarray(x,        dtype=float)
        y_bottom = np.asarray(y_bottom, dtype=float)
        y_top    = np.asarray(y_top,    dtype=float)

        if x.ndim != 1 or x.size < 2:
            raise ValueError("x must be a 1-D array with at least 2 points")
        if y_bottom.shape != x.shape or y_top.shape != x.shape:
            raise ValueError(
                f"Wall profiles must match x: x{x.shape}, "
                f"y_bottom{y_bottom.shape}, y_top{y_top.shape}")
        if np.any(np.diff(x) <= 0):
            raise ValueError("x must be strictly increasing")
        if np.any(y_top <= y_bottom):
            i = int(np.argmin(y_top - y_bottom))
            raise ValueError(
                f"y_top must lie above y_bottom everywhere; walls cross "
                f"at x = {x[i]:.6g} m (gap = {(y_top - y_bottom)[i]:.3g} m)")

        geom = cls.__new__(cls)
        geom.L        = float(x[-1] - x[0])
        geom.H        = float(np.mean(y_top - y_bottom))
        geom.lam      = geom.L                 # no intrinsic wavelength
        geom.AR       = 0.0
        geom.phi_deg  = 0.0
        geom.A        = 0.0
        geom.phi_rad  = 0.0
        geom.n_pts    = int(x.size)
        geom.x        = x - x[0]               # normalise to start at x=0
        geom.y_bottom = y_bottom
        geom.y_top    = y_top
        return geom

    @classmethod
    def from_functions(cls, L: float,
                       f_bottom: Callable[[np.ndarray], np.ndarray],
                       f_top: Callable[[np.ndarray], np.ndarray],
                       n_pts: int = 500) -> "ChannelGeometry":
        """Build a channel from callable wall shape functions.

        Parameters
        ----------
        L        : channel length [m]
        f_bottom : vectorised y_bottom(x) callable [m]
        f_top    : vectorised y_top(x) callable [m]
        n_pts    : number of sample points along x
        """
        x = np.linspace(0.0, float(L), int(n_pts))
        yb = np.broadcast_to(np.asarray(f_bottom(x), dtype=float), x.shape)
        yt = np.broadcast_to(np.asarray(f_top(x),    dtype=float), x.shape)
        return cls.from_profiles(x, yb, yt)

    # ------------------------------------------------------------------
    # Validity check
    # ------------------------------------------------------------------

    def check_validity(self) -> tuple[float, float, float]:
        """Check the channel has no self-intersections.

        Computes the local gap h(x) = y_top(x) - y_bottom(x) at every
        sample point and enforces a minimum-gap constraint.

        Returns
        -------
        (min_gap, max_gap, mean_gap) in metres

        Raises
        ------
        ValueError
            If min_gap < 0.001 m (self-intersection risk).
        """
        gap     = self.y_top - self.y_bottom
        min_gap  = float(np.min(gap))
        max_gap  = float(np.max(gap))
        mean_gap = float(np.mean(gap))

        if min_gap < 0.001:
            raise ValueError(
                f"Channel self-intersection risk: "
                f"min gap = {min_gap * 1e3:.3f} mm  "
                f"(AR={self.AR:.3f}, phi={self.phi_deg:.1f}°). "
                f"Reduce AR or adjust phi."
            )
        return min_gap, max_gap, mean_gap

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, filepath: str | Path) -> None:
        """Save x, y_bottom, y_top to a CSV file.

        Parameters
        ----------
        filepath : destination path (parent directory is created if needed)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "x [m]":        self.x,
            "y_bottom [m]": self.y_bottom,
            "y_top [m]":    self.y_top,
        }).to_csv(filepath, index=False)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(self, save_path: str | Path | None = None) -> plt.Figure:
        """Plot the channel profile with both walls, inlet/outlet markers,
        and amplitude / wavelength annotations.

        Parameters
        ----------
        save_path : if given, save the figure to this path (PNG, 150 dpi).
                    The parent directory is created if it does not exist.

        Returns
        -------
        matplotlib Figure
        """
        x_mm   = self.x * 1e3
        yb_mm  = self.y_bottom * 1e3
        yt_mm  = self.y_top * 1e3

        # y-axis limits with a small margin
        margin = max(self.H * 0.12, 0.5e-3) * 1e3  # mm
        y_lo   = yb_mm.min() - margin
        y_hi   = yt_mm.max() + margin
        x_lo   = 0.0
        x_hi   = self.L * 1e3

        fig, ax = plt.subplots(figsize=(11, 3.2))
        apply_axes_style(ax)

        # Channel interior fill
        ax.fill_between(x_mm, yb_mm, yt_mm, alpha=0.12,
                        color=STYLE["fluid_fill"], lw=0,
                        label="fluid domain")

        # Wall lines
        ax.plot(x_mm, yb_mm, color=STYLE["wall_bottom"], lw=1.6,
                label="bottom wall")
        ax.plot(x_mm, yt_mm, color=STYLE["wall_top"], lw=1.6,
                label="top wall")

        # Inlet / outlet verticals
        ax.axvline(x_lo, color=STYLE["inlet"],  ls="--", lw=1.2,
                   label="inlet")
        ax.axvline(x_hi, color=STYLE["outlet"], ls="--", lw=1.2,
                   label="outlet")

        # ── Amplitude annotation (only for wavy cases) ────────────────
        if self.AR > 0.0:
            # Arrow at x = lam/4 (peak of bottom wall)
            x_ann_mm = self.lam / 4.0 * 1e3
            y_base   = float(np.interp(self.lam / 4.0, self.x, self.y_bottom)) * 1e3
            y_peak   = y_base  # peak of bottom wall above y=0 reference
            # We annotate the amplitude A relative to the zero line
            ax.annotate(
                "", xy=(x_ann_mm, self.A * 1e3), xytext=(x_ann_mm, 0.0),
                arrowprops=dict(arrowstyle="<->",
                                color=STYLE["wall_bottom"], lw=1.1),
            )
            ax.text(x_ann_mm + x_hi * 0.012, self.A * 1e3 / 2.0,
                    f"A = {self.A * 1e3:.2f} mm",
                    fontsize=8, color=STYLE["wall_bottom"], va="center")

            # Wavelength bracket just above the top wall peak
            y_bracket = yt_mm.max() + margin * 0.55
            ax.annotate(
                "", xy=(self.lam * 1e3, y_bracket),
                xytext=(0.0, y_bracket),
                arrowprops=dict(arrowstyle="<->",
                                color=STYLE["wall_top"], lw=1.1),
            )
            ax.text(self.lam * 1e3 / 2.0, y_bracket + margin * 0.18,
                    f"λ = {self.lam * 1e3:.1f} mm",
                    fontsize=8, color=STYLE["wall_top"],
                    ha="center", va="bottom")

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("x  [mm]", fontsize=10)
        ax.set_ylabel("y  [mm]", fontsize=10)

        title = (f"Channel geometry  —  AR = {self.AR:.2f},  "
                 f"φ = {self.phi_deg:.0f}°,  "
                 f"A = {self.A * 1e3:.2f} mm,  "
                 f"λ = {self.lam * 1e3:.1f} mm")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right", fontsize=8, ncol=3,
                  framealpha=0.9, edgecolor="#cccccc",
                  labelcolor=STYLE["text"])
        ax.grid(True, linestyle=":", alpha=0.35, color=STYLE["grid"])
        fig.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

        return fig

    # ------------------------------------------------------------------
    # Convenience properties used by downstream modules
    # ------------------------------------------------------------------

    @property
    def gap(self) -> np.ndarray:
        """Local channel gap h(x) = y_top - y_bottom [m]."""
        return self.y_top - self.y_bottom

    @property
    def arc_length_bottom(self) -> float:
        """Arc length of the bottom wall [m] via trapezoidal rule."""
        dx = np.diff(self.x)
        dy = np.diff(self.y_bottom)
        return float(np.sum(np.sqrt(dx**2 + dy**2)))

    @property
    def arc_length_top(self) -> float:
        """Arc length of the top wall [m] via trapezoidal rule."""
        dx = np.diff(self.x)
        dy = np.diff(self.y_top)
        return float(np.sum(np.sqrt(dx**2 + dy**2)))

    def to_dict(self) -> dict:
        """Return arrays as a dict compatible with older build_channel callers."""
        return {
            "x":          self.x,
            "y_bottom":   self.y_bottom,
            "y_top":      self.y_top,
            "h_local":    self.gap,
            "arc_length": self.arc_length_bottom,
        }

    def __repr__(self) -> str:
        return (f"ChannelGeometry(L={self.L}, H={self.H}, lam={self.lam}, "
                f"AR={self.AR}, phi_deg={self.phi_deg})")


# ---------------------------------------------------------------------------
# Backward-compatible helper (used by mesh, solver, sweep modules)
# ---------------------------------------------------------------------------

def build_channel(H: float, L: float, AR: float,
                  lam: float, phi_deg: float,
                  n_points: int = 300) -> dict:
    """Thin wrapper around ChannelGeometry for backward compatibility.

    Returns the same dict keys as the original implementation:
      x, y_bottom, y_top, h_local, arc_length
    """
    return ChannelGeometry(L, H, lam, AR, phi_deg, n_pts=n_points).to_dict()


# ---------------------------------------------------------------------------
# Test block — generates 20 wavy + 1 straight geometry plots
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    L   = 0.060   # m
    H   = 0.010   # m
    lam = 0.010   # m

    AR_wavy  = [0.05, 0.10, 0.20, 0.30]   # 4 amplitude ratios (excluding 0)
    phi_list = [0, 45, 90, 135, 180]       # 5 phase shifts → 20 wavy cases

    out_dir = Path(__file__).parent.parent / "results" / "geometry"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Straight channel baseline ────────────────────────────────────
    print("Generating straight channel baseline …")
    g0 = ChannelGeometry(L, H, lam, AR=0.0, phi_deg=0)
    min_g, max_g, mean_g = g0.check_validity()
    print(f"  straight  gap: min={min_g*1e3:.3f}  mean={mean_g*1e3:.3f}  "
          f"max={max_g*1e3:.3f} mm")
    g0.plot(save_path=out_dir / "straight.png")
    g0.export_csv(out_dir / "straight.csv")

    # ── 4 AR × 5 phi = 20 wavy geometries ───────────────────────────
    n_ok = 0
    n_invalid = 0
    for AR in AR_wavy:
        for phi in phi_list:
            geom = ChannelGeometry(L, H, lam, AR=AR, phi_deg=phi)
            try:
                min_g, max_g, mean_g = geom.check_validity()
            except ValueError as exc:
                print(f"  INVALID  AR={AR:.2f}  phi={phi:3d}°  →  {exc}")
                n_invalid += 1
                continue

            print(f"  AR={AR:.2f}  phi={phi:3d}°  "
                  f"gap: min={min_g*1e3:.3f}  mean={mean_g*1e3:.3f}  "
                  f"max={max_g*1e3:.3f} mm")

            stem = f"AR{int(AR * 100):03d}_phi{phi:03d}"
            geom.plot(save_path=out_dir / f"{stem}.png")
            geom.export_csv(out_dir / f"{stem}.csv")
            n_ok += 1

    print(f"\n{n_ok} wavy + 1 straight geometries saved to {out_dir}/")
    if n_invalid:
        print(f"{n_invalid} case(s) skipped (self-intersection).")
    sys.exit(0)
