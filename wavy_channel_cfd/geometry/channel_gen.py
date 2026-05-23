"""
geometry/channel_gen.py

Generates the wavy-channel geometry.

Responsibilities:
  - Parametrise the sinusoidal bottom wall profile given amplitude ratio (AR),
    wavelength (lambda), channel height (H), length (L), and phase offset (phi).
  - Return arrays of (x, y_bottom, y_top) that define the channel walls.
  - Provide helpers to compute the local cross-section height and the arc length
    of the corrugated wall, used downstream by the mesh and post-processing
    modules.
"""

import numpy as np


def bottom_wall(x: np.ndarray, H: float, AR: float,
                lam: float, phi_deg: float) -> np.ndarray:
    """Return y-coordinates of the sinusoidal bottom wall.

    The wall shape is:  y_b(x) = AR * H * sin(2*pi*x/lam + phi)

    Parameters
    ----------
    x       : 1-D array of x-positions [m]
    H       : channel height [m]
    AR      : amplitude ratio (a/H), dimensionless
    lam     : wavelength [m]
    phi_deg : phase shift [degrees]
    """
    phi = np.deg2rad(phi_deg)
    return AR * H * np.sin(2.0 * np.pi * x / lam + phi)


def top_wall(x: np.ndarray, H: float) -> np.ndarray:
    """Return y-coordinates of the flat top wall (y = H)."""
    return np.full_like(x, H)


def local_height(x: np.ndarray, H: float, AR: float,
                 lam: float, phi_deg: float) -> np.ndarray:
    """Return local gap height h(x) = y_top - y_bottom."""
    y_b = bottom_wall(x, H, AR, lam, phi_deg)
    y_t = top_wall(x, H)
    return y_t - y_b


def wall_arc_length(x: np.ndarray, H: float, AR: float,
                    lam: float, phi_deg: float) -> float:
    """Approximate arc length of the bottom wall by trapezoidal integration."""
    y_b = bottom_wall(x, H, AR, lam, phi_deg)
    dx = np.diff(x)
    dy = np.diff(y_b)
    return float(np.sum(np.sqrt(dx**2 + dy**2)))


def build_channel(H: float, L: float, AR: float,
                  lam: float, phi_deg: float,
                  n_points: int = 500) -> dict:
    """Build and return a dictionary describing the channel walls.

    Returns
    -------
    dict with keys: x, y_bottom, y_top, h_local, arc_length
    """
    x = np.linspace(0.0, L, n_points)
    y_b = bottom_wall(x, H, AR, lam, phi_deg)
    y_t = top_wall(x, H)
    h   = y_t - y_b
    s   = wall_arc_length(x, H, AR, lam, phi_deg)
    return {"x": x, "y_bottom": y_b, "y_top": y_t,
            "h_local": h, "arc_length": s}
