"""
postprocess/plot_results.py

Generates publication-quality figures from CFD results.

Responsibilities:
  - Plot 2-D contour maps of velocity magnitude, pressure, and temperature
    on the physical (body-fitted) mesh.
  - Plot streamwise profiles of local Nusselt number and bulk temperature.
  - Plot Nu vs Re and friction factor vs Re for multiple AR / phi cases,
    with error bars and reference Dittus–Boelter or Shah–London correlations.
  - Save figures to the results/ directory in PNG and PDF formats.
  - All plotting functions accept an optional `ax` argument so they can be
    embedded in multi-panel figures.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from wavy_channel_cfd.geometry.channel_gen import STYLE, apply_axes_style

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def contour_field(field: np.ndarray, mesh: dict,
                  title: str = "", label: str = "",
                  filename: str | None = None,
                  ax: plt.Axes | None = None) -> plt.Figure:
    """Plot a filled-contour map of a 2-D cell-centred field.

    Parameters
    ----------
    field    : (ny, nx) 2-D array to visualise
    mesh     : mesh dict from mesh.mesh_gen.generate_mesh
    title    : figure title
    label    : colorbar label
    filename : save path relative to results/ (no extension); None → no save
    ax       : existing Axes to draw into; created if None
    """
    _ensure_results_dir()
    fig, ax_ = (None, ax) if ax is not None else plt.subplots(figsize=(10, 3))
    if fig is None:
        fig = ax_.get_figure()

    apply_axes_style(ax_)
    cf = ax_.contourf(mesh["cc_x"], mesh["cc_y"], field, levels=50,
                      cmap="turbo")
    fig.colorbar(cf, ax=ax_, label=label)
    ax_.set_xlabel("x [m]")
    ax_.set_ylabel("y [m]")
    ax_.set_title(title)
    ax_.set_aspect("equal")

    if filename:
        _save(fig, filename)
    return fig


def plot_nusselt_profile(x: np.ndarray, Nu: np.ndarray,
                          label: str = "",
                          filename: str | None = None,
                          ax: plt.Axes | None = None) -> plt.Figure:
    """Plot the streamwise local Nusselt number distribution."""
    _ensure_results_dir()
    fig, ax_ = (None, ax) if ax is not None else plt.subplots(figsize=(7, 4))
    if fig is None:
        fig = ax_.get_figure()

    apply_axes_style(ax_)
    ax_.plot(x, Nu, label=label)
    ax_.set_xlabel("x [m]")
    ax_.set_ylabel("Nu(x)")
    ax_.set_title("Local Nusselt Number")
    ax_.legend()
    ax_.grid(True, linestyle="--", alpha=0.35, color=STYLE["grid"])

    if filename:
        _save(fig, filename)
    return fig


def plot_nu_vs_re(re_list: list[float],
                  nu_dict: dict[str, list[float]],
                  filename: str | None = "Nu_vs_Re",
                  ax: plt.Axes | None = None) -> plt.Figure:
    """Plot average Nu vs Re for multiple geometry cases.

    Parameters
    ----------
    re_list  : list of Reynolds numbers
    nu_dict  : {label: [Nu_avg per Re]} mapping
    filename : output filename stem (saved to results/)
    """
    _ensure_results_dir()
    fig, ax_ = (None, ax) if ax is not None else plt.subplots(figsize=(7, 5))
    if fig is None:
        fig = ax_.get_figure()

    apply_axes_style(ax_)
    for lbl, nu_vals in nu_dict.items():
        ax_.plot(re_list, nu_vals, marker="o", label=lbl)

    ax_.set_xlabel("Re")
    ax_.set_ylabel(r"$\overline{Nu}$")
    ax_.set_title("Average Nusselt Number vs Reynolds Number")
    ax_.legend(fontsize=8)
    ax_.grid(True, linestyle="--", alpha=0.35, color=STYLE["grid"])

    if filename:
        _save(fig, filename)
    return fig


def plot_friction_vs_re(re_list: list[float],
                         f_dict: dict[str, list[float]],
                         filename: str | None = "f_vs_Re",
                         ax: plt.Axes | None = None) -> plt.Figure:
    """Plot Fanning friction factor vs Re for multiple geometry cases."""
    _ensure_results_dir()
    fig, ax_ = (None, ax) if ax is not None else plt.subplots(figsize=(7, 5))
    if fig is None:
        fig = ax_.get_figure()

    apply_axes_style(ax_)
    for lbl, f_vals in f_dict.items():
        ax_.semilogy(re_list, f_vals, marker="s", label=lbl)

    ax_.set_xlabel("Re")
    ax_.set_ylabel("Fanning friction factor f")
    ax_.set_title("Friction Factor vs Reynolds Number")
    ax_.legend(fontsize=8)
    ax_.grid(True, which="both", linestyle="--", alpha=0.35,
              color=STYLE["grid"])

    if filename:
        _save(fig, filename)
    return fig


def plot_tpf_heatmap(ar_list: list[float],
                      phi_list: list[float],
                      tpf_matrix: np.ndarray,
                      Re: float,
                      filename: str | None = None) -> plt.Figure:
    """Plot a heatmap of TPF over the (AR, phi) parameter space at fixed Re."""
    _ensure_results_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    apply_axes_style(ax)
    im = ax.imshow(tpf_matrix, aspect="auto", origin="lower",
                   extent=[phi_list[0], phi_list[-1], ar_list[0], ar_list[-1]],
                   cmap="RdBu_r", vmin=0.8, vmax=1.5)
    fig.colorbar(im, ax=ax, label="TPF")
    ax.set_xlabel("Phase φ [°]")
    ax.set_ylabel("Amplitude ratio AR")
    ax.set_title(f"Thermal Performance Factor  (Re = {Re:.0f})")

    fname = filename or f"TPF_heatmap_Re{Re:.0f}"
    _save(fig, fname)
    return fig


def _save(fig: plt.Figure, stem: str) -> None:
    """Save figure as PNG and PDF to results/."""
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS_DIR / f"{stem}.{ext}",
                    dpi=200, bbox_inches="tight")
