"""
validation/validate_straight.py

Validates the CFD solver against analytical benchmarks for a straight
(AR = 0) parallel-plate channel.

Responsibilities:
  - Run the SIMPLE solver for the straight channel at multiple Re values.
  - Compare the computed friction factor against the laminar Poiseuille
    relation: f = 24 / Re (Fanning, based on D_h = 2H).
  - Compare the *outlet-window* Nusselt number (last 10% of the channel)
    against the analytical fully-developed value for this project's actual
    thermal BCs: one wall at constant heat flux, the other adiabatic.
  - Report absolute and relative errors, and flag cases that exceed a
    specified tolerance (default 5 %).
  - Generate a validation summary plot saved to results/validation.png.

Entrance-length note
---------------------
config.yaml's production channel (L=0.06 m, Dh=0.02 m -> L/Dh=3) is far
shorter than the thermal entry length at the higher Re values in its
Re_list (Lt ~ 0.05*Re*Pr*Dh, which already exceeds L by 10x+ at Re=100 for
water, Pr~7). Comparing a domain-averaged Nu against a fully-developed
reference over that short a domain is not meaningful — entrance-region Nu
is always higher than the fully-developed value, and averaging over the
whole (mostly-undeveloped) domain would bias the comparison regardless of
solver correctness. This validator therefore:
  (a) uses its own validation-only channel length, long enough for the
      flow to approach thermal development at each Re being checked
      (decoupled from config.yaml's production L, which stays as configured
      for the sweep), and
  (b) compares Nu averaged over only the last 10% of that length (near the
      outlet), where the flow has actually had a chance to develop.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from wavy_channel_cfd.geometry.channel_gen import ChannelGeometry
from wavy_channel_cfd.mesh.mesh_gen import StructuredMesh
from wavy_channel_cfd.solver.boundary_conditions import BoundaryConditions
from wavy_channel_cfd.solver.fluid_properties import FluidProperties
from wavy_channel_cfd.solver.simple import solve
from wavy_channel_cfd.postprocess.extract_metrics import (
    bulk_temperature, local_nusselt, friction_factor)

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Analytical references (parallel-plate laminar, fully developed).
# NU_FD_ANALYTICAL is the ASYMMETRIC-heating case (one wall at constant
# heat flux, the other adiabatic) matching this project's actual boundary
# conditions — NOT the symmetric two-wall-heated case (whose fully
# developed value is 8.235). Reference: Incropera et al., Fundamentals of
# Heat and Mass Transfer, Table 8.1 ("one surface insulated, other at
# uniform heat flux"); consistent with Shah & London, Laminar Flow Forced
# Convection in Ducts.
NU_FD_ANALYTICAL = 5.385
F_RE_ANALYTICAL  = 24.0    # Fanning: f * Re = 24  (D_h = 2H)

OUTLET_WINDOW_FRACTION = 0.10   # compare Nu averaged over the last 10% of L


def _validation_length(Re: float, Pr: float, Dh: float,
                       L_min: float, margin: float = 1.5) -> float:
    """Validation-only channel length, long enough to approach thermal
    development at this Re (Lt ~ 0.05*Re*Pr*Dh), with a safety margin.
    Never shorter than L_min (the production config's own length).
    """
    Lt = 0.05 * Re * Pr * Dh
    return max(L_min, margin * Lt)


def run_validation(config: dict,
                   nx: int = 120, ny: int = 60,
                   tolerance: float = 0.05) -> pd.DataFrame:
    """Run straight-channel validation for all Re in config['Re_list'].

    Parameters
    ----------
    config    : parsed config.yaml dict
    nx, ny    : mesh resolution
    tolerance : acceptable relative error threshold (default 5 %)

    Returns
    -------
    DataFrame with columns:
      Re, Nu_computed, Nu_analytical, Nu_err_pct,
      f_computed, f_analytical, f_err_pct, L_validation, passed
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    Re_list = config.get("Re_list", [100, 200, 400])
    H       = config["H"]
    L_min   = config["L"]
    lam     = config["lambda"]
    q_flux  = config["q_flux"]
    fluid   = FluidProperties(rho=config["rho"], mu=config["mu"],
                              k_f=config["k_f"], Cp=config["Cp"])
    Dh      = 2.0 * H

    records = []
    for Re in Re_list:
        L = _validation_length(Re, fluid.Pr, Dh, L_min)
        geom = ChannelGeometry(L=L, H=H, lam=lam, AR=0.0, phi_deg=0.0, n_pts=500)
        mesh = StructuredMesh.for_yplus(geom, Re=Re, rho=fluid.rho, mu=fluid.mu,
                                        y_plus=1.0, nx=nx, ny=ny)
        bc   = BoundaryConditions(fluid, Re=Re, q_flux=q_flux, Dh=Dh)
        sp   = {"max_iter": 1000, "tol": 1e-6}

        u, v, p, T = solve(mesh, bc, fluid, sp, verbose=False)[:4]

        dy = 0.5 * (mesh.face_len_west + mesh.face_len_east).T
        T_bulk = bulk_temperature(T, u, dy)
        T_wall = T[0, :]
        x_stations = mesh.Xc[:, 0]
        Dh_local = 2.0 * np.interp(x_stations, geom.x, geom.gap)
        nu_loc = local_nusselt(T_wall, T_bulk, q_flux, fluid.k_f, Dh_local)

        window = x_stations >= (1.0 - OUTLET_WINDOW_FRACTION) * L
        Nu_c = float(np.mean(nu_loc[window]))
        f_c  = friction_factor(p, fluid.rho, bc.u_inlet, L, H)

        Nu_a = NU_FD_ANALYTICAL
        f_a  = F_RE_ANALYTICAL / Re

        Nu_err = abs(Nu_c - Nu_a) / Nu_a * 100.0
        f_err  = abs(f_c  - f_a)  / f_a  * 100.0
        passed = (Nu_err < tolerance * 100) and (f_err < tolerance * 100)

        print(f"  Re={Re:5d}  L={L*1e3:.1f}mm  "
              f"Nu={Nu_c:.3f} (err={Nu_err:.1f}%)  "
              f"f={f_c:.5f} (err={f_err:.1f}%)  {'PASS' if passed else 'FAIL'}")

        records.append({
            "Re": Re, "L_validation": L,
            "Nu_computed": Nu_c, "Nu_analytical": Nu_a, "Nu_err_pct": Nu_err,
            "f_computed":  f_c,  "f_analytical":  f_a,  "f_err_pct":  f_err,
            "passed": passed,
        })

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "validation_results.csv", index=False)
    _plot_validation(df)
    return df


def _plot_validation(df: pd.DataFrame) -> None:
    """Save a two-panel validation plot (Nu and f vs Re)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(df["Re"], df["Nu_computed"],  "o-", label="CFD (outlet window)")
    ax.axhline(NU_FD_ANALYTICAL, color="r", linestyle="--",
               label=f"Analytical ({NU_FD_ANALYTICAL})")
    ax.set_xlabel("Re");  ax.set_ylabel("Nu")
    ax.set_title("Nusselt Number Validation")
    ax.legend();  ax.grid(True, linestyle="--", alpha=0.5)

    ax = axes[1]
    ax.plot(df["Re"], df["f_computed"],  "s-", label="CFD")
    f_analytical = F_RE_ANALYTICAL / df["Re"]
    ax.plot(df["Re"], f_analytical,  "r--", label="f = 24/Re")
    ax.set_xlabel("Re");  ax.set_ylabel("Fanning f")
    ax.set_title("Friction Factor Validation")
    ax.legend();  ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "validation.png", dpi=150, bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "validation.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Validation plot saved to {RESULTS_DIR}/validation.png")
