"""
validation/validate_straight.py

Validates the CFD solver against analytical benchmarks for a straight
(AR = 0) parallel-plate channel.

Responsibilities:
  - Run the SIMPLE solver for the straight channel at multiple Re values.
  - Compare the computed friction factor against the laminar Poiseuille
    relation: f = 24 / Re (Fanning, based on D_h = 2H).
  - Compare the fully-developed Nusselt number against the analytical
    value for uniform heat flux: Nu_fd = 8.235 (parallel plates).
  - Report absolute and relative errors, and flag cases that exceed a
    specified tolerance (default 5 %).
  - Generate a validation summary plot saved to results/validation.png.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from wavy_channel_cfd.geometry.channel_gen import build_channel
from wavy_channel_cfd.mesh.mesh_gen import generate_mesh
from wavy_channel_cfd.solver.boundary_conditions import build_bc
from wavy_channel_cfd.solver.fluid_properties import get_properties
from wavy_channel_cfd.solver.simple import solve
from wavy_channel_cfd.postprocess.extract_metrics import extract_all

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Analytical references (parallel-plate laminar)
NU_FD_ANALYTICAL   = 8.235   # uniform heat flux, fully developed
F_RE_ANALYTICAL    = 24.0    # Fanning: f * Re = 24  (D_h = 2H)


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
      f_computed, f_analytical, f_err_pct, passed
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    Re_list = config.get("Re_list", [100, 200, 400])
    H       = config["H"]
    L       = config["L"]
    lam     = config["lambda"]
    q_flux  = config["q_flux"]
    fluid   = get_properties(config)

    records = []
    for Re in Re_list:
        geom = build_channel(H, L, AR=0.0, lam=lam, phi_deg=0.0, n_points=500)
        mesh = generate_mesh(geom, nx=nx, ny=ny)
        bc   = build_bc(Re, H, q_flux, fluid)
        sp   = {"max_iter": 1000, "tol": 1e-6}

        u, v, p, T, _ = solve(mesh, bc, fluid, sp, verbose=False)
        fields  = {"u": u, "v": v, "p": p, "T": T}
        metrics = extract_all(fields, mesh, fluid, bc, geom)

        Nu_c = metrics["Nu_avg"]
        f_c  = metrics["f"]
        Nu_a = NU_FD_ANALYTICAL
        f_a  = F_RE_ANALYTICAL / Re

        Nu_err = abs(Nu_c - Nu_a) / Nu_a * 100.0
        f_err  = abs(f_c  - f_a)  / f_a  * 100.0
        passed = (Nu_err < tolerance * 100) and (f_err < tolerance * 100)

        print(f"  Re={Re:5d}  Nu={Nu_c:.3f} (err={Nu_err:.1f}%)  "
              f"f={f_c:.5f} (err={f_err:.1f}%)  {'PASS' if passed else 'FAIL'}")

        records.append({
            "Re": Re,
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
    ax.plot(df["Re"], df["Nu_computed"],  "o-", label="CFD")
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
