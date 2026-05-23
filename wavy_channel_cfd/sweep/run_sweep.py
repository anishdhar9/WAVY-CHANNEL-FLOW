"""
sweep/run_sweep.py

Orchestrates the full parameter sweep over Re, AR, and phi combinations.

Responsibilities:
  - Read the sweep parameter lists (Re_list, AR_list, phi_list) from
    config.yaml (or a supplied config dict).
  - Build a Cartesian product of all parameter combinations.
  - For each combination: build geometry → generate mesh → set BCs →
    run SIMPLE solver → extract metrics.
  - Collect results in a Pandas DataFrame and persist to
    results/sweep_results.csv after every batch (checkpoint strategy).
  - Report progress via tqdm progress bars and print a summary table at
    the end.
  - Optionally run cases in parallel using concurrent.futures.ProcessPoolExecutor
    when `parallel=True` is passed.
"""

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from wavy_channel_cfd.geometry.channel_gen import build_channel
from wavy_channel_cfd.mesh.mesh_gen import generate_mesh
from wavy_channel_cfd.solver.boundary_conditions import build_bc
from wavy_channel_cfd.solver.fluid_properties import get_properties
from wavy_channel_cfd.solver.simple import solve
from wavy_channel_cfd.postprocess.extract_metrics import extract_all

RESULTS_DIR = Path(__file__).parent.parent / "results"
CHECKPOINT_FILE = RESULTS_DIR / "sweep_results.csv"


def _run_single_case(params: dict, config: dict,
                     nx: int = 80, ny: int = 40,
                     solver_params: dict | None = None) -> dict:
    """Run one CFD case and return a flat metrics dict."""
    Re      = params["Re"]
    AR      = params["AR"]
    phi_deg = params["phi"]

    H   = config["H"]
    L   = config["L"]
    lam = config["lambda"]

    fluid   = get_properties(config)
    geom    = build_channel(H, L, AR, lam, phi_deg)
    mesh    = generate_mesh(geom, nx=nx, ny=ny)
    bc      = build_bc(Re, H, config["q_flux"], fluid)

    sp      = solver_params or {"max_iter": 500, "tol": 1e-5}
    u, v, p, T, _monitor = solve(mesh, bc, fluid, sp, verbose=False)

    fields  = {"u": u, "v": v, "p": p, "T": T}
    metrics = extract_all(fields, mesh, fluid, bc, geom)
    metrics.update({"AR": AR, "phi": phi_deg})
    return metrics


def run_sweep(config: dict,
              nx: int = 80, ny: int = 40,
              solver_params: dict | None = None,
              parallel: bool = False,
              checkpoint_every: int = 10) -> pd.DataFrame:
    """Run the full (Re × AR × phi) parameter sweep.

    Parameters
    ----------
    config           : parsed config.yaml dict
    nx, ny           : mesh resolution
    solver_params    : override dict for solver.simple.solve
    parallel         : if True, use ProcessPoolExecutor
    checkpoint_every : save CSV every N completed cases

    Returns
    -------
    DataFrame with one row per case and columns:
      Re, AR, phi, Nu_avg, f, dP, T_wall_max, T_bulk_out
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    Re_list  = config.get("Re_list",  [100, 400])
    AR_list  = config.get("AR_list",  [0.0, 0.10])
    phi_list = config.get("phi_list", [0, 90])

    combos = list(itertools.product(Re_list, AR_list, phi_list))
    records: list[dict] = []

    print(f"Starting sweep: {len(combos)} cases "
          f"({len(Re_list)} Re × {len(AR_list)} AR × {len(phi_list)} phi)")

    if parallel:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        futures = {}
        with ProcessPoolExecutor() as executor:
            for Re, AR, phi in combos:
                p = {"Re": Re, "AR": AR, "phi": phi}
                fut = executor.submit(_run_single_case, p, config,
                                      nx, ny, solver_params)
                futures[fut] = p

            for i, fut in enumerate(tqdm(as_completed(futures),
                                          total=len(combos),
                                          desc="Sweep"), 1):
                records.append(fut.result())
                if i % checkpoint_every == 0:
                    _checkpoint(records)
    else:
        for i, (Re, AR, phi) in enumerate(
                tqdm(combos, desc="Sweep"), 1):
            params = {"Re": Re, "AR": AR, "phi": phi}
            rec = _run_single_case(params, config, nx, ny, solver_params)
            records.append(rec)
            if i % checkpoint_every == 0:
                _checkpoint(records)

    df = pd.DataFrame(records)
    df.to_csv(CHECKPOINT_FILE, index=False)
    print(f"\nSweep complete. Results saved to {CHECKPOINT_FILE}")
    print(df.describe().to_string())
    return df


def _checkpoint(records: list[dict]) -> None:
    """Save intermediate results to CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(CHECKPOINT_FILE, index=False)
