"""
postprocess/extract_metrics.py

Extracts engineering performance metrics from converged flow and
temperature fields produced by the SIMPLE solver.

Responsibilities:
  - Compute the local and average Nusselt number (Nu) along the heated wall
    using the bulk temperature and the wall heat flux.
  - Compute the Fanning friction factor (f) and the overall pressure drop
    (dP) between inlet and outlet.
  - Evaluate the thermal performance factor (TPF) and performance evaluation
    criterion (PEC) that compare a wavy channel against the straight baseline.
  - Return results as a flat dict suitable for writing to CSV / Pandas.
"""

import numpy as np


def bulk_temperature(T: np.ndarray, u: np.ndarray,
                     dy: np.ndarray) -> np.ndarray:
    """Return the streamwise profile of the bulk (mixing-cup) temperature.

    T_b(x) = ∫ u·T dy  /  ∫ u dy   (integration over y at each x-column)

    Parameters
    ----------
    T  : (ny, nx) temperature field [K]
    u  : (ny, nx) streamwise velocity field [m/s]
    dy : (ny, nx) cell heights [m]
    """
    num = np.sum(u * T * dy, axis=0)
    den = np.sum(u * dy,     axis=0)
    return num / np.where(np.abs(den) > 1e-30, den, 1e-30)


def local_nusselt(T_wall: np.ndarray, T_bulk: np.ndarray,
                  q_flux: float, k_f: float, H: float) -> np.ndarray:
    """Return the local Nusselt number profile along the heated bottom wall.

    Nu(x) = q_flux * D_h / (k_f * (T_wall - T_bulk))

    Parameters
    ----------
    T_wall : (nx,) wall temperature array [K]
    T_bulk : (nx,) bulk temperature array [K]
    q_flux : bottom-wall heat flux [W/m²]
    k_f    : fluid thermal conductivity [W/(m·K)]
    H      : channel height for D_h = 2*H [m]
    """
    D_h = 2.0 * H
    dT  = np.where(T_wall - T_bulk > 1e-6, T_wall - T_bulk, 1e-6)
    return q_flux * D_h / (k_f * dT)


def average_nusselt(nu_local: np.ndarray) -> float:
    """Return the length-averaged Nusselt number."""
    return float(np.mean(nu_local))


def friction_factor(p: np.ndarray, rho: float, u_mean: float,
                    L: float, H: float) -> float:
    """Fanning friction factor from the mean pressure drop.

    f = (dP / L) * D_h / (2 * rho * U_mean²)

    Parameters
    ----------
    p      : (ny, nx) pressure field [Pa]
    rho    : fluid density [kg/m³]
    u_mean : mean inlet velocity [m/s]
    L      : channel length [m]
    H      : channel height [m]
    """
    p_in  = float(np.mean(p[:, 0]))
    p_out = float(np.mean(p[:, -1]))
    dP    = p_in - p_out
    D_h   = 2.0 * H
    denom = 2.0 * rho * u_mean**2
    if denom < 1e-30:
        return float("nan")
    return (dP / L) * D_h / denom


def thermal_performance_factor(Nu_wavy: float, Nu_straight: float,
                                f_wavy: float,  f_straight: float) -> float:
    """Thermal Performance Factor (TPF).

    TPF = (Nu_wavy / Nu_straight) / (f_wavy / f_straight)^(1/3)
    """
    if f_straight <= 0 or Nu_straight <= 0:
        return float("nan")
    return (Nu_wavy / Nu_straight) / (f_wavy / f_straight) ** (1.0 / 3.0)


def extract_all(fields: dict, mesh: dict, fluid: dict,
                bc: dict, geometry: dict) -> dict:
    """Convenience wrapper: compute all metrics and return a flat dict.

    Parameters
    ----------
    fields   : dict with keys u, v, p, T (2-D arrays from solver)
    mesh     : dict from mesh.mesh_gen.generate_mesh
    fluid    : dict from solver.fluid_properties.get_properties
    bc       : dict from solver.boundary_conditions.build_bc
    geometry : dict from geometry.channel_gen.build_channel

    Returns
    -------
    metrics dict with keys: Nu_avg, f, dP, T_wall_max, T_bulk_out, Re
    """
    u   = fields["u"]
    T   = fields["T"]
    p   = fields["p"]
    dy  = mesh["cell_dy"]

    T_bulk = bulk_temperature(T, u, dy)
    T_wall = T[0, :]  # bottom-wall cell centres

    nu_loc = local_nusselt(T_wall, T_bulk, bc["q_flux"], fluid["k_f"],
                            geometry["x"][-1] - geometry["x"][0])  # use H equiv
    Nu_avg = average_nusselt(nu_loc)
    f      = friction_factor(p, fluid["rho"], bc["u_in"],
                              geometry["x"][-1], geometry["x"][-1])

    return {
        "Re":        bc["Re"],
        "Nu_avg":    Nu_avg,
        "f":         f,
        "dP":        float(np.mean(p[:, 0]) - np.mean(p[:, -1])),
        "T_wall_max": float(np.max(T_wall)),
        "T_bulk_out": float(T_bulk[-1]),
    }
