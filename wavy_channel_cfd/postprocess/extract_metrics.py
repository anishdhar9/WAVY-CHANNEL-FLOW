"""
postprocess/extract_metrics.py

Extracts engineering performance metrics from converged flow and
temperature fields produced by the SIMPLE solver.

Responsibilities:
  - Compute the local and average Nusselt number (Nu) along the heated wall
    using the bulk temperature and the wall heat flux, referenced to the
    *local* hydraulic diameter (the wavy channel's gap varies with x).
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
                  q_flux: float, k_f: float,
                  Dh_local: np.ndarray) -> np.ndarray:
    """Return the local Nusselt number profile along the heated bottom wall.

    Nu(x) = q_flux * Dh_local(x) / (k_f * (T_wall(x) - T_bulk(x)))

    Parameters
    ----------
    T_wall   : (nx,) wall temperature array [K]
    T_bulk   : (nx,) bulk temperature array [K]
    q_flux   : bottom-wall heat flux [W/m²]
    k_f      : fluid thermal conductivity [W/(m·K)]
    Dh_local : (nx,) LOCAL hydraulic diameter Dh(x) = 2 * gap(x) [m] — the
               wavy channel's wall-to-wall gap varies with x, so a single
               scalar Dh (as used for the domain length or the mean height)
               is not the right length scale at each streamwise station.
    """
    dT = np.where(T_wall - T_bulk > 1e-6, T_wall - T_bulk, 1e-6)
    return q_flux * Dh_local / (k_f * dT)


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
    H      : mean channel height [m]  (Dh = 2*H for parallel plates)
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


def extract_all(fields: dict, mesh, fluid, bc, geom) -> dict:
    """Convenience wrapper: compute all metrics and return a flat dict.

    Parameters
    ----------
    fields : dict with keys u, v, p, T (2-D (ny,nx) arrays from solver)
    mesh   : a mesh.mesh_gen.StructuredMesh instance
    fluid  : a solver.fluid_properties.FluidProperties instance
    bc     : a solver.boundary_conditions.BoundaryConditions instance
    geom   : a geometry.channel_gen.ChannelGeometry instance

    Returns
    -------
    metrics dict with keys: Re, Nu_avg, f, dP, T_wall_max, T_bulk_out
    """
    u = fields["u"]
    T = fields["T"]
    p = fields["p"]

    # Cell wall-normal height, (ny,nx) — averaged west/east face length,
    # matching the convention used elsewhere in the legacy mesh dict.
    dy = 0.5 * (mesh.face_len_west + mesh.face_len_east).T

    T_bulk = bulk_temperature(T, u, dy)
    T_wall = T[0, :]  # bottom-wall cell centres

    x_stations = mesh.Xc[:, 0]                     # (nx,) streamwise stations
    Dh_local = 2.0 * np.interp(x_stations, geom.x, geom.gap)

    nu_loc = local_nusselt(T_wall, T_bulk, bc.q_flux, fluid.k_f, Dh_local)
    Nu_avg = average_nusselt(nu_loc)
    f = friction_factor(p, fluid.rho, bc.u_inlet, geom.L, geom.H)

    return {
        "Re":         bc.Re,
        "Nu_avg":     Nu_avg,
        "f":          f,
        "dP":         float(np.mean(p[:, 0]) - np.mean(p[:, -1])),
        "T_wall_max": float(np.max(T_wall)),
        "T_bulk_out": float(T_bulk[-1]),
    }
