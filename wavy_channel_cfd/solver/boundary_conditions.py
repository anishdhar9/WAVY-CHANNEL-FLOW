"""
solver/boundary_conditions.py

Assembles the boundary-condition dictionary consumed by the SIMPLE solver.

Responsibilities:
  - Compute the inlet velocity magnitude from the Reynolds number, channel
    height, and fluid properties.
  - Package inlet velocity, inlet temperature, wall heat flux, and fluid
    thermal conductivity into a single `bc` dict.
  - Validate that all required parameters are present and physically
    reasonable (positive Re, positive H, non-negative q_flux, etc.).
  - Provide a helper to print a human-readable summary of the applied BCs.
"""

import math


def inlet_velocity(Re: float, H: float, rho: float, mu: float) -> float:
    """Return mean inlet velocity from the hydraulic-diameter Reynolds number.

    U_in = Re * mu / (rho * D_h)   with D_h = 2*H for a wide channel.

    Parameters
    ----------
    Re  : Reynolds number (dimensionless)
    H   : channel height [m]
    rho : fluid density [kg/m³]
    mu  : dynamic viscosity [Pa·s]
    """
    if Re <= 0:
        raise ValueError(f"Reynolds number must be positive, got {Re}")
    if H <= 0:
        raise ValueError(f"Channel height must be positive, got {H}")
    D_h = 2.0 * H
    return Re * mu / (rho * D_h)


def build_bc(Re: float, H: float, q_flux: float,
             fluid: dict, T_in: float = 293.15) -> dict:
    """Build the boundary-condition dictionary for the SIMPLE solver.

    Parameters
    ----------
    Re     : Reynolds number
    H      : channel height [m]
    q_flux : bottom-wall heat flux [W/m²]
    fluid  : dict with keys rho, mu, k_f, Cp
    T_in   : inlet temperature [K] (default 293.15 K = 20 °C)

    Returns
    -------
    bc dict with keys: u_in, T_in, q_flux, k_f, Re
    """
    rho = fluid["rho"]
    mu  = fluid["mu"]
    k_f = fluid["k_f"]

    if q_flux < 0:
        raise ValueError(f"Heat flux must be non-negative, got {q_flux}")

    u_in = inlet_velocity(Re, H, rho, mu)

    return {
        "u_in":  u_in,
        "T_in":  T_in,
        "q_flux": q_flux,
        "k_f":   k_f,
        "Re":    Re,
    }


def summarise_bc(bc: dict) -> None:
    """Print a human-readable summary of the boundary conditions."""
    print("Boundary Conditions:")
    print(f"  Inlet velocity  u_in  = {bc['u_in']:.4f} m/s  (Re = {bc['Re']})")
    print(f"  Inlet temperature     = {bc['T_in']:.2f} K")
    print(f"  Bottom heat flux      = {bc['q_flux']:.1f} W/m²")
    print(f"  Fluid conductivity k  = {bc['k_f']:.3f} W/(m·K)")


def parabolic_inlet_profile(u_mean: float, H: float,
                             y_cc: float) -> float:
    """Return the fully-developed parabolic velocity at wall-normal position y.

    u(y) = 6 * u_mean * y/H * (1 - y/H)

    Parameters
    ----------
    u_mean : mean (bulk) inlet velocity [m/s]
    H      : channel height [m]
    y_cc   : wall-normal coordinate of cell centre [m]
    """
    eta = y_cc / H
    return 6.0 * u_mean * eta * (1.0 - eta)
