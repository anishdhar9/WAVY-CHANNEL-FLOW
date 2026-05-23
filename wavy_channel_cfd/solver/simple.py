"""
solver/simple.py

SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) solver for
steady, incompressible, laminar flow with conjugate heat transfer in the
wavy channel.

Responsibilities:
  - Implement the pressure-velocity coupling loop (predictor → pressure
    correction → corrector) on the structured mesh.
  - Discretise the momentum and energy equations with a second-order
    upwind/central scheme on collocated or staggered cell-centred variables.
  - Accept boundary conditions from solver/boundary_conditions.py and fluid
    properties from solver/fluid_properties.py.
  - Iterate until residuals of u, v, p, and T drop below specified tolerances.
  - Return the converged field arrays (u, v, p, T) and a residual history dict.
"""

import numpy as np
from typing import Callable


# ---------------------------------------------------------------------------
# Residual tracking helper
# ---------------------------------------------------------------------------

class ResidualMonitor:
    """Accumulates and reports per-iteration residuals for each field."""

    def __init__(self, fields: list[str]):
        self.history: dict[str, list[float]] = {f: [] for f in fields}

    def record(self, field: str, residual: float) -> None:
        self.history[field].append(residual)

    def latest(self) -> dict[str, float]:
        return {f: vals[-1] if vals else float("nan")
                for f, vals in self.history.items()}

    def converged(self, tol: float) -> bool:
        return all(v[-1] < tol for v in self.history.values() if v)


# ---------------------------------------------------------------------------
# Core SIMPLE iteration
# ---------------------------------------------------------------------------

def solve(mesh: dict,
          bc: dict,
          fluid: dict,
          params: dict,
          verbose: bool = True) -> tuple[np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray,
                                          ResidualMonitor]:
    """Run the SIMPLE loop and return converged flow and temperature fields.

    Parameters
    ----------
    mesh   : dict from mesh.mesh_gen.generate_mesh
    bc     : dict from solver.boundary_conditions.build_bc
    fluid  : dict from solver.fluid_properties.get_properties
    params : solver control parameters:
               max_iter  – maximum outer iterations (default 2000)
               tol       – convergence tolerance (default 1e-6)
               alpha_u   – under-relaxation for velocity (default 0.7)
               alpha_p   – under-relaxation for pressure (default 0.3)
               alpha_T   – under-relaxation for temperature (default 0.9)
    verbose: print residuals every 100 iterations

    Returns
    -------
    u, v, p, T  : 2-D field arrays of shape (ny, nx)
    monitor     : ResidualMonitor with iteration history
    """
    nx, ny = mesh["nx"], mesh["nx"]
    ny     = mesh["ny"]
    nx     = mesh["nx"]

    max_iter = params.get("max_iter", 2000)
    tol      = params.get("tol",      1e-6)
    alpha_u  = params.get("alpha_u",  0.7)
    alpha_p  = params.get("alpha_p",  0.3)
    alpha_T  = params.get("alpha_T",  0.9)

    rho = fluid["rho"]
    mu  = fluid["mu"]
    k_f = fluid["k_f"]
    Cp  = fluid["Cp"]

    dV  = mesh["cell_volume"]
    dx  = mesh["cell_dx"]
    dy  = mesh["cell_dy"]

    # Initialise fields
    u  = np.zeros((ny, nx))
    v  = np.zeros((ny, nx))
    p  = np.zeros((ny, nx))
    T  = np.full((ny, nx), bc.get("T_in", 293.15))

    monitor = ResidualMonitor(["u", "v", "p", "T"])

    for iteration in range(1, max_iter + 1):
        u_old, v_old, p_old, T_old = u.copy(), v.copy(), p.copy(), T.copy()

        # --- Momentum predictor (placeholder: diffusion-only stencil) ---
        u_star, v_star = _momentum_predictor(
            u, v, p, rho, mu, dx, dy, dV, bc, alpha_u)

        # --- Pressure correction ---
        p_prime = _pressure_correction(u_star, v_star, rho, dx, dy, dV, bc)

        # --- Velocity / pressure corrector ---
        u, v, p = _apply_correction(
            u_star, v_star, p, p_prime, rho, dx, dy, alpha_u, alpha_p)

        # --- Energy equation ---
        T = _energy_equation(u, v, T, rho, Cp, k_f, dx, dy, dV, bc, alpha_T)

        # --- Apply boundary conditions ---
        u, v, p, T = _apply_bc(u, v, p, T, bc, mesh)

        # --- Residuals ---
        monitor.record("u", float(np.mean(np.abs(u - u_old))))
        monitor.record("v", float(np.mean(np.abs(v - v_old))))
        monitor.record("p", float(np.mean(np.abs(p - p_old))))
        monitor.record("T", float(np.mean(np.abs(T - T_old))))

        if verbose and iteration % 100 == 0:
            r = monitor.latest()
            print(f"  iter {iteration:4d} | "
                  f"u={r['u']:.2e} v={r['v']:.2e} "
                  f"p={r['p']:.2e} T={r['T']:.2e}")

        if monitor.converged(tol):
            if verbose:
                print(f"  Converged at iteration {iteration}.")
            break

    return u, v, p, T, monitor


# ---------------------------------------------------------------------------
# Private helpers (simplified stencils — replace with full FV discretisation)
# ---------------------------------------------------------------------------

def _momentum_predictor(u, v, p, rho, mu, dx, dy, dV, bc, alpha):
    """Diffusion-dominated momentum predictor (Jacobi sweep)."""
    u_star = u.copy()
    v_star = v.copy()

    # Interior cells only; boundary rows handled in _apply_bc
    u_star[1:-1, 1:-1] = (
        u[1:-1, :-2] + u[1:-1, 2:]
        + u[:-2, 1:-1] + u[2:, 1:-1]
    ) / 4.0
    v_star[1:-1, 1:-1] = (
        v[1:-1, :-2] + v[1:-1, 2:]
        + v[:-2, 1:-1] + v[2:, 1:-1]
    ) / 4.0
    return alpha * u_star + (1 - alpha) * u, alpha * v_star + (1 - alpha) * v


def _pressure_correction(u_star, v_star, rho, dx, dy, dV, bc):
    """Solve pressure-Poisson equation for p' (Jacobi, one sweep)."""
    ny, nx = u_star.shape
    p_prime = np.zeros((ny, nx))
    div = (np.roll(u_star, -1, axis=1) - u_star) / np.maximum(dx, 1e-12) \
        + (np.roll(v_star, -1, axis=0) - v_star) / np.maximum(dy, 1e-12)
    p_prime = -rho * div * dx * dy / (2.0 * (dx + dy + 1e-30))
    return p_prime


def _apply_correction(u_star, v_star, p, p_prime, rho, dx, dy, alpha_u, alpha_p):
    """Correct velocities and update pressure."""
    u_new = u_star - (np.roll(p_prime, -1, axis=1) - p_prime) / (rho * np.maximum(dx, 1e-12))
    v_new = v_star - (np.roll(p_prime, -1, axis=0) - p_prime) / (rho * np.maximum(dy, 1e-12))
    p_new = p + alpha_p * p_prime
    return alpha_u * u_new + (1 - alpha_u) * u_star, \
           alpha_u * v_new + (1 - alpha_u) * v_star, \
           p_new


def _energy_equation(u, v, T, rho, Cp, k_f, dx, dy, dV, bc, alpha):
    """Convection-diffusion energy equation (Jacobi sweep)."""
    T_new = T.copy()
    alpha_diff = k_f / (rho * Cp)
    T_new[1:-1, 1:-1] = (
        alpha_diff * (
            (T[1:-1, 2:] + T[1:-1, :-2]) / dx[1:-1, 1:-1]**2
            + (T[2:, 1:-1] + T[:-2, 1:-1]) / dy[1:-1, 1:-1]**2
        )
    ) / (
        2.0 * alpha_diff * (1.0 / dx[1:-1, 1:-1]**2
                            + 1.0 / dy[1:-1, 1:-1]**2) + 1e-30
    )
    return alpha * T_new + (1 - alpha) * T


def _apply_bc(u, v, p, T, bc, mesh):
    """Enforce boundary conditions on all four channel faces."""
    # Inlet (left): uniform velocity profile, fixed temperature
    u[:, 0]  = bc.get("u_in", 0.0)
    v[:, 0]  = 0.0
    T[:, 0]  = bc.get("T_in", 293.15)
    p[:, 0]  = p[:, 1]  # zero-gradient

    # Outlet (right): zero-gradient for u, v, T; fixed reference pressure
    u[:, -1] = u[:, -2]
    v[:, -1] = v[:, -2]
    T[:, -1] = T[:, -2]
    p[:, -1] = 0.0

    # Top wall: no-slip, adiabatic
    u[-1, :] = 0.0
    v[-1, :] = 0.0
    T[-1, :] = T[-2, :]

    # Bottom wall: no-slip, constant heat flux q_flux
    u[0, :] = 0.0
    v[0, :] = 0.0
    q_flux = bc.get("q_flux", 0.0)
    k_f    = bc.get("k_f",   0.6)
    dy0    = mesh["cell_dy"][0, :]
    T[0, :] = T[1, :] + q_flux * dy0 / k_f

    return u, v, p, T
