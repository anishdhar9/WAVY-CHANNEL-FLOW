"""
solver/boundary_conditions.py

Boundary condition application for the 2-D wavy-channel FV solver.

Field array convention (used by every method in this module):
    shape (ny, nx)
    axis-0 (rows)    → wall-normal direction j  (j=0 bottom wall, j=ny-1 top wall)
    axis-1 (columns) → streamwise direction  i  (i=0 inlet,       i=nx-1 outlet)

This layout makes the four boundary stencils read as:
    inlet       : u[:,  0],  zero-index column
    outlet      : u[:, -1],  last column
    bottom wall : u[0,  :],  zero-index row
    top wall    : u[-1, :],  last row

All methods operate in-place on the NumPy arrays passed as arguments.
No arrays are allocated; no Python loops over cells.
"""

from __future__ import annotations
import numpy as np

from wavy_channel_cfd.solver.fluid_properties import FluidProperties


class BoundaryConditions:
    """Encapsulates and applies all boundary conditions for one CFD case.

    Parameters
    ----------
    fluid  : FluidProperties instance
    Re     : Reynolds number  Re = ρ U Dh / μ  (must be > 0)
    q_flux : bottom-wall heat flux [W/m²]  (must be ≥ 0)
    Dh     : hydraulic diameter [m]  (Dh = 2 H for wide parallel plates)

    Attributes
    ----------
    u_inlet : computed mean inlet velocity [m/s]
    T_INLET : fixed inlet temperature = 300.0 K (class constant)
    """

    T_INLET: float = 300.0   # K — fixed by project specification

    def __init__(self, fluid: FluidProperties,
                 Re: float, q_flux: float, Dh: float) -> None:
        if Re <= 0:
            raise ValueError(f"Re must be positive, got {Re!r}")
        if q_flux < 0:
            raise ValueError(f"q_flux must be non-negative, got {q_flux!r}")
        if Dh <= 0:
            raise ValueError(f"Dh must be positive, got {Dh!r}")

        self.fluid   = fluid
        self.Re      = float(Re)
        self.q_flux  = float(q_flux)
        self.Dh      = float(Dh)
        self.u_inlet = fluid.u_inlet_from_Re(Re, Dh)

    # ------------------------------------------------------------------
    # Inlet  (left face, i = 0)
    # ------------------------------------------------------------------

    def apply_inlet(self, u: np.ndarray, v: np.ndarray,
                    T: np.ndarray, mesh=None) -> None:
        """Apply uniform plug-flow inlet on the left face (all j, i = 0).

            u[:, 0] = U_inlet
            v[:, 0] = 0
            T[:, 0] = T_INLET = 300 K

        The ``mesh`` argument is accepted for API uniformity but is not used
        here (the inlet profile is uniform and needs no geometric information).
        """
        u[:, 0] = self.u_inlet
        v[:, 0] = 0.0
        T[:, 0] = self.T_INLET

    # ------------------------------------------------------------------
    # Outlet  (right face, i = nx-1)
    # ------------------------------------------------------------------

    def apply_outlet(self, u: np.ndarray, v: np.ndarray,
                     p: np.ndarray, T: np.ndarray, mesh=None) -> None:
        """Apply outlet BCs on the right face (all j, i = nx-1).

        Zero-gradient (Neumann) extrapolation for u, v, T:
            u[:, -1] = u[:, -2]
            v[:, -1] = v[:, -2]
            T[:, -1] = T[:, -2]

        Reference pressure (Dirichlet) to anchor the pressure level:
            p[:, -1] = 0.0
        """
        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]
        T[:, -1] = T[:, -2]
        p[:, -1] = 0.0

    # ------------------------------------------------------------------
    # Bottom wall  (j = 0)
    # ------------------------------------------------------------------

    def apply_bottom_wall(self, u: np.ndarray, v: np.ndarray,
                          T: np.ndarray, mesh,
                          q_flux: float | None = None) -> None:
        """Apply bottom-wall BCs (all i, j = 0).

        No-slip (Dirichlet):
            u[0, :] = 0
            v[0, :] = 0

        Constant heat flux — Neumann condition discretised as a
        one-sided first-order difference across the wall-adjacent cell:
            dT/dn = −q / k_f
            →  T[0, :] = T[1, :] + q · Δy_wall / k_f

        where Δy_wall = mesh.dy_wall_bottom (shape (nx,)) is the height of
        the first inflation cell, and n points into the wall (downward).

        Parameters
        ----------
        q_flux : heat flux override [W/m²].  If None, uses ``self.q_flux``.
        """
        u[0, :] = 0.0
        v[0, :] = 0.0

        _q  = self.q_flux if q_flux is None else float(q_flux)
        dy  = mesh.dy_wall_bottom          # (nx,)  — streamwise-indexed
        T[0, :] = T[1, :] + _q * dy / self.fluid.k_f

    # ------------------------------------------------------------------
    # Top wall  (j = ny-1)
    # ------------------------------------------------------------------

    def apply_top_wall(self, u: np.ndarray, v: np.ndarray,
                       T: np.ndarray, mesh=None) -> None:
        """Apply top-wall BCs (all i, j = ny-1).

        No-slip (Dirichlet):
            u[-1, :] = 0
            v[-1, :] = 0

        Adiabatic — zero heat flux (Neumann):
            dT/dn = 0  →  T[-1, :] = T[-2, :]
        """
        u[-1, :] = 0.0
        v[-1, :] = 0.0
        T[-1, :] = T[-2, :]

    # ------------------------------------------------------------------
    # Convenience: apply all four faces at once
    # ------------------------------------------------------------------

    def apply_all(self, u: np.ndarray, v: np.ndarray,
                  p: np.ndarray, T: np.ndarray, mesh) -> None:
        """Apply all four boundary conditions in the canonical order:
        inlet → outlet → bottom wall → top wall.
        """
        self.apply_inlet(u, v, T, mesh)
        self.apply_outlet(u, v, p, T, mesh)
        self.apply_bottom_wall(u, v, T, mesh)
        self.apply_top_wall(u, v, T, mesh)

    # ------------------------------------------------------------------
    # y+ diagnostic
    # ------------------------------------------------------------------

    def compute_yplus(self, u_field: np.ndarray, mesh,
                      fluid: FluidProperties | None = None) -> np.ndarray:
        """Return the y⁺ distribution along the bottom wall.

        Because ``apply_bottom_wall`` forces ``u[0, :] = 0`` (the wall-adjacent
        ghost row), the wall shear stress is estimated from the first interior
        cell row (j = 1):

            τ_w  = μ · |u[1, :]| / Δy_wall          [Pa]
            u_τ  = √(τ_w / ρ)                         [m/s]
            y⁺   = Δy_wall · u_τ / ν                  [−]

        The reference distance ``Δy_wall = mesh.dy_wall_bottom`` is the full
        height of the near-wall cell (treating row 0 as a ghost cell between
        the wall face and the first interior centroid).

        Parameters
        ----------
        u_field : (ny, nx) streamwise velocity array [m/s]
        mesh    : StructuredMesh instance supplying ``dy_wall_bottom``
        fluid   : optional FluidProperties override (defaults to ``self.fluid``)

        Returns
        -------
        y_plus : (nx,) array of y⁺ values at each streamwise position
        """
        _f    = self.fluid if fluid is None else fluid
        u_p   = np.abs(u_field[1, :])           # (nx,) first interior row
        dy    = mesh.dy_wall_bottom              # (nx,)
        tau_w = _f.mu  * u_p / (dy + 1e-300)   # wall shear stress [Pa]
        u_tau = np.sqrt(tau_w / _f.rho)         # friction velocity [m/s]
        return dy * u_tau / (_f.nu + 1e-300)    # y+ [−]

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a formatted summary of the applied boundary conditions."""
        print("BoundaryConditions:")
        print(f"  Re        = {self.Re:.1f}")
        print(f"  Dh        = {self.Dh * 1e3:.3f} mm")
        print(f"  u_inlet   = {self.u_inlet:.5f} m/s")
        print(f"  T_inlet   = {self.T_INLET:.1f} K  (fixed)")
        print(f"  q_flux    = {self.q_flux:.1f} W/m²")
        print(f"  k_fluid   = {self.fluid.k_f:.4f} W/(m·K)")
        print(f"  (top wall: adiabatic, outlet: zero-gradient + p_ref = 0)")

    def __repr__(self) -> str:
        return (f"BoundaryConditions("
                f"Re={self.Re}, Dh={self.Dh:.4f} m, "
                f"u_inlet={self.u_inlet:.5f} m/s, "
                f"q_flux={self.q_flux:.1f} W/m²)")


# ---------------------------------------------------------------------------
# Backward-compatible module-level helpers
# ---------------------------------------------------------------------------

def build_bc(Re: float, H: float, q_flux: float,
             fluid: dict, T_in: float = 293.15) -> dict:
    """Legacy wrapper: return a BC dict for old solver.simple code.

    Creates a BoundaryConditions instance internally but returns the same
    plain dict structure expected by the original SIMPLE solver.
    """
    from wavy_channel_cfd.solver.fluid_properties import FluidProperties
    fp  = FluidProperties(rho=fluid["rho"], mu=fluid["mu"],
                          k_f=fluid["k_f"], Cp=fluid["Cp"])
    Dh  = 2.0 * H
    bc  = BoundaryConditions(fp, Re=Re, q_flux=q_flux, Dh=Dh)
    return {
        "u_in":   bc.u_inlet,
        "T_in":   T_in,
        "q_flux": q_flux,
        "k_f":    fluid["k_f"],
        "Re":     Re,
    }


def summarise_bc(bc: dict) -> None:
    """Print a BC dict produced by the legacy ``build_bc`` function."""
    print("Boundary Conditions (legacy dict):")
    print(f"  u_in   = {bc['u_in']:.5f} m/s  (Re = {bc['Re']})")
    print(f"  T_in   = {bc['T_in']:.2f} K")
    print(f"  q_flux = {bc['q_flux']:.1f} W/m²")
    print(f"  k_f    = {bc['k_f']:.4f} W/(m·K)")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import numpy as np

    # ── Fluid and BC setup ────────────────────────────────────────────
    fluid = FluidProperties.water_300K()
    H     = 0.010          # channel height [m]
    Dh    = 2.0 * H
    Re    = 400
    q     = 5000.0         # W/m²

    bc = BoundaryConditions(fluid, Re=Re, q_flux=q, Dh=Dh)
    bc.summary()
    print()
    print(repr(bc))
    print()

    # ── Synthetic field arrays (ny=10, nx=20 toy mesh) ────────────────
    ny, nx = 10, 20
    u = np.ones((ny, nx)) * bc.u_inlet
    v = np.zeros((ny, nx))
    p = np.linspace(10, 0, nx)[None, :] * np.ones((ny, 1))
    T = np.full((ny, nx), bc.T_INLET)

    # Minimal mesh stub with dy_wall_bottom
    class _MeshStub:
        dy_wall_bottom = np.full(nx, 2e-5)  # 20 µm uniform

    mesh = _MeshStub()

    # Apply all BCs and spot-check.
    # Corner cells (e.g. j=0 at i=0) are set by the last-applied BC; we
    # verify only interior spans to avoid corner-cell ambiguity.
    bc.apply_all(u, v, p, T, mesh)

    assert np.allclose(u[1:-1, 0],  bc.u_inlet),        "inlet u (interior) failed"
    assert np.allclose(v[1:-1, 0],  0.0),                "inlet v failed"
    assert np.allclose(T[1:-1, 0],  bc.T_INLET),         "inlet T failed"
    assert np.allclose(u[1:-1, -1], u[1:-1, -2]),        "outlet u zero-grad failed"
    assert np.allclose(p[:, -1],    0.0),                "outlet p ref failed"
    assert np.allclose(u[0,  1:-1], 0.0),                "bottom no-slip u failed"
    assert np.allclose(u[-1, 1:-1], 0.0),                "top no-slip u failed"
    assert np.allclose(T[-1, 1:-1], T[-2, 1:-1]),        "top adiabatic T failed"

    # Check bottom wall heat-flux BC (T[0,:] set after walls are applied)
    dy  = mesh.dy_wall_bottom
    T_expected_wall = T[1, :] + q * dy / fluid.k_f
    assert np.allclose(T[0, :], T_expected_wall), "bottom heat-flux T failed"

    print("All BC assertions passed.")
    print(f"  Bottom wall temperature range: "
          f"{T[0,:].min():.4f} – {T[0,:].max():.4f} K")
    print()

    # ── y⁺ check (non-zero u[1,:] needed) ────────────────────────────
    u[1, :] = bc.u_inlet * 0.3   # synthetic near-wall velocity
    y_plus  = bc.compute_yplus(u, mesh)
    print(f"y⁺ (synthetic u₁ = {bc.u_inlet*0.3:.4f} m/s):  "
          f"min={y_plus.min():.4f}  mean={y_plus.mean():.4f}  "
          f"max={y_plus.max():.4f}")

    # Target y⁺ < 1 for medium mesh with h1 = 20 µm at Re=400
    target_utau = np.sqrt(fluid.mu * bc.u_inlet / (H * fluid.rho))
    y_plus_est  = 2e-5 * target_utau / fluid.nu
    print(f"Estimated y⁺ for Re={Re}, h₁=20 µm: {y_plus_est:.4f}")

    sys.exit(0)
