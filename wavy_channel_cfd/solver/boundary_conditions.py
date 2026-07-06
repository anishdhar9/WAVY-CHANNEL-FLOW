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

from wavy_channel_cfd.mesh.boundary import BCType, BoundaryPatch
from wavy_channel_cfd.solver.fluid_properties import FluidProperties


# Field-array slices for each index-space side, in the (ny, nx) layout:
# (boundary row/column, adjacent interior row/column)
_SIDE_SLICES = {
    "west":  (np.s_[:, 0],  np.s_[:, 1]),
    "east":  (np.s_[:, -1], np.s_[:, -2]),
    "south": (np.s_[0, :],  np.s_[1, :]),
    "north": (np.s_[-1, :], np.s_[-2, :]),
}


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
    # Patch-based application (mesh-driven)
    # ------------------------------------------------------------------

    def apply_patch(self, patch: BoundaryPatch,
                    u: np.ndarray, v: np.ndarray,
                    p: np.ndarray, T: np.ndarray, mesh) -> None:
        """Apply one boundary patch, dispatching on its BCType.

        The patch's index-space ``side`` selects the field-array slice;
        the ``bc_type`` selects the stencil. Patch ``params`` override
        solver-level defaults (e.g. a per-patch ``q_flux`` or ``T_wall``).
        """
        b, n = _SIDE_SLICES[patch.side]     # boundary slice, neighbour slice
        bt   = patch.bc_type

        if bt is BCType.VELOCITY_INLET:
            # Plug flow entering normal to the patch, into the domain.
            u_in = float(patch.params.get("u_inlet", self.u_inlet))
            T_in = float(patch.params.get("T_inlet", self.T_INLET))
            sign = {"west": +1.0, "east": -1.0,
                    "south": +1.0, "north": -1.0}[patch.side]
            if patch.side in ("west", "east"):
                u[b] = sign * u_in
                v[b] = 0.0
            else:
                u[b] = 0.0
                v[b] = sign * u_in
            T[b] = T_in

        elif bt is BCType.PRESSURE_OUTLET:
            u[b] = u[n]
            v[b] = v[n]
            T[b] = T[n]
            p[b] = float(patch.params.get("p_ref", 0.0))

        elif bt.is_wall():
            u[b] = 0.0
            v[b] = 0.0
            if bt is BCType.WALL_ADIABATIC:
                T[b] = T[n]
            elif bt is BCType.WALL_ISOTHERMAL:
                if "T_wall" not in patch.params:
                    raise ValueError(
                        f"Patch {patch.name!r} is wall-isothermal but has "
                        f"no 'T_wall' parameter")
                T[b] = float(patch.params["T_wall"])
            else:  # WALL_HEATFLUX
                q = float(patch.params.get("q_flux", self.q_flux))
                if patch.side == "south":
                    dy = mesh.dy_wall_bottom
                elif patch.side == "north":
                    dy = mesh.dy_wall_top
                else:
                    raise NotImplementedError(
                        f"Heat-flux wall on side {patch.side!r} is not "
                        f"supported (no wall-normal spacing array)")
                T[b] = T[n] + q * dy / self.fluid.k_f

        elif bt is BCType.SYMMETRY:
            # Zero normal velocity; zero-gradient tangential velocity and T.
            if patch.side in ("west", "east"):
                u[b] = 0.0
                v[b] = v[n]
            else:
                v[b] = 0.0
                u[b] = u[n]
            T[b] = T[n]

        else:  # pragma: no cover — new BCType without a stencil
            raise NotImplementedError(f"No stencil for BC type {bt!r}")

    # ------------------------------------------------------------------
    # Convenience: apply all boundaries at once
    # ------------------------------------------------------------------

    def apply_all(self, u: np.ndarray, v: np.ndarray,
                  p: np.ndarray, T: np.ndarray, mesh) -> None:
        """Apply all boundary conditions.

        If the mesh carries boundary patches (StructuredMesh), each patch
        is applied according to its BCType — inlets/outlets first, walls
        last, so walls win the corner cells (matching the legacy order
        inlet → outlet → bottom wall → top wall).

        Meshes without patches fall back to the four fixed-face methods.
        """
        patches = getattr(mesh, "patches", None)
        if patches:
            ordered = ([pt for pt in patches.values()
                        if not pt.bc_type.is_wall()]
                       + [pt for pt in patches.values()
                          if pt.bc_type.is_wall()])
            for pt in ordered:
                self.apply_patch(pt, u, v, p, T, mesh)
        else:
            self.apply_inlet(u, v, T, mesh)
            self.apply_outlet(u, v, p, T, mesh)
            self.apply_bottom_wall(u, v, T, mesh)
            self.apply_top_wall(u, v, T, mesh)

    # ------------------------------------------------------------------
    # Matrix-form BC contribution (for the FV assembler in
    # solver/discretization.py::assemble_transport_matrix)
    # ------------------------------------------------------------------

    def matrix_contribution(self, patch: BoundaryPatch, field_name: str) -> dict:
        """Return the assembler-facing BC spec for one patch and field.

        The *same* physical patch can be a different kind of condition
        per transported field — e.g. ``wall_bottom`` is Dirichlet
        (no-slip, value 0) for ``u``/``v`` but Neumann (prescribed heat
        flux) for ``T``. This method is the single place that maps
        ``BCType`` + field name to the ``{"kind", "value"/"flux"}`` spec
        ``assemble_transport_matrix`` consumes.

        Parameters
        ----------
        patch      : a BoundaryPatch from mesh.patches
        field_name : one of "u", "v", "T"

        Returns
        -------
        dict, one of:
          {"kind": "dirichlet", "value": <scalar or (n_faces,) array>}
          {"kind": "neumann",   "flux":  <scalar or (n_faces,) array>}
          {"kind": "outflow"}
        """
        bt = patch.bc_type

        if field_name in ("u", "v"):
            if bt is BCType.VELOCITY_INLET:
                u_in = float(patch.params.get("u_inlet", self.u_inlet))
                value = u_in if field_name == "u" else 0.0
                return {"kind": "dirichlet", "value": value}
            if bt is BCType.PRESSURE_OUTLET:
                return {"kind": "outflow"}
            if bt.is_wall():
                return {"kind": "dirichlet", "value": 0.0}
            if bt is BCType.SYMMETRY:
                return {"kind": "neumann", "flux": 0.0}
            raise NotImplementedError(
                f"No {field_name} BC mapping for {bt!r} on patch {patch.name!r}")

        if field_name == "T":
            if bt is BCType.VELOCITY_INLET:
                T_in = float(patch.params.get("T_inlet", self.T_INLET))
                return {"kind": "dirichlet", "value": T_in}
            if bt is BCType.PRESSURE_OUTLET:
                return {"kind": "outflow"}
            if bt is BCType.WALL_HEATFLUX:
                q = float(patch.params.get("q_flux", self.q_flux))
                return {"kind": "neumann", "flux": q}
            if bt is BCType.WALL_ADIABATIC:
                return {"kind": "neumann", "flux": 0.0}
            if bt is BCType.WALL_ISOTHERMAL:
                if "T_wall" not in patch.params:
                    raise ValueError(
                        f"Patch {patch.name!r} is wall-isothermal but has "
                        f"no 'T_wall' parameter")
                return {"kind": "dirichlet", "value": float(patch.params["T_wall"])}
            if bt is BCType.SYMMETRY:
                return {"kind": "neumann", "flux": 0.0}
            raise NotImplementedError(
                f"No T BC mapping for {bt!r} on patch {patch.name!r}")

        raise ValueError(f"Unknown field_name {field_name!r}; expected u, v, or T")

    def build_bc_spec(self, mesh, field_name: str) -> dict:
        """Build the full per-patch BC spec dict for one transported field,
        ready to pass as ``bc=`` into ``assemble_transport_matrix``.
        """
        return {name: self.matrix_contribution(patch, field_name)
                for name, patch in mesh.patches.items()}

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
    print()

    # ── Patch-driven BC application on a real StructuredMesh ─────────
    from wavy_channel_cfd.geometry.channel_gen import ChannelGeometry
    from wavy_channel_cfd.mesh.mesh_gen import StructuredMesh

    geom  = ChannelGeometry(L=0.060, H=H, lam=0.010, AR=0.10, phi_deg=90)
    smesh = StructuredMesh.coarse(geom)
    ny2, nx2 = smesh.ny, smesh.nx

    u2 = np.random.default_rng(0).uniform(0.0, 0.1, (ny2, nx2))
    v2 = np.zeros((ny2, nx2))
    p2 = np.ones((ny2, nx2))
    T2 = np.full((ny2, nx2), bc.T_INLET)

    bc.apply_all(u2, v2, p2, T2, smesh)   # dispatches through smesh.patches

    assert np.allclose(u2[1:-1, 0],  bc.u_inlet),  "patch inlet u failed"
    assert np.allclose(p2[:, -1],    0.0),          "patch outlet p failed"
    assert np.allclose(u2[0,  :],    0.0),          "patch bottom no-slip failed"
    assert np.allclose(u2[-1, :],    0.0),          "patch top no-slip failed"
    assert np.allclose(T2[-1, 1:-1], T2[-2, 1:-1]), "patch adiabatic failed"
    T_exp = T2[1, :] + bc.q_flux * smesh.dy_wall_bottom / fluid.k_f
    assert np.allclose(T2[0, :], T_exp),            "patch heat-flux failed"

    # Isothermal override on the top wall
    from wavy_channel_cfd.mesh.boundary import BCType
    smesh.set_patch_bc("wall_top", BCType.WALL_ISOTHERMAL, T_wall=350.0)
    bc.apply_all(u2, v2, p2, T2, smesh)
    assert np.allclose(T2[-1, :], 350.0),           "patch isothermal failed"

    print("All patch-driven BC assertions passed.")
    for patch in smesh.patches.values():
        print(f"  {patch!r}")
    print()

    # ── matrix_contribution() / build_bc_spec() ──────────────────────
    smesh.set_patch_bc("wall_top", BCType.WALL_ADIABATIC)  # reset from isothermal
    spec_u = bc.build_bc_spec(smesh, "u")
    spec_T = bc.build_bc_spec(smesh, "T")
    assert spec_u["inlet"] == {"kind": "dirichlet", "value": bc.u_inlet}
    assert spec_u["outlet"] == {"kind": "outflow"}
    assert spec_u["wall_bottom"] == {"kind": "dirichlet", "value": 0.0}
    assert spec_T["wall_bottom"] == {"kind": "neumann", "flux": bc.q_flux}
    assert spec_T["wall_top"] == {"kind": "neumann", "flux": 0.0}
    print("matrix_contribution()/build_bc_spec() dispatch correctly:")
    print("  u-spec:", spec_u)
    print("  T-spec:", spec_T)

    sys.exit(0)
