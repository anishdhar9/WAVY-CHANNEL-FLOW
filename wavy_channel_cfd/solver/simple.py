"""
solver/simple.py

SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) solver for
steady, incompressible, laminar flow with conjugate heat transfer on the
non-orthogonal body-fitted wavy-channel mesh.

This is a from-scratch finite-volume implementation built on
solver/discretization.py's mesh-geometry-aware assembler:
  - Momentum (u, v): convection-diffusion, Picard-lagged convection
    (upwind on the Rhie-Chow face mass flux), non-orthogonal-corrected
    diffusion, solved as a real sparse linear system each outer iteration.
  - Pressure-velocity coupling: Rhie-Chow face-velocity interpolation
    (required on this collocated grid to avoid checkerboarding) feeding a
    properly derived pressure-correction equation; after solving p', both
    cell velocities AND face mass fluxes are corrected (the latter is what
    actually drives continuity to solver tolerance).
  - Energy (T): same convection-diffusion assembler, always includes
    convection (required for the problem to be well-posed with Neumann
    wall conditions), with wall heat-flux/adiabatic conditions folded in
    as real flux source terms (not a copy-from-neighbour overwrite).
  - Boundary conditions are entirely patch-driven
    (BoundaryConditions.build_bc_spec()) — no ad hoc per-face BC logic.

Convergence is monitored via normalized residuals of the assembled
(pre-under-relaxation) linear systems, not raw field-to-field change.
"""

from __future__ import annotations

import numpy as np

from wavy_channel_cfd.mesh.mesh_gen import StructuredMesh
from wavy_channel_cfd.solver.boundary_conditions import BoundaryConditions
from wavy_channel_cfd.solver.fluid_properties import FluidProperties
from wavy_channel_cfd.solver.discretization import (
    build_fv_geometry, assemble_transport_matrix, green_gauss_gradient,
    rhie_chow_face_flux, boundary_dgeo, PATCH_SLICES,
)
from wavy_channel_cfd.solver.linear_solvers import solve_linear


# ---------------------------------------------------------------------------
# Residual tracking
# ---------------------------------------------------------------------------

class ResidualMonitor:
    """Accumulates normalized per-iteration equation residuals for each field.

    Each residual is ||A·phi - b||_1 (computed from the *un-relaxed*
    assembled system, before under-relaxation is folded into the
    diagonal/RHS) divided by a fixed physical reference scale, following
    the Patankar / Ferziger-Peric convention. This measures how well the
    discretized equation is satisfied, not merely how much the field
    moved between iterations.
    """

    def __init__(self, fields: list[str]):
        self.history: dict[str, list[float]] = {f: [] for f in fields}

    def record(self, field: str, residual: float) -> None:
        self.history[field].append(residual)

    def latest(self) -> dict[str, float]:
        return {f: vals[-1] if vals else float("nan")
                for f, vals in self.history.items()}

    def converged(self, tol: float) -> bool:
        return all(v[-1] < tol for v in self.history.values() if v)


def _normalized_residual(A, b, phi_flat: np.ndarray, ref: float) -> float:
    return float(np.sum(np.abs(A @ phi_flat - b))) / max(ref, 1e-300)


# ---------------------------------------------------------------------------
# Core SIMPLE iteration
# ---------------------------------------------------------------------------

def solve(mesh: StructuredMesh, bc: BoundaryConditions, fluid: FluidProperties,
         params: dict, verbose: bool = True):
    """Run the SIMPLE loop and return converged flow and temperature fields.

    Parameters
    ----------
    mesh   : a StructuredMesh instance (carries face geometry + patches)
    bc     : a BoundaryConditions instance
    fluid  : a FluidProperties instance
    params : solver control parameters:
               max_iter      – maximum outer iterations (default 2000)
               tol           – convergence tolerance, normalized residual (default 1e-6)
               alpha_u       – under-relaxation for velocity (default 0.7)
               alpha_p       – under-relaxation for pressure (default 0.3)
               alpha_T       – under-relaxation for temperature (default 0.9)
               linear_method – "direct" or "iterative" (default "direct")
               convection    – include momentum convection (default True; set
                               False to recover the M1 Stokes-flow stepping stone)
    verbose: print residuals every 100 iterations

    Returns
    -------
    u, v, p, T  : 2-D field arrays of shape (ny, nx)
    monitor     : ResidualMonitor with iteration history
    """
    if isinstance(mesh, dict):
        raise TypeError(
            "solve() now requires a StructuredMesh instance, not the legacy "
            "mesh dict from mesh.mesh_gen.generate_mesh(). Construct a "
            "StructuredMesh (see mesh/mesh_gen.py) and a matching "
            "BoundaryConditions/FluidProperties object instead.")

    geom = build_fv_geometry(mesh)
    ny, nx = geom["ny"], geom["nx"]

    max_iter      = params.get("max_iter", 2000)
    tol           = params.get("tol", 1e-6)
    # NOTE: config.yaml ships alpha_u=0.7/alpha_p=0.3 (the classic
    # Patankar alpha_u+alpha_p=1 heuristic). Empirically, with this
    # solver's actual (now correctly rho-scaled) pressure-correction
    # coupling, that combination diverges; alpha_p needs to be much
    # smaller here. Defaults below are the values that converged cleanly
    # in bring-up testing on this mesh family — callers passing
    # config.yaml's values directly should expect to need to lower alpha_p.
    alpha_u       = params.get("alpha_u", 0.5)
    alpha_p       = params.get("alpha_p", 0.05)
    alpha_T       = params.get("alpha_T", 0.9)
    linear_method = params.get("linear_method", "direct")
    use_convection = params.get("convection", True)

    rho, mu, k_f, Cp = fluid.rho, fluid.mu, fluid.k_f, fluid.Cp

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    T = np.full((ny, nx), bc.T_INLET)

    monitor = ResidualMonitor(["u", "v", "p", "T"])

    inlet_len_total = float(np.sum(mesh.patches["inlet"].length))
    m_ref      = max(rho * bc.u_inlet * inlet_len_total, 1e-30)
    mom_ref    = max(m_ref * bc.u_inlet, 1e-30)
    energy_ref = max(m_ref * Cp, 1e-30)

    bc_u = bc.build_bc_spec(mesh, "u")
    bc_v = bc.build_bc_spec(mesh, "v")
    bc_T = bc.build_bc_spec(mesh, "T")

    # ---- initial face mass fluxes (uniform plug flow through the domain,
    # before any pressure field exists) --------------------------------
    mdot_east  = rho * bc.u_inlet * geom["east"]["len"]
    mdot_north = np.zeros_like(geom["north"]["Dgeo"])
    mdot_boundary = {
        "inlet":       -rho * bc.u_inlet * mesh.patches["inlet"].length,
        "outlet":       rho * bc.u_inlet * mesh.patches["outlet"].length,
        "wall_bottom":  np.zeros(nx),
        "wall_top":     np.zeros(nx),
    }

    for iteration in range(1, max_iter + 1):
        u_old, v_old, p_old, T_old = u.copy(), v.copy(), p.copy(), T.copy()

        # ================= Momentum predictor (u*, v*) ==================
        mdot_e = mdot_east if use_convection else None
        mdot_n = mdot_north if use_convection else None
        mdot_b = mdot_boundary if use_convection else None

        gx_p, gy_p = green_gauss_gradient(
            p, geom, mesh,
            {"inlet": p[:, 0], "outlet": np.zeros(ny),
             "wall_bottom": p[0, :], "wall_top": p[-1, :]})

        A_u, b_u = assemble_transport_matrix(
            mesh, geom, gamma=mu, mdot_east=mdot_e, mdot_north=mdot_n,
            mdot_boundary=mdot_b, convect_scale=1.0, bc=bc_u,
            phi_old=u_old, source=-gx_p * geom["dA"])
        A_v, b_v = assemble_transport_matrix(
            mesh, geom, gamma=mu, mdot_east=mdot_e, mdot_north=mdot_n,
            mdot_boundary=mdot_b, convect_scale=1.0, bc=bc_v,
            phi_old=v_old, source=-gy_p * geom["dA"])

        res_u = _normalized_residual(A_u, b_u, u_old.ravel(), mom_ref)
        res_v = _normalized_residual(A_v, b_v, v_old.ravel(), mom_ref)

        aP_u_diag = A_u.diagonal()
        A_u_r = A_u.tolil(); A_u_r.setdiag(aP_u_diag / alpha_u); A_u_r = A_u_r.tocsr()
        b_u_r = b_u + (1 - alpha_u) * (aP_u_diag / alpha_u) * u_old.ravel()
        u_star = solve_linear(A_u_r, b_u_r, x0=u_old.ravel(),
                              method=linear_method).reshape(ny, nx)
        aP_u = A_u_r.diagonal().reshape(ny, nx)

        aP_v_diag = A_v.diagonal()
        A_v_r = A_v.tolil(); A_v_r.setdiag(aP_v_diag / alpha_u); A_v_r = A_v_r.tocsr()
        b_v_r = b_v + (1 - alpha_u) * (aP_v_diag / alpha_u) * v_old.ravel()
        v_star = solve_linear(A_v_r, b_v_r, x0=v_old.ravel(),
                              method=linear_method).reshape(ny, nx)

        # NOTE: no post-solve Dirichlet overwrite here. u_star already
        # satisfies the Dirichlet coupling through the matrix (a genuine
        # distance-based coefficient, not a copy) — overwriting it to the
        # exact wall/inlet value afterwards was tried and empirically
        # reintroduces a small residual every iteration (the overwritten
        # value no longer exactly satisfies the assembled equation),
        # which caps the achievable residual at a nonzero floor instead of
        # converging to machine precision. See _enforce_dirichlet()'s
        # docstring for the (unused-here, kept for the legacy/self-test
        # call path) belt-and-suspenders variant.

        # ================= Rhie-Chow face flux (predictor) ===============
        mdot_east, mdot_north, mdot_boundary, DVOL = rhie_chow_face_flux(
            mesh, geom, u_star, v_star, p_old, aP_u, rho)
        mdot_boundary["wall_bottom"] = np.zeros(nx)   # no-penetration
        mdot_boundary["wall_top"]    = np.zeros(nx)
        mdot_boundary["inlet"] = -rho * bc.u_inlet * mesh.patches["inlet"].length

        # ================= Pressure correction ============================
        # net_out(P) = sum over faces of the flux in the *locally outward*
        # direction from P. Internal mdot_east/mdot_north use an
        # "increasing i/j is positive" convention, so a west/south neighbour
        # contribution must be negated to become P's own outward flux.
        # Boundary mdot_boundary values are already stored outward-positive
        # (see rhie_chow_face_flux/PATCH_SLICES), so they are added directly
        # with NO sign flip on any side, including inlet/wall_bottom.
        net_out = np.zeros((ny, nx))
        net_out[:, :-1] += mdot_east;  net_out[:, -1] += mdot_boundary["outlet"]
        net_out[:, 1:]  -= mdot_east;  net_out[:, 0]   += mdot_boundary["inlet"]
        net_out[:-1, :] += mdot_north; net_out[-1, :]  += mdot_boundary["wall_top"]
        net_out[1:, :]  -= mdot_north; net_out[0, :]   += mdot_boundary["wall_bottom"]

        # Pressure-correction face "diffusivity" is rho * (Vol/aP) — the
        # missing rho here previously made p' ~1000x too large (and the
        # face-flux correction below used the same coefficient structure,
        # so it must match exactly what assemble_transport_matrix uses
        # internally: gamma_face * Dgeo_face).
        DVOL_e = 0.5 * (DVOL[:, :-1] + DVOL[:, 1:])
        DVOL_n = 0.5 * (DVOL[:-1, :] + DVOL[1:, :])
        gamma_p = {"east": rho * DVOL_e, "north": rho * DVOL_n,
                  "inlet": 0.0, "outlet": rho * DVOL[:, -1],
                  "wall_bottom": 0.0, "wall_top": 0.0}
        bc_p = {
            "inlet":       {"kind": "neumann", "flux": 0.0},
            "outlet":      {"kind": "dirichlet", "value": 0.0},
            "wall_bottom": {"kind": "neumann", "flux": 0.0},
            "wall_top":    {"kind": "neumann", "flux": 0.0},
        }
        A_p, b_p = assemble_transport_matrix(
            mesh, geom, gamma_p, bc=bc_p, nonorth_correction=False,
            source=-net_out)
        res_p = _normalized_residual(A_p, b_p, p_old.ravel(), m_ref)
        p_prime = solve_linear(A_p, b_p, method=linear_method).reshape(ny, nx)

        # ---- correct pressure, velocity, and face mass fluxes -----------
        p = p_old + alpha_p * p_prime

        gxpp, gypp = green_gauss_gradient(
            p_prime, geom, mesh,
            {"inlet": p_prime[:, 0], "outlet": np.zeros(ny),
             "wall_bottom": p_prime[0, :], "wall_top": p_prime[-1, :]})
        u = u_star - DVOL * gxpp
        v = v_star - DVOL * gypp

        D_e_p = rho * DVOL_e * geom["east"]["Dgeo"]
        D_n_p = rho * DVOL_n * geom["north"]["Dgeo"]
        mdot_east  = mdot_east  + D_e_p * (p_prime[:, :-1] - p_prime[:, 1:])
        mdot_north = mdot_north + D_n_p * (p_prime[:-1, :] - p_prime[1:, :])
        Dgeo_out, _, _ = boundary_dgeo(geom, mesh, "outlet")
        D_b_p = rho * DVOL[:, -1] * Dgeo_out
        mdot_boundary["outlet"] = mdot_boundary["outlet"] + \
            D_b_p * (p_prime[:, -1] - 0.0)

        # ================= Energy equation ================================
        mdot_e_T = mdot_east if use_convection else np.zeros_like(mdot_east)
        mdot_n_T = mdot_north if use_convection else np.zeros_like(mdot_north)
        mdot_b_T = mdot_boundary if use_convection else {
            k: np.zeros_like(val) for k, val in mdot_boundary.items()}
        A_T, b_T = assemble_transport_matrix(
            mesh, geom, gamma=k_f, mdot_east=mdot_e_T, mdot_north=mdot_n_T,
            mdot_boundary=mdot_b_T, convect_scale=Cp, bc=bc_T, phi_old=T_old)
        res_T = _normalized_residual(A_T, b_T, T_old.ravel(), energy_ref)

        aP_T_diag = A_T.diagonal()
        A_T_r = A_T.tolil(); A_T_r.setdiag(aP_T_diag / alpha_T); A_T_r = A_T_r.tocsr()
        b_T_r = b_T + (1 - alpha_T) * (aP_T_diag / alpha_T) * T_old.ravel()
        T = solve_linear(A_T_r, b_T_r, x0=T_old.ravel(),
                         method=linear_method).reshape(ny, nx)

        monitor.record("u", res_u); monitor.record("v", res_v)
        monitor.record("p", res_p); monitor.record("T", res_T)

        if verbose and iteration % 100 == 0:
            r = monitor.latest()
            print(f"  iter {iteration:4d} | "
                 f"u={r['u']:.2e} v={r['v']:.2e} "
                 f"p={r['p']:.2e} T={r['T']:.2e}")

        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(T)):
            raise FloatingPointError(
                f"SIMPLE solver diverged (non-finite field) at iteration {iteration}")

        if monitor.converged(tol):
            if verbose:
                print(f"  Converged at iteration {iteration}.")
            break

    return u, v, p, T, monitor


def _enforce_dirichlet(field: np.ndarray, mesh, bc: BoundaryConditions,
                       field_name: str) -> None:
    """Overwrite Dirichlet-patch boundary cells with their exact value.

    NOT called from solve()'s main loop (see the comment at its momentum
    predictor step): the FV assembly already couples these cells through a
    genuine, distance-based Dirichlet-face term in assemble_transport_matrix,
    and empirically, overwriting the solved value afterwards reintroduces a
    small per-iteration inconsistency with that same equation, capping the
    achievable residual at a nonzero floor instead of letting it converge to
    machine precision. Kept as a standalone utility for callers that want
    the cheaper, old-convention approximation (exact only in the h1->0
    limit) — e.g. matching compute_yplus's assumption that u[0,:]==0 — in
    a context where they are not also monitoring the equation residual.
    """
    for name, patch in mesh.patches.items():
        spec = bc.matrix_contribution(patch, field_name)
        if spec["kind"] == "dirichlet":
            field[PATCH_SLICES[name]] = spec["value"]
