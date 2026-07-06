"""
solver/discretization.py

Finite-volume assembly primitives for the 2-D non-orthogonal body-fitted
mesh produced by mesh.mesh_gen.StructuredMesh.

Shape convention (landmine, documented once here):
  StructuredMesh geometry arrays (Xc, Yc, dA, face_len_*, face_nx_*,
  face_ny_*) are (nx, ny) — streamwise, wall-normal.
  Solver field arrays (u, v, p, T) are (ny, nx) — wall-normal, streamwise
  (see solver/boundary_conditions.py). build_fv_geometry() transposes
  every 2-D geometry array exactly once and caches the result; 1-D patch
  arrays (mesh.patches[name].xf/yf/nx_f/ny_f/length) are already in the
  correct per-boundary orientation and must NOT be transposed.

Non-orthogonal diffusion uses the over-relaxed decomposition (Jasak
1996): each internal face's area vector S_f is split into an orthogonal
component E_f (collinear with the cell-centre connection d) and a
cross-diffusion component T_f. E_f gives an implicit, unconditionally
bounded coefficient; T_f is applied via deferred correction using
face-interpolated Green-Gauss gradients from the previous outer
iteration — no extra inner loop needed, it converges alongside the
outer SIMPLE loop. On an axis-aligned (straight-channel) mesh T_f == 0
everywhere and this reduces to a plain 5-point Laplacian.

Convection uses first-order upwind on the face mass flux (unconditionally
stable at any local Peclet number). The generic assembler below is
reused, unchanged, for the u, v, and T transport equations, and for the
pressure-correction equation (as a pure-diffusion special case with a
face-varying "diffusivity" derived from the momentum equation's 1/a_P).

All boundary geometry is taken directly from mesh.patches (already
outward-normal, already correctly oriented — no transpose needed).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

PATCH_SLICES = {
    "inlet":       (slice(None), 0),
    "outlet":      (slice(None), -1),
    "wall_bottom": (0, slice(None)),
    "wall_top":    (-1, slice(None)),
}


def build_fv_geometry(mesh) -> dict:
    """Precompute transposed, (ny,nx)-oriented internal-face geometry.

    Returns a dict with keys:
      nx, ny         : cell counts
      Xc, Yc, dA     : (ny,nx) cell centres [m] and cell area [m^2]
      east, north    : dict of internal-face geometry (see _internal_face)
    """
    nx, ny = mesh.nx, mesh.ny
    Xc = mesh.Xc.T
    Yc = mesh.Yc.T
    dA = mesh.dA.T
    Xn = mesh.X.T
    Yn = mesh.Y.T

    fx_e = 0.5 * (Xn[:-1, 1:-1] + Xn[1:, 1:-1])
    fy_e = 0.5 * (Yn[:-1, 1:-1] + Yn[1:, 1:-1])
    fx_n = 0.5 * (Xn[1:-1, :-1] + Xn[1:-1, 1:])
    fy_n = 0.5 * (Yn[1:-1, :-1] + Yn[1:-1, 1:])

    east  = _internal_face(Xc, Yc, mesh.face_len_east,  mesh.face_nx_east,
                           mesh.face_ny_east,  axis=1, fx=fx_e, fy=fy_e)
    north = _internal_face(Xc, Yc, mesh.face_len_north, mesh.face_nx_north,
                           mesh.face_ny_north, axis=0, fx=fx_n, fy=fy_n)

    return {"nx": nx, "ny": ny, "Xc": Xc, "Yc": Yc, "dA": dA,
            "east": east, "north": north}


def _internal_face(Xc, Yc, face_len, face_nx, face_ny, axis: int,
                   fx: np.ndarray, fy: np.ndarray) -> dict:
    """Internal-face geometry + non-orthogonal E_f/T_f split.

    axis=1 -> east faces (between column i and i+1)
    axis=0 -> north faces (between row j and j+1)

    fx, fy are the true geometric face-centre coordinates (from mesh
    nodes) — needed for distance-weighted face interpolation, since the
    inflation-layer mesh is strongly non-uniform in the wall-normal
    direction and a naive 0.5/0.5 average is not the face value even
    for a linear field.
    """
    L, NX, NY = face_len.T, face_nx.T, face_ny.T
    if axis == 1:
        L, NX, NY = L[:, :-1], NX[:, :-1], NY[:, :-1]
        XcP, YcP = Xc[:, :-1], Yc[:, :-1]
        dx = Xc[:, 1:] - Xc[:, :-1]
        dy = Yc[:, 1:] - Yc[:, :-1]
    else:
        L, NX, NY = L[:-1, :], NX[:-1, :], NY[:-1, :]
        XcP, YcP = Xc[:-1, :], Yc[:-1, :]
        dx = Xc[1:, :] - Xc[:-1, :]
        dy = Yc[1:, :] - Yc[:-1, :]

    Sx, Sy = L * NX, L * NY
    dS    = dx * Sx + dy * Sy
    dS    = np.where(np.abs(dS) > 1e-300, dS, 1e-300)
    Smag2 = Sx**2 + Sy**2
    factor = Smag2 / dS
    Dgeo  = np.abs(factor)                 # |E_f| / |d|  (dimensionless)
    Ex, Ey = factor * dx, factor * dy
    Tx, Ty = Sx - Ex, Sy - Ey
    dmag = np.sqrt(dx**2 + dy**2)

    # Distance-weighted face-interpolation factor (fraction toward the
    # "N" neighbour), using the true face-centre location.
    dist_Pf = np.sqrt((fx - XcP)**2 + (fy - YcP)**2)
    wN = np.clip(dist_Pf / np.where(dmag > 1e-300, dmag, 1e-300), 0.0, 1.0)

    return {"len": L, "nx": NX, "ny": NY, "dx": dx, "dy": dy,
            "dmag": dmag, "Sx": Sx, "Sy": Sy,
            "Dgeo": Dgeo, "Tx": Tx, "Ty": Ty, "wN": wN}


def boundary_dgeo(geom: dict, mesh, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Dgeo_b, dist_b, patch) for a boundary patch.

    Dgeo_b = length_b / dist_b, where dist_b is the distance from the
    owner cell centre to the face centre (orthogonal boundary
    approximation — no non-orthogonal correction at boundaries).
    """
    patch = mesh.patches[name]
    sl = PATCH_SLICES[name]
    xo, yo = geom["Xc"][sl], geom["Yc"][sl]
    dist = np.sqrt((xo - patch.xf)**2 + (yo - patch.yf)**2)
    dist = np.where(dist > 1e-300, dist, 1e-300)
    Dgeo = patch.length / dist
    return Dgeo, dist, patch


def green_gauss_gradient(phi: np.ndarray, geom: dict, mesh,
                         boundary_values: dict) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct cell-centred gradients via Green-Gauss over cell faces.

    boundary_values: {"inlet": (ny,), "outlet": (ny,),
                      "wall_bottom": (nx,), "wall_top": (nx,)}
    — the face value to use at each boundary (the Dirichlet value for
    Dirichlet fields, or the owner-cell value itself for zero-gradient/
    Neumann fields; callers build this per field).
    """
    ny, nx = geom["ny"], geom["nx"]
    dA = geom["dA"]
    gx = np.zeros((ny, nx))
    gy = np.zeros((ny, nx))

    e = geom["east"]
    phi_e = (1.0 - e["wN"]) * phi[:, :-1] + e["wN"] * phi[:, 1:]
    gx[:, :-1] += phi_e * e["Sx"];  gx[:, 1:] -= phi_e * e["Sx"]
    gy[:, :-1] += phi_e * e["Sy"];  gy[:, 1:] -= phi_e * e["Sy"]

    n = geom["north"]
    phi_n = (1.0 - n["wN"]) * phi[:-1, :] + n["wN"] * phi[1:, :]
    gx[:-1, :] += phi_n * n["Sx"];  gx[1:, :] -= phi_n * n["Sx"]
    gy[:-1, :] += phi_n * n["Sy"];  gy[1:, :] -= phi_n * n["Sy"]

    for name, sl in PATCH_SLICES.items():
        patch = mesh.patches[name]
        Sx, Sy = patch.length * patch.nx_f, patch.length * patch.ny_f
        val = boundary_values[name]
        gx[sl] += val * Sx
        gy[sl] += val * Sy

    gx /= dA
    gy /= dA
    return gx, gy


def assemble_transport_matrix(mesh, geom: dict, gamma, *,
                              mdot_east=None, mdot_north=None,
                              mdot_boundary: dict | None = None,
                              convect_scale: float = 1.0,
                              bc: dict | None = None,
                              phi_old: np.ndarray | None = None,
                              nonorth_correction: bool = True,
                              source: np.ndarray | None = None):
    """Assemble the sparse transport matrix for one scalar field.

    Discretizes, over every cell, the steady convection-diffusion
    balance  div(rho*u*phi) = div(gamma*grad(phi)) + source, using
    upwind convection and non-orthogonal-corrected diffusion on the
    actual mesh faces.

    Parameters
    ----------
    gamma   : diffusivity. Either a scalar, or a dict with keys
              "east","north" giving per-internal-face arrays (broadcast
              against geom["east"]["Dgeo"].shape / geom["north"]["Dgeo"].shape)
              and keys "inlet","outlet","wall_bottom","wall_top" giving
              per-boundary-face diffusivity (broadcast against the patch
              face-count shape). A bare scalar is broadcast to all of these.
    mdot_east, mdot_north : internal-face mass flow rates [same convention
              as convect_scale], positive = flow toward increasing i / j.
              None => diffusion-only (e.g. pressure correction).
    mdot_boundary : dict of boundary mass flow rates, positive = outward
              from the domain. Required (non-None per key) wherever
              bc[name]["kind"] involves convection ("outflow", or a
              "dirichlet" advecting inflow). Use zeros for walls.
    convect_scale : multiplier converting mdot to the transported flux
              coefficient (1.0 for momentum components, Cp for energy).
    bc      : dict per patch name -> {"kind": "dirichlet"|"neumann"|"outflow",
              "value": array/scalar (dirichlet), "flux": array/scalar
              (neumann, positive = into the domain)}.
    phi_old : previous outer-iteration field, used for the non-orthogonal
              deferred correction (and ignored, safely, if
              nonorth_correction=False or phi_old is None).
    source  : (ny,nx) explicit per-cell source term added directly to the
              RHS (e.g. the mass-imbalance source for the pressure-
              correction equation, or a pressure-gradient momentum source).

    Returns
    -------
    A   : scipy.sparse.csr_matrix, shape (ny*nx, ny*nx)
    b   : (ny*nx,) right-hand side
    """
    ny, nx = geom["ny"], geom["nx"]
    N = ny * nx
    idx = np.arange(N).reshape(ny, nx)
    bc = bc or {}

    if not isinstance(gamma, dict):
        gamma = {k: gamma for k in
                 ("east", "north", "inlet", "outlet", "wall_bottom", "wall_top")}

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    b = np.zeros(N)

    def _add(r, c, d):
        rows.append(r.ravel()); cols.append(c.ravel()); data.append(d.ravel())

    # ---- non-orthogonal deferred-correction gradients (previous iterate) --
    gx = gy = None
    if nonorth_correction and phi_old is not None:
        grad_bvals = {}
        for name, sl in PATCH_SLICES.items():
            kind = bc.get(name, {}).get("kind", "neumann")
            if kind == "dirichlet":
                grad_bvals[name] = np.broadcast_to(
                    np.asarray(bc[name]["value"], dtype=float),
                    phi_old[sl].shape)
            else:
                grad_bvals[name] = phi_old[sl]
        gx, gy = green_gauss_gradient(phi_old, geom, mesh, grad_bvals)

    # ---- internal east faces -------------------------------------------
    e = geom["east"]
    gamma_e = np.broadcast_to(np.asarray(gamma["east"], dtype=float), e["Dgeo"].shape)
    D_e = gamma_e * e["Dgeo"]
    F_e = mdot_east * convect_scale if mdot_east is not None else 0.0

    idxP = idx[:, :-1]; idxE = idx[:, 1:]
    aEP = D_e + np.maximum(F_e, 0.0)   # P's diagonal contribution
    aPE = D_e + np.maximum(-F_e, 0.0)  # E's diagonal contribution
    _add(idxP, idxP, aEP)
    _add(idxE, idxE, aPE)
    _add(idxP, idxE, -aPE)
    _add(idxE, idxP, -aEP)

    if gx is not None:
        gxf = (1.0 - e["wN"]) * gx[:, :-1] + e["wN"] * gx[:, 1:]
        gyf = (1.0 - e["wN"]) * gy[:, :-1] + e["wN"] * gy[:, 1:]
        corr = gamma_e * (e["Tx"] * gxf + e["Ty"] * gyf)
        b[idxP.ravel()] += corr.ravel()
        b[idxE.ravel()] -= corr.ravel()

    # ---- internal north faces ------------------------------------------
    n = geom["north"]
    gamma_n = np.broadcast_to(np.asarray(gamma["north"], dtype=float), n["Dgeo"].shape)
    D_n = gamma_n * n["Dgeo"]
    F_n = mdot_north * convect_scale if mdot_north is not None else 0.0

    idxP = idx[:-1, :]; idxN = idx[1:, :]
    aNP = D_n + np.maximum(F_n, 0.0)
    aPN = D_n + np.maximum(-F_n, 0.0)
    _add(idxP, idxP, aNP)
    _add(idxN, idxN, aPN)
    _add(idxP, idxN, -aPN)
    _add(idxN, idxP, -aNP)

    if gx is not None:
        gxf = (1.0 - n["wN"]) * gx[:-1, :] + n["wN"] * gx[1:, :]
        gyf = (1.0 - n["wN"]) * gy[:-1, :] + n["wN"] * gy[1:, :]
        corr = gamma_n * (n["Tx"] * gxf + n["Ty"] * gyf)
        b[idxP.ravel()] += corr.ravel()
        b[idxN.ravel()] -= corr.ravel()

    # ---- boundary patches ------------------------------------------------
    for name, sl in PATCH_SLICES.items():
        spec = bc.get(name, {"kind": "neumann", "flux": 0.0})
        kind = spec["kind"]
        Dgeo_b, dist_b, patch = boundary_dgeo(geom, mesh, name)
        gamma_b = np.broadcast_to(np.asarray(gamma[name], dtype=float), Dgeo_b.shape)
        D_b = gamma_b * Dgeo_b
        idxB = idx[sl]
        F_b = (mdot_boundary[name] * convect_scale
               if mdot_boundary is not None and mdot_boundary.get(name) is not None
               else 0.0)

        if kind == "dirichlet":
            value = np.broadcast_to(np.asarray(spec["value"], dtype=float), Dgeo_b.shape)
            diag = D_b + np.maximum(F_b, 0.0)
            rhs  = (D_b + np.maximum(-F_b, 0.0)) * value
            _add(idxB, idxB, diag)
            b[idxB.ravel()] += rhs.ravel()

        elif kind == "neumann":
            flux = np.broadcast_to(np.asarray(spec.get("flux", 0.0), dtype=float),
                                   Dgeo_b.shape)
            b[idxB.ravel()] += (flux * patch.length).ravel()

        elif kind == "outflow":
            diag = np.broadcast_to(np.maximum(F_b, 0.0), Dgeo_b.shape)
            _add(idxB, idxB, diag)

        else:
            raise ValueError(f"Unknown boundary kind {kind!r} for patch {name!r}")

    A = sp.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N, N)).tocsr()
    if source is not None:
        b += np.asarray(source, dtype=float).ravel()
    return A, b


def rhie_chow_face_flux(mesh, geom: dict, u, v, p, aP_u, rho: float):
    """Rhie-Chow-interpolated face normal mass flux (prevents collocated-grid
    pressure checkerboarding).

    Uses Volume/a_P (from the u-momentum diagonal) as the interpolation
    weight for both velocity components — a standard simplification when
    a_P,u and a_P,v are of similar magnitude (true here: both momentum
    components share the same diffusion/convection coefficients, only the
    pressure-gradient source differs).

    Returns
    -------
    mdot_east, mdot_north : internal-face mass flow rates
    mdot_boundary          : dict of boundary mass flow rates (outward +)
    DVOL                   : (ny,nx) dA/aP_u, exposed for the pressure
                              correction equation's face "diffusivity"
    """
    dA = geom["dA"]
    DVOL = dA / aP_u

    gx_p, gy_p = green_gauss_gradient(
        p, geom, mesh,
        {"inlet": p[:, 0], "outlet": np.zeros(geom["ny"]),
         "wall_bottom": p[0, :], "wall_top": p[-1, :]})

    def _face(u_, v_, f, axis):
        if axis == 1:
            uP, uN = u_[:, :-1], u_[:, 1:]
            vP, vN = v_[:, :-1], v_[:, 1:]
            gxP, gxN = gx_p[:, :-1], gx_p[:, 1:]
            gyP, gyN = gy_p[:, :-1], gy_p[:, 1:]
            pP, pN = p[:, :-1], p[:, 1:]
            dvP, dvN = DVOL[:, :-1], DVOL[:, 1:]
        else:
            uP, uN = u_[:-1, :], u_[1:, :]
            vP, vN = v_[:-1, :], v_[1:, :]
            gxP, gxN = gx_p[:-1, :], gx_p[1:, :]
            gyP, gyN = gy_p[:-1, :], gy_p[1:, :]
            pP, pN = p[:-1, :], p[1:, :]
            dvP, dvN = DVOL[:-1, :], DVOL[1:, :]

        nx_f, ny_f, dmag, wN = f["nx"], f["ny"], f["dmag"], f["wN"]
        Vn_lin = ((1 - wN) * (uP * nx_f + vP * ny_f)
                 + wN * (uN * nx_f + vN * ny_f))
        d_hat_x = f["dx"] / np.where(dmag > 1e-300, dmag, 1e-300)
        d_hat_y = f["dy"] / np.where(dmag > 1e-300, dmag, 1e-300)
        gradp_avg_dot_d = ((1 - wN) * (gxP * d_hat_x + gyP * d_hat_y)
                          + wN * (gxN * d_hat_x + gyN * d_hat_y))
        d_f = 0.5 * (dvP + dvN)
        correction = d_f * ((pP - pN) / np.where(dmag > 1e-300, dmag, 1e-300)
                           - gradp_avg_dot_d)
        Vn = Vn_lin + correction
        return rho * Vn * f["len"]

    mdot_east  = _face(u, v, geom["east"],  axis=1)
    mdot_north = _face(u, v, geom["north"], axis=0)

    mdot_boundary = {}
    for name, sl in PATCH_SLICES.items():
        patch = mesh.patches[name]
        Vn = u[sl] * patch.nx_f + v[sl] * patch.ny_f
        mdot_boundary[name] = rho * Vn * patch.length

    return mdot_east, mdot_north, mdot_boundary, DVOL


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from wavy_channel_cfd.geometry.channel_gen import ChannelGeometry
    from wavy_channel_cfd.mesh.mesh_gen import StructuredMesh

    # ── 1. Straight mesh: non-orthogonal correction should vanish ────────
    geom_s = ChannelGeometry(L=0.06, H=0.01, lam=0.01, AR=0.0, phi_deg=0.0, n_pts=200)
    mesh_s = StructuredMesh(geom_s, nx=40, ny=20, n_inflation=4,
                            growth_rate=1.2, first_layer_height=2e-4)
    fvg_s = build_fv_geometry(mesh_s)
    assert np.allclose(fvg_s["east"]["Tx"], 0.0, atol=1e-12), "T_x should vanish (straight mesh)"
    assert np.allclose(fvg_s["east"]["Ty"], 0.0, atol=1e-12), "T_y should vanish (straight mesh)"
    assert np.allclose(fvg_s["north"]["Tx"], 0.0, atol=1e-12)
    assert np.allclose(fvg_s["north"]["Ty"], 0.0, atol=1e-12)
    print("Straight-mesh non-orthogonal correction is exactly zero. OK")

    # ── 2. Green-Gauss gradient of a linear field should match analytically ─
    ny, nx = mesh_s.ny, mesh_s.nx
    Xc, Yc = fvg_s["Xc"], fvg_s["Yc"]
    a, c, d0 = 3.0, -2.0, 1.5
    phi = a * Xc + c * Yc + d0
    bvals = {
        "inlet":       a * mesh_s.patches["inlet"].xf + c * mesh_s.patches["inlet"].yf + d0,
        "outlet":      a * mesh_s.patches["outlet"].xf + c * mesh_s.patches["outlet"].yf + d0,
        "wall_bottom": a * mesh_s.patches["wall_bottom"].xf + c * mesh_s.patches["wall_bottom"].yf + d0,
        "wall_top":    a * mesh_s.patches["wall_top"].xf + c * mesh_s.patches["wall_top"].yf + d0,
    }
    gx, gy = green_gauss_gradient(phi, fvg_s, mesh_s, bvals)
    # skip the outermost ring (Green-Gauss boundary treatment is 1st-order there)
    assert np.allclose(gx[1:-1, 1:-1], a, atol=1e-2), f"gx off: {gx[1:-1,1:-1].mean()} vs {a}"
    assert np.allclose(gy[1:-1, 1:-1], c, atol=1e-2), f"gy off: {gy[1:-1,1:-1].mean()} vs {c}"
    print("Green-Gauss gradient of a linear field matches analytically. OK")

    # ── 3. Pure-diffusion assembly reduces to a solvable, checkable system ──
    from wavy_channel_cfd.mesh.boundary import BCType
    bc_spec = {
        "inlet":       {"kind": "dirichlet", "value": 0.0},
        "outlet":      {"kind": "dirichlet", "value": 1.0},
        "wall_bottom": {"kind": "neumann", "flux": 0.0},
        "wall_top":    {"kind": "neumann", "flux": 0.0},
    }
    A, b = assemble_transport_matrix(mesh_s, fvg_s, gamma=1.0, bc=bc_spec,
                                     nonorth_correction=False)
    import scipy.sparse.linalg as spla
    T_sol = spla.spsolve(A, b).reshape(ny, nx)
    # 1-D conduction between two Dirichlet ends with adiabatic walls -> T
    # should vary linearly with x and be uniform across y.
    assert np.allclose(T_sol, np.broadcast_to(T_sol[0, :], (ny, nx)), atol=1e-8), \
        "pure axial conduction should be uniform across y"
    # Exact analytic solution for T(0)=0, T(L)=1, no source: T(x) = x/L,
    # evaluated at true cell centres (not endpoint-interpolated — the FV
    # cell centres sit half a cell inboard of the Dirichlet boundaries).
    lin_expected = Xc[0, :] / geom_s.L
    assert np.allclose(T_sol[0, :], lin_expected, atol=1e-9), "should vary linearly in x"
    print("Pure-diffusion assembly reproduces 1-D linear conduction. OK")

    sys.exit(0)
