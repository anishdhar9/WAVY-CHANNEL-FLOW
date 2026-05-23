"""
mesh/mesh_gen.py

Structured quadrilateral mesh generator for the wavy channel.

Node layout convention:
  X[i, j], Y[i, j]  —  i is the streamwise index (0…nx),
                          j is the wall-normal index (0…ny).
  Cell (i, j) has its SW corner at node (i, j) and NE corner at (i+1, j+1).

Wall-normal distribution uses geometric inflation layers from both walls
that meet a uniform core region.  All construction is fully vectorised
over the (nx+1) streamwise stations — no Python loops over cells.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from wavy_channel_cfd.geometry.channel_gen import ChannelGeometry


class StructuredMesh:
    """Body-fitted structured quad mesh with dual-wall inflation layers.

    Parameters
    ----------
    geom               : ChannelGeometry instance
    nx                 : streamwise cell count
    ny                 : wall-normal cell count
    n_inflation        : inflation layers from each wall  (default 12)
    growth_rate        : geometric growth ratio r  (default 1.2)
    first_layer_height : height of the very first inflation cell [m]

    Node layout (all arrays with i=streamwise, j=wall-normal)
    ----------
    X, Y               : (nx+1, ny+1) node coordinates [m]
    Xc, Yc             : (nx,   ny)   cell-centre coordinates [m]
    dA                 : (nx,   ny)   cell areas [m²]

    Face geometry (one entry per cell, positive outward sense):
    face_len_{e,w,n,s}  : (nx, ny)  edge lengths [m]
    face_nx_{e,w,n,s}   : (nx, ny)  x-component of unit outward normal
    face_ny_{e,w,n,s}   : (nx, ny)  y-component of unit outward normal

    Wall quantities:
    dy_wall_bottom      : (nx,)  first-cell height at bottom wall [m]
    dy_wall_top         : (nx,)  first-cell height at top wall [m]
    """

    def __init__(self, geom: ChannelGeometry,
                 nx: int, ny: int,
                 n_inflation: int = 12,
                 growth_rate: float = 1.2,
                 first_layer_height: float = 2e-5):
        self.geom  = geom
        self.nx    = int(nx)
        self.ny    = int(ny)
        self.n_inf = int(n_inflation)
        self.r     = float(growth_rate)
        self.h1    = float(first_layer_height)
        self._build()

    # ------------------------------------------------------------------
    # Internal construction (fully vectorised)
    # ------------------------------------------------------------------

    def _build(self) -> None:
        nx, ny, n_inf, r, h1 = self.nx, self.ny, self.n_inf, self.r, self.h1
        n_core = ny - 2 * n_inf

        if n_core < 1:
            raise ValueError(
                f"Too few core cells: n_core = {n_core} "
                f"(ny={ny}, n_inflation={n_inf}). "
                f"Require ny > 2 × n_inflation."
            )

        # ── 1. Streamwise node x-positions ───────────────────────────
        x_nodes = np.linspace(0.0, self.geom.L, nx + 1)          # (nx+1,)
        y_b     = np.interp(x_nodes, self.geom.x, self.geom.y_bottom)  # (nx+1,)
        y_t     = np.interp(x_nodes, self.geom.x, self.geom.y_top)     # (nx+1,)
        H_local = y_t - y_b                                       # (nx+1,)

        # ── 2. Inflation layer offsets (shared across all columns) ────
        # offsets[k] = cumulative distance after k cells from the wall
        #   offsets[0] = 0     (wall node)
        #   offsets[1] = h1    (after 1st cell)
        #   offsets[k] = h1 * (r^k − 1) / (r − 1)
        k        = np.arange(n_inf + 1, dtype=float)              # (n_inf+1,)
        if abs(r - 1.0) < 1e-10:
            offsets = h1 * k
        else:
            offsets = h1 * (r**k - 1.0) / (r - 1.0)              # (n_inf+1,)

        infl_thickness = float(offsets[-1])
        H_min = float(np.min(H_local))
        if 2.0 * infl_thickness >= H_min * 0.99:
            raise ValueError(
                f"Inflation layers overlap: 2 × {infl_thickness*1e3:.3f} mm "
                f"≥ min channel gap {H_min*1e3:.3f} mm. "
                f"Reduce first_layer_height or n_inflation."
            )

        # ── 3. Vectorised wall-normal node positions ──────────────────
        #
        # Y_bot[i, k] = y_b[i] + offsets[k]
        #   k=0 → wall, k=n_inf → top of bottom inflation
        Y_bot = y_b[:, None] + offsets[None, :]                   # (nx+1, n_inf+1)

        # Y_top[i, k] = y_t[i] − offsets[n_inf − k]
        #   k=0 → bottom of top inflation, k=n_inf → top wall
        Y_top = y_t[:, None] - offsets[None, ::-1]                # (nx+1, n_inf+1)

        # Core: uniform between Y_bot[:,-1] and Y_top[:,0]
        core_start = Y_bot[:, -1]                                  # (nx+1,)
        core_end   = Y_top[:, 0]                                   # (nx+1,)
        t_core     = np.linspace(0.0, 1.0, n_core + 1)            # (n_core+1,)
        Y_core     = (core_start[:, None]
                      + t_core[None, :] * (core_end - core_start)[:, None])
        # shape (nx+1, n_core+1)

        # Concatenate, excluding shared boundary nodes:
        #   Y_bot         → j = 0 … n_inf          (n_inf+1 nodes)
        #   Y_core[:,1:-1] → j = n_inf+1 … ny−n_inf−1  (n_core−1 nodes)
        #   Y_top          → j = ny−n_inf … ny     (n_inf+1 nodes)
        # Total: (n_inf+1) + (n_core−1) + (n_inf+1) = ny+1 ✓
        Y = np.concatenate([Y_bot, Y_core[:, 1:-1], Y_top], axis=1)
        # shape: (nx+1, ny+1)

        X = np.broadcast_to(x_nodes[:, None], (nx + 1, ny + 1)).copy()
        # shape: (nx+1, ny+1)

        self.X = X
        self.Y = Y

        # ── 4. Cell centres ───────────────────────────────────────────
        self.Xc = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[:-1, 1:] + X[1:, 1:])
        self.Yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[:-1, 1:] + Y[1:, 1:])
        # shape: (nx, ny)

        # ── 5. Cell areas via diagonal cross-product ──────────────────
        # d1 = NE − SW diagonal, d2 = NW − SE diagonal
        d1x = X[1:, 1:] - X[:-1, :-1]
        d1y = Y[1:, 1:] - Y[:-1, :-1]
        d2x = X[:-1, 1:] - X[1:, :-1]
        d2y = Y[:-1, 1:] - Y[1:, :-1]
        self.dA = 0.5 * np.abs(d1x * d2y - d1y * d2x)            # (nx, ny)

        # ── 6. Face lengths and outward unit normals ──────────────────
        # Convention: for a face from node A to node B,
        #   tangent = B − A,  outward = clockwise rotation for E/S faces,
        #                               counter-clockwise rotation for W/N faces.
        #
        # East face  — edge from (i+1, j) → (i+1, j+1)
        dx_e = X[1:, 1:] - X[1:, :-1]   # ≈ 0 (x constant along j)
        dy_e = Y[1:, 1:] - Y[1:, :-1]
        self.face_len_east = np.sqrt(dx_e**2 + dy_e**2)
        _le = self.face_len_east + 1e-300
        self.face_nx_east  =  dy_e / _le   # clockwise: (ty, -tx)
        self.face_ny_east  = -dx_e / _le   # → (+1, 0) for flat channel ✓

        # West face  — edge from (i, j) → (i, j+1)
        dx_w = X[:-1, 1:] - X[:-1, :-1]  # ≈ 0
        dy_w = Y[:-1, 1:] - Y[:-1, :-1]
        self.face_len_west = np.sqrt(dx_w**2 + dy_w**2)
        _lw = self.face_len_west + 1e-300
        self.face_nx_west  = -dy_w / _lw   # counter-clockwise: (-ty, tx)
        self.face_ny_west  =  dx_w / _lw   # → (-1, 0) for flat channel ✓

        # South face — edge from (i, j) → (i+1, j)
        dx_s = X[1:, :-1] - X[:-1, :-1]
        dy_s = Y[1:, :-1] - Y[:-1, :-1]
        self.face_len_south = np.sqrt(dx_s**2 + dy_s**2)
        _ls = self.face_len_south + 1e-300
        self.face_nx_south  =  dy_s / _ls   # clockwise: (ty, -tx)
        self.face_ny_south  = -dx_s / _ls   # → (0, -1) for flat channel ✓

        # North face — edge from (i, j+1) → (i+1, j+1)
        dx_n = X[1:, 1:] - X[:-1, 1:]
        dy_n = Y[1:, 1:] - Y[:-1, 1:]
        self.face_len_north = np.sqrt(dx_n**2 + dy_n**2)
        _ln = self.face_len_north + 1e-300
        self.face_nx_north  = -dy_n / _ln   # counter-clockwise: (-ty, tx)
        self.face_ny_north  =  dx_n / _ln   # → (0, +1) for flat channel ✓

        # ── 7. Wall-adjacent first-cell heights ───────────────────────
        # Average west and east face heights at j=0 (bottom) and j=ny-1 (top)
        self.dy_wall_bottom = 0.5 * (
            (Y[:-1, 1] - Y[:-1, 0]) + (Y[1:, 1] - Y[1:, 0]))    # (nx,)
        self.dy_wall_top    = 0.5 * (
            (Y[:-1, ny] - Y[:-1, ny - 1])
            + (Y[1:, ny] - Y[1:, ny - 1]))                        # (nx,)

    # ------------------------------------------------------------------
    # Mesh quality metrics
    # ------------------------------------------------------------------

    def mesh_quality(self) -> dict:
        """Compute and return mesh quality metrics as a dict.

        Returns
        -------
        dict with keys:
          n_cells
          aspect_ratio_{min,max,mean}
          skewness_{min,max,mean}        (0 = perfect rectangle, 1 = degenerate)
          orthogonality_{min,max,mean}   (1 = fully orthogonal)
        """
        X, Y = self.X, self.Y

        # ── Aspect ratio: longest edge / shortest edge per cell ───────
        edges = np.stack([self.face_len_east,  self.face_len_west,
                          self.face_len_south, self.face_len_north],
                         axis=-1)                                  # (nx, ny, 4)
        AR_cell = edges.max(axis=-1) / (edges.min(axis=-1) + 1e-300)

        # ── Equi-angle skewness ───────────────────────────────────────
        # EAS = max(θ_max − 90°, 90° − θ_min) / 90°
        # 0 = perfect rectangle, 1 = degenerate.
        # Edge vectors from SW corner of each cell
        AB_x = X[1:, :-1] - X[:-1, :-1]; AB_y = Y[1:, :-1] - Y[:-1, :-1]  # → SE
        AD_x = X[:-1, 1:] - X[:-1, :-1]; AD_y = Y[:-1, 1:] - Y[:-1, :-1]  # → NW
        BC_x = X[1:, 1:]  - X[1:, :-1];  BC_y = Y[1:, 1:]  - Y[1:, :-1]   # SE → NE
        CD_x = X[:-1, 1:] - X[1:, 1:];   CD_y = Y[:-1, 1:] - Y[1:, 1:]    # NE → NW

        _lenAB = np.sqrt(AB_x**2 + AB_y**2) + 1e-300
        _lenAD = np.sqrt(AD_x**2 + AD_y**2) + 1e-300
        _lenBC = np.sqrt(BC_x**2 + BC_y**2) + 1e-300
        _lenCD = np.sqrt(CD_x**2 + CD_y**2) + 1e-300

        cos_A = np.clip((AB_x*AD_x + AB_y*AD_y)    / (_lenAB*_lenAD), -1, 1)
        cos_B = np.clip((-AB_x*BC_x - AB_y*BC_y)   / (_lenAB*_lenBC), -1, 1)
        cos_C = np.clip((-BC_x*CD_x - BC_y*CD_y)   / (_lenBC*_lenCD), -1, 1)
        cos_D = np.clip((CD_x*AD_x  + CD_y*AD_y)   / (_lenCD*_lenAD), -1, 1)

        thetas = np.degrees(np.arccos(
            np.stack([cos_A, cos_B, cos_C, cos_D], axis=-1)))   # (nx, ny, 4)
        theta_max = thetas.max(axis=-1)
        theta_min = thetas.min(axis=-1)
        skew = np.maximum(theta_max - 90.0, 90.0 - theta_min) / 90.0  # (nx, ny)

        # ── Orthogonality: cell-centre connection vs east-face normal ─
        # Evaluated at all nx−1 interior east-face interfaces
        if self.nx > 1:
            conn_x  = self.Xc[1:, :] - self.Xc[:-1, :]           # (nx-1, ny)
            conn_y  = self.Yc[1:, :] - self.Yc[:-1, :]
            conn_mag = np.sqrt(conn_x**2 + conn_y**2) + 1e-300
            # East-face normal at the shared interface (face of left cell)
            nx_f = self.face_nx_east[:-1, :]                       # (nx-1, ny)
            ny_f = self.face_ny_east[:-1, :]
            orth = (conn_x * nx_f + conn_y * ny_f) / conn_mag
        else:
            orth = np.ones((1, self.ny))

        def _stats(a):
            return float(a.min()), float(a.max()), float(a.mean())

        ar_min,  ar_max,  ar_mean  = _stats(AR_cell)
        sk_min,  sk_max,  sk_mean  = _stats(skew)
        or_min,  or_max,  or_mean  = _stats(orth)

        return {
            "n_cells":             self.nx * self.ny,
            "aspect_ratio_min":    ar_min,
            "aspect_ratio_max":    ar_max,
            "aspect_ratio_mean":   ar_mean,
            "skewness_min":        sk_min,
            "skewness_max":        sk_max,
            "skewness_mean":       sk_mean,
            "orthogonality_min":   or_min,
            "orthogonality_max":   or_max,
            "orthogonality_mean":  or_mean,
        }

    def print_quality(self) -> None:
        """Print a formatted mesh quality report."""
        q = self.mesh_quality()
        w = self.dy_wall_bottom
        print(f"Mesh quality  ({self.nx}×{self.ny} = {q['n_cells']:,} cells,  "
              f"n_inf={self.n_inf},  r={self.r},  h₁={self.h1*1e3:.4f} mm)")
        print(f"  Aspect ratio   : "
              f"min={q['aspect_ratio_min']:.2f}   "
              f"mean={q['aspect_ratio_mean']:.2f}   "
              f"max={q['aspect_ratio_max']:.2f}")
        print(f"  Skewness       : "
              f"min={q['skewness_min']:.5f}  "
              f"mean={q['skewness_mean']:.5f}  "
              f"max={q['skewness_max']:.5f}  "
              f"(0=ideal)")
        print(f"  Orthogonality  : "
              f"min={q['orthogonality_min']:.5f}  "
              f"mean={q['orthogonality_mean']:.5f}  "
              f"max={q['orthogonality_max']:.5f}  "
              f"(1=ideal)")
        print(f"  dy_wall_bottom : "
              f"min={w.min()*1e6:.2f}  "
              f"mean={w.mean()*1e6:.2f}  "
              f"max={w.max()*1e6:.2f}  [µm]")

    # ------------------------------------------------------------------
    # Mesh plot
    # ------------------------------------------------------------------

    def plot_mesh(self, every_n: int = 5,
                  save_path: str | Path | None = None) -> plt.Figure:
        """Plot every nth gridline in both directions.

        Parameters
        ----------
        every_n   : stride for gridline selection (default 5)
        save_path : if given, save as PNG (parent dirs created automatically)

        Returns
        -------
        matplotlib Figure
        """
        Xmm = self.X * 1e3   # convert to mm for readability
        Ymm = self.Y * 1e3

        fig, ax = plt.subplots(figsize=(13, 3.8))

        # ── Streamwise gridlines: constant i, vary j ──────────────────
        # X[::every_n, :].T → shape (ny+1, n_lines); each column is one line
        ax.plot(Xmm[::every_n, :].T, Ymm[::every_n, :].T,
                color="steelblue", lw=0.35, alpha=0.7)

        # ── Wall-normal gridlines: constant j, vary i ─────────────────
        # X[:, ::every_n] → shape (nx+1, n_lines); each column is one line
        ax.plot(Xmm[:, ::every_n], Ymm[:, ::every_n],
                color="steelblue", lw=0.35, alpha=0.7)

        # ── Boundary highlights ───────────────────────────────────────
        ax.plot(Xmm[:, 0],  Ymm[:, 0],  color="royalblue",  lw=1.8,
                label="bottom wall")
        ax.plot(Xmm[:, -1], Ymm[:, -1], color="firebrick",   lw=1.8,
                label="top wall")
        ax.plot(Xmm[0, :],  Ymm[0, :],  color="seagreen",    lw=1.8,
                ls="--", label="inlet")
        ax.plot(Xmm[-1, :], Ymm[-1, :], color="darkorange",  lw=1.8,
                ls="--", label="outlet")

        # ── Inflation zone shading ────────────────────────────────────
        # Bottom inflation band
        ax.fill_between(
            Xmm[:, 0],
            Ymm[:, 0],
            Ymm[:, self.n_inf],
            color="royalblue", alpha=0.10, label=f"inflation ({self.n_inf} layers)")
        # Top inflation band
        ax.fill_between(
            Xmm[:, 0],
            Ymm[:, self.ny - self.n_inf],
            Ymm[:, self.ny],
            color="firebrick", alpha=0.10)

        ax.set_xlabel("x  [mm]", fontsize=10)
        ax.set_ylabel("y  [mm]", fontsize=10)
        ax.set_title(
            f"Structured mesh  —  {self.nx}×{self.ny}  "
            f"(n_inf={self.n_inf}, r={self.r}, "
            f"h₁={self.h1*1e3:.3f} mm)  |  every {every_n}th gridline",
            fontsize=9)
        ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.75)
        ax.set_xlim(Xmm.min(), Xmm.max())
        ax.set_ylim(Ymm.min() - 0.4, Ymm.max() + 0.4)
        ax.set_aspect("auto")
        ax.grid(False)
        fig.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return fig

    # ------------------------------------------------------------------
    # NPZ export
    # ------------------------------------------------------------------

    def save_npz(self, path: str | Path) -> None:
        """Save all node and derived arrays to a compressed .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=self.X, Y=self.Y, Xc=self.Xc, Yc=self.Yc, dA=self.dA,
            face_len_east=self.face_len_east,   face_len_west=self.face_len_west,
            face_len_south=self.face_len_south,  face_len_north=self.face_len_north,
            face_nx_east=self.face_nx_east,     face_nx_west=self.face_nx_west,
            face_nx_south=self.face_nx_south,    face_nx_north=self.face_nx_north,
            face_ny_east=self.face_ny_east,     face_ny_west=self.face_ny_west,
            face_ny_south=self.face_ny_south,    face_ny_north=self.face_ny_north,
            dy_wall_bottom=self.dy_wall_bottom,  dy_wall_top=self.dy_wall_top,
        )

    # ------------------------------------------------------------------
    # Standard mesh factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def coarse(cls, geom: ChannelGeometry) -> "StructuredMesh":
        """150 × 40 mesh, first layer h₁ = 0.04 mm."""
        return cls(geom, nx=150, ny=40,
                   n_inflation=12, growth_rate=1.2,
                   first_layer_height=4e-5)

    @classmethod
    def medium(cls, geom: ChannelGeometry) -> "StructuredMesh":
        """300 × 80 mesh, first layer h₁ = 0.02 mm."""
        return cls(geom, nx=300, ny=80,
                   n_inflation=12, growth_rate=1.2,
                   first_layer_height=2e-5)

    @classmethod
    def fine(cls, geom: ChannelGeometry) -> "StructuredMesh":
        """600 × 160 mesh, first layer h₁ = 0.01 mm."""
        return cls(geom, nx=600, ny=160,
                   n_inflation=12, growth_rate=1.2,
                   first_layer_height=1e-5)

    def __repr__(self) -> str:
        return (f"StructuredMesh(nx={self.nx}, ny={self.ny}, "
                f"n_inf={self.n_inf}, r={self.r}, "
                f"h1={self.h1*1e3:.4f}mm, "
                f"cells={self.nx*self.ny:,})")


# ---------------------------------------------------------------------------
# Backward-compatible helper (used by solver modules built on old mesh dict)
# ---------------------------------------------------------------------------

def generate_mesh(geometry: dict, nx: int, ny: int,
                  stretch_factor: float = 1.5) -> dict:
    """Legacy wrapper: build a StructuredMesh and return as a plain dict.

    The StructuredMesh uses geometric inflation layers rather than the
    old single-sided geometric stretch, so 'stretch_factor' is used to
    derive first_layer_height = H * (stretch_factor - 1) / stretch_factor.
    """
    from wavy_channel_cfd.geometry.channel_gen import ChannelGeometry
    L   = float(geometry["x"][-1])
    H   = float(np.mean(geometry["y_top"] - geometry["y_bottom"]))
    lam = L / max(round(L / H), 1)

    geom = ChannelGeometry.__new__(ChannelGeometry)
    geom.L        = L
    geom.H        = H
    geom.lam      = lam
    geom.AR       = 0.0
    geom.phi_deg  = 0.0
    geom.n_pts    = len(geometry["x"])
    geom.A        = 0.0
    geom.phi_rad  = 0.0
    geom.x        = geometry["x"]
    geom.y_bottom = geometry["y_bottom"]
    geom.y_top    = geometry["y_top"]

    n_inf = min(12, ny // 4)
    h1    = H * (1.0 - 1.0 / stretch_factor) / max(n_inf, 1)
    mesh  = StructuredMesh(geom, nx=nx, ny=ny,
                           n_inflation=n_inf, growth_rate=1.2,
                           first_layer_height=h1)

    return {
        "node_x":      mesh.X.T,
        "node_y":      mesh.Y.T,
        "cc_x":        mesh.Xc.T,
        "cc_y":        mesh.Yc.T,
        "cell_dx":     (mesh.face_len_south + mesh.face_len_north).T * 0.5,
        "cell_dy":     (mesh.face_len_west  + mesh.face_len_east).T  * 0.5,
        "face_area_x": mesh.face_len_east.T,
        "face_area_y": mesh.face_len_south.T,
        "cell_volume": mesh.dA.T,
        "nx": nx, "ny": ny,
    }


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    OUT = Path(__file__).parent.parent / "results" / "mesh"
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Straight-channel medium mesh  (300 × 80)")
    print("=" * 60)

    from wavy_channel_cfd.geometry.channel_gen import ChannelGeometry

    geom = ChannelGeometry(L=0.060, H=0.010, lam=0.010,
                           AR=0.0, phi_deg=0.0, n_pts=500)
    geom.check_validity()

    t0   = time.perf_counter()
    mesh = StructuredMesh.medium(geom)
    dt   = time.perf_counter() - t0
    print(f"Build time: {dt*1e3:.1f} ms")
    print(repr(mesh))
    print()

    mesh.print_quality()
    print()

    # Spot-check: inflation layer heights at inlet column
    col = 0
    print("Wall-normal node y-positions at inlet (first 5 and last 5) [µm]:")
    y_col = mesh.Y[col, :] * 1e6
    print("  bottom-5:", np.round(y_col[:5], 1))
    print("  top-5:   ", np.round(y_col[-5:], 1))
    print(f"  First layer (bottom): {mesh.dy_wall_bottom[col]*1e6:.2f} µm  "
          f"(target {mesh.h1*1e6:.2f} µm)")
    print(f"  First layer (top):    {mesh.dy_wall_top[col]*1e6:.2f} µm")
    print()

    # Verify face normal orientation for interior cell
    i, j = mesh.nx // 2, mesh.ny // 2
    print(f"Face normals at cell ({i},{j}):")
    print(f"  East : nx={mesh.face_nx_east[i,j]:+.6f}  ny={mesh.face_ny_east[i,j]:+.6f}")
    print(f"  West : nx={mesh.face_nx_west[i,j]:+.6f}  ny={mesh.face_ny_west[i,j]:+.6f}")
    print(f"  South: nx={mesh.face_nx_south[i,j]:+.6f}  ny={mesh.face_ny_south[i,j]:+.6f}")
    print(f"  North: nx={mesh.face_nx_north[i,j]:+.6f}  ny={mesh.face_ny_north[i,j]:+.6f}")
    print()

    save_path = OUT / "medium_straight.png"
    mesh.plot_mesh(every_n=10, save_path=save_path)
    print(f"Mesh plot saved → {save_path}")

    mesh.save_npz(OUT / "medium_straight.npz")
    print(f"Mesh data saved → {OUT / 'medium_straight.npz'}")

    sys.exit(0)
