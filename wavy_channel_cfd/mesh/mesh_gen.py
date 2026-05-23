"""
mesh/mesh_gen.py

Constructs a structured 2-D finite-volume mesh for the wavy channel.

Responsibilities:
  - Accept a channel geometry (from geometry/channel_gen.py) and mesh
    resolution parameters (nx, ny).
  - Generate a body-fitted, algebraically stretched grid by transfinite
    interpolation between the bottom (wavy) and top (flat) walls.
  - Return node coordinates, cell-centre coordinates, face areas, and
    cell volumes needed by the SIMPLE solver.
  - Optionally export mesh data to a CSV or NumPy .npz file for inspection.
"""

import numpy as np


def _transfinite_grid(x_line: np.ndarray,
                      y_bottom: np.ndarray,
                      y_top: np.ndarray,
                      ny: int,
                      stretch_factor: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, Y) node arrays via transfinite interpolation with wall stretching.

    Parameters
    ----------
    x_line       : 1-D array of nx+1 x-node positions
    y_bottom     : y-coordinates of bottom wall at each x node
    y_top        : y-coordinates of top wall at each x node
    ny           : number of cells in the wall-normal direction
    stretch_factor: geometric stretching ratio toward walls

    Returns
    -------
    X, Y : 2-D arrays of shape (ny+1, nx+1)
    """
    nx1 = len(x_line)
    ny1 = ny + 1

    # Build stretched eta in [0, 1]
    j = np.arange(ny1, dtype=float)
    if stretch_factor == 1.0:
        eta = j / ny
    else:
        r = stretch_factor
        eta = (r**j - 1.0) / (r**ny - 1.0)

    X = np.tile(x_line, (ny1, 1))
    Y = np.outer(eta, y_top) + np.outer(1.0 - eta, y_bottom)
    return X, Y


def generate_mesh(geometry: dict, nx: int, ny: int,
                  stretch_factor: float = 1.5) -> dict:
    """Generate a structured mesh from the channel geometry.

    Parameters
    ----------
    geometry      : dict returned by geometry.channel_gen.build_channel
    nx            : number of cells in the streamwise direction
    ny            : number of cells in the wall-normal direction
    stretch_factor: geometric stretching ratio for wall-normal clustering

    Returns
    -------
    mesh dict with keys:
      node_x, node_y   – (ny+1, nx+1) node coordinate arrays
      cc_x, cc_y       – (ny, nx) cell-centre coordinate arrays
      dx, dy           – (ny, nx) representative cell spacings
      face_area_x      – (ny, nx+1) x-face areas (heights)
      face_area_y      – (ny+1, nx) y-face areas (widths)
      cell_volume      – (ny, nx) cell areas (2-D domain)
      nx, ny           – mesh counts
    """
    x_nodes = np.linspace(geometry["x"][0], geometry["x"][-1], nx + 1)

    # Interpolate wall y-coordinates onto the mesh x-nodes
    y_b_nodes = np.interp(x_nodes, geometry["x"], geometry["y_bottom"])
    y_t_nodes = np.interp(x_nodes, geometry["x"], geometry["y_top"])

    node_x, node_y = _transfinite_grid(x_nodes, y_b_nodes, y_t_nodes,
                                        ny, stretch_factor)

    # Cell-centre coordinates (average of four surrounding nodes)
    cc_x = 0.25 * (node_x[:-1, :-1] + node_x[:-1, 1:]
                   + node_x[1:, :-1] + node_x[1:, 1:])
    cc_y = 0.25 * (node_y[:-1, :-1] + node_y[:-1, 1:]
                   + node_y[1:, :-1] + node_y[1:, 1:])

    dx = node_x[:, 1:] - node_x[:, :-1]   # shape (ny+1, nx)
    dy = node_y[1:, :] - node_y[:-1, :]   # shape (ny, nx+1)

    cell_dx = 0.5 * (dx[:-1, :] + dx[1:, :])   # (ny, nx)
    cell_dy = 0.5 * (dy[:, :-1] + dy[:, 1:])   # (ny, nx)
    cell_volume = cell_dx * cell_dy

    face_area_x = dy                             # (ny, nx+1) – wall-normal extent
    face_area_y = dx                             # (ny+1, nx) – streamwise extent

    return {
        "node_x": node_x, "node_y": node_y,
        "cc_x": cc_x, "cc_y": cc_y,
        "cell_dx": cell_dx, "cell_dy": cell_dy,
        "face_area_x": face_area_x,
        "face_area_y": face_area_y,
        "cell_volume": cell_volume,
        "nx": nx, "ny": ny,
    }


def save_mesh(mesh: dict, path: str) -> None:
    """Save mesh arrays to a compressed NumPy file."""
    np.savez_compressed(path, **{k: v for k, v in mesh.items()
                                 if isinstance(v, np.ndarray)})
