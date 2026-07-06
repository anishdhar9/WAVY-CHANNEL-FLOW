"""
mesh/boundary.py

Boundary patch system for the structured 2-D mesh.

A mesh boundary is decomposed into named *patches* (inlet, outlet,
wall_bottom, wall_top), each carrying:

  - the boundary condition type (BCType),
  - the discrete face geometry along that patch: face centres, unit
    outward normals, and face lengths,
  - optional physical parameters (heat flux, wall temperature, …).

Patches are built by StructuredMesh at construction time and consumed by
solver.boundary_conditions.BoundaryConditions, so the mesh — not the
solver — is the single source of truth for where each boundary lives and
what condition applies there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class BCType(Enum):
    """Supported boundary condition types."""

    VELOCITY_INLET  = "velocity-inlet"    # Dirichlet u, v, T
    PRESSURE_OUTLET = "pressure-outlet"   # Neumann u, v, T; Dirichlet p_ref
    WALL_HEATFLUX   = "wall-heatflux"     # no-slip + constant q'' (Neumann T)
    WALL_ADIABATIC  = "wall-adiabatic"    # no-slip + dT/dn = 0
    WALL_ISOTHERMAL = "wall-isothermal"   # no-slip + fixed T (Dirichlet)
    SYMMETRY        = "symmetry"          # zero normal velocity, zero gradients

    def is_wall(self) -> bool:
        return self in (BCType.WALL_HEATFLUX,
                        BCType.WALL_ADIABATIC,
                        BCType.WALL_ISOTHERMAL)

    def __str__(self) -> str:
        return self.value


# Index-space side of the structured block each patch occupies.
# west  : i = 0        (first streamwise column)
# east  : i = nx - 1   (last streamwise column)
# south : j = 0        (bottom wall row)
# north : j = ny - 1   (top wall row)
VALID_SIDES = ("west", "east", "south", "north")


@dataclass
class BoundaryPatch:
    """One contiguous boundary patch of the structured mesh.

    Attributes
    ----------
    name    : patch identifier, e.g. "inlet", "wall_bottom"
    side    : index-space side — one of "west", "east", "south", "north"
    bc_type : boundary condition type applied on this patch
    xf, yf  : (n_faces,) face-centre coordinates [m]
    nx_f, ny_f : (n_faces,) unit outward normal components
    length  : (n_faces,) face lengths [m]
    params  : BC parameters, e.g. {"q_flux": 5000.0} or {"T_wall": 350.0}
    """

    name:    str
    side:    str
    bc_type: BCType
    xf:      np.ndarray
    yf:      np.ndarray
    nx_f:    np.ndarray
    ny_f:    np.ndarray
    length:  np.ndarray
    params:  dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.side not in VALID_SIDES:
            raise ValueError(
                f"Patch {self.name!r}: side must be one of {VALID_SIDES}, "
                f"got {self.side!r}")
        n = len(self.xf)
        for attr in ("yf", "nx_f", "ny_f", "length"):
            if len(getattr(self, attr)) != n:
                raise ValueError(
                    f"Patch {self.name!r}: {attr} has length "
                    f"{len(getattr(self, attr))}, expected {n}")

    # ------------------------------------------------------------------

    @property
    def n_faces(self) -> int:
        return len(self.xf)

    @property
    def total_length(self) -> float:
        """Total arc length of the patch [m]."""
        return float(np.sum(self.length))

    def set_bc(self, bc_type: BCType, **params) -> None:
        """Reassign the BC type (and parameters) of this patch."""
        if not isinstance(bc_type, BCType):
            raise TypeError(f"bc_type must be a BCType, got {type(bc_type)}")
        self.bc_type = bc_type
        self.params  = dict(params)

    def __repr__(self) -> str:
        extras = "".join(f", {k}={v:g}" for k, v in self.params.items()
                         if isinstance(v, (int, float)))
        return (f"BoundaryPatch({self.name!r}, side={self.side!r}, "
                f"bc={self.bc_type.value!r}, n_faces={self.n_faces}, "
                f"L={self.total_length*1e3:.2f}mm{extras})")


def default_patch_bc() -> dict[str, tuple[BCType, dict]]:
    """Project-default BC assignment for the four channel patches.

    inlet       : uniform velocity inlet
    outlet      : pressure outlet (zero-gradient + reference pressure)
    wall_bottom : no-slip, constant heat flux (value supplied by the solver
                  BC object or via patch params)
    wall_top    : no-slip, adiabatic
    """
    return {
        "inlet":       (BCType.VELOCITY_INLET,  {}),
        "outlet":      (BCType.PRESSURE_OUTLET, {}),
        "wall_bottom": (BCType.WALL_HEATFLUX,   {}),
        "wall_top":    (BCType.WALL_ADIABATIC,  {}),
    }
