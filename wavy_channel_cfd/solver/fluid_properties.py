"""
solver/fluid_properties.py

Thermophysical fluid properties for the wavy-channel solver.

Primary API: the `FluidProperties` dataclass, which holds the four
independent properties (rho, mu, k_f, Cp) and exposes derived quantities
(nu, Pr, alpha) as read-only properties.  All arithmetic is scalar; no
NumPy dependency in this module.

Backward-compatible helpers (`get_properties`, `water_properties_polynomial`)
are retained so that existing sweep/solver code that works with plain dicts
continues to function without modification.
"""

from __future__ import annotations
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Primary dataclass
# ---------------------------------------------------------------------------

@dataclass
class FluidProperties:
    """Immutable container for thermophysical properties of a single-phase fluid.

    Parameters
    ----------
    rho : density [kg/m³]
    mu  : dynamic viscosity [Pa·s]
    k_f : thermal conductivity [W/(m·K)]
    Cp  : specific heat capacity at constant pressure [J/(kg·K)]

    All four parameters must be strictly positive; a ``ValueError`` is raised
    in ``__post_init__`` otherwise.
    """

    rho: float
    mu:  float
    k_f: float
    Cp:  float

    def __post_init__(self) -> None:
        for name, val in (("rho", self.rho), ("mu", self.mu),
                          ("k_f", self.k_f), ("Cp",  self.Cp)):
            if val <= 0:
                raise ValueError(
                    f"FluidProperties.{name} must be positive, got {val!r}")

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def nu(self) -> float:
        """Kinematic viscosity  ν = μ / ρ  [m²/s]."""
        return self.mu / self.rho

    @property
    def Pr(self) -> float:
        """Prandtl number  Pr = μ · Cp / k  [dimensionless]."""
        return self.mu * self.Cp / self.k_f

    @property
    def alpha(self) -> float:
        """Thermal diffusivity  α = k / (ρ · Cp)  [m²/s]."""
        return self.k_f / (self.rho * self.Cp)

    # ------------------------------------------------------------------
    # Standard fluid factories
    # ------------------------------------------------------------------

    @classmethod
    def water_300K(cls) -> FluidProperties:
        """Standard liquid water at T = 300 K (≈ 27 °C).

        Reference: Incropera et al. *Fundamentals of Heat and Mass Transfer*,
        7th ed., Appendix A, Table A.6 at 300 K.

        Properties
        ----------
        ρ   = 996.5  kg/m³
        μ   = 8.55 × 10⁻⁴  Pa·s
        k   = 0.615  W/(m·K)
        Cp  = 4181   J/(kg·K)
        Pr  ≈ 5.81
        """
        return cls(rho=996.5, mu=8.55e-4, k_f=0.615, Cp=4181.0)

    # ------------------------------------------------------------------
    # Inlet velocity helper
    # ------------------------------------------------------------------

    def u_inlet_from_Re(self, Re: float, Dh: float) -> float:
        """Return the mean inlet velocity for a prescribed Reynolds number.

        Derived from the hydraulic-diameter definition:
            Re = ρ · U · Dh / μ   →   U = Re · μ / (ρ · Dh)

        Parameters
        ----------
        Re  : Reynolds number (must be > 0)
        Dh  : hydraulic diameter [m]  (Dh = 2H for infinite parallel plates)

        Returns
        -------
        U_inlet : mean streamwise inlet velocity [m/s]
        """
        if Re <= 0:
            raise ValueError(f"Re must be positive, got {Re!r}")
        if Dh <= 0:
            raise ValueError(f"Dh must be positive, got {Dh!r}")
        return Re * self.mu / (self.rho * Dh)

    # ------------------------------------------------------------------
    # Interoperability helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a flat dict for backward-compatibility with legacy solver code.

        Includes derived quantities so that old code using
        ``fluid["nu"]``, ``fluid["Pr"]``, ``fluid["alpha_th"]`` continues to work.
        """
        return {
            "rho":      self.rho,
            "mu":       self.mu,
            "k_f":      self.k_f,
            "Cp":       self.Cp,
            "nu":       self.nu,
            "Pr":       self.Pr,
            "alpha_th": self.alpha,
        }

    def print_summary(self) -> None:
        """Print a formatted property table to stdout."""
        print("FluidProperties:")
        print(f"  ρ    = {self.rho:10.4f}  kg/m³")
        print(f"  μ    = {self.mu:10.3e}  Pa·s")
        print(f"  k_f  = {self.k_f:10.4f}  W/(m·K)")
        print(f"  Cp   = {self.Cp:10.2f}  J/(kg·K)")
        print(f"  ν    = {self.nu:10.3e}  m²/s")
        print(f"  α    = {self.alpha:10.3e}  m²/s")
        print(f"  Pr   = {self.Pr:10.4f}  −")

    def __repr__(self) -> str:
        return (f"FluidProperties("
                f"rho={self.rho}, mu={self.mu:.3e}, "
                f"k_f={self.k_f}, Cp={self.Cp}, "
                f"nu={self.nu:.3e}, Pr={self.Pr:.4f})")


# ---------------------------------------------------------------------------
# Backward-compatible module-level helpers
# ---------------------------------------------------------------------------

def get_properties(config: dict | None = None, **kwargs) -> dict:
    """Return a validated fluid-property dict (legacy API).

    Wraps ``FluidProperties`` so that existing code receiving a plain dict
    from ``yaml.safe_load`` continues to work unchanged.
    """
    defaults: dict[str, float] = dict(rho=998.0, mu=0.001, k_f=0.6, Cp=4182.0)
    if config is not None:
        for key in defaults:
            if key in config:
                defaults[key] = float(config[key])
    defaults.update({k: v for k, v in kwargs.items() if k in defaults})
    return FluidProperties(**defaults).to_dict()


def water_properties_polynomial(T_C: float) -> dict:
    """Approximate temperature-dependent properties of water [10 – 80 °C].

    Simple polynomial curve-fits to NIST data for sensitivity studies.
    The default solver uses constant properties (``water_300K``).

    Parameters
    ----------
    T_C : temperature [°C]
    """
    if not (10.0 <= T_C <= 80.0):
        raise ValueError(
            f"Temperature {T_C} °C is outside the valid range [10, 80].")
    t = T_C
    rho = 999.84 - 0.0643 * t - 0.00366 * t**2
    mu  = 1.787e-3 * 10 ** (-0.01913 * (t - 20.0))
    k_f = 0.5715  + 0.00193 * t - 9.0e-6 * t**2
    Cp  = 4217.6  - 3.83   * t + 0.0388 * t**2
    return FluidProperties(rho=rho, mu=mu, k_f=k_f, Cp=Cp).to_dict()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    fluid = FluidProperties.water_300K()
    fluid.print_summary()
    print()

    # Check u_inlet for the Re values used in the sweep
    Dh = 2 * 0.010   # 2H, H = 10 mm
    print(f"{'Re':>6}   {'U_inlet [m/s]':>15}   {'Re_check':>10}")
    print("-" * 38)
    for Re in (100, 200, 400, 600, 800, 1200):
        U = fluid.u_inlet_from_Re(Re, Dh)
        Re_check = fluid.rho * U * Dh / fluid.mu
        print(f"{Re:>6}   {U:>15.5f}   {Re_check:>10.2f}")
    print()

    # Temperature-dependent polynomial
    print("Water properties at selected temperatures:")
    print(f"  {'T [°C]':>8}  {'rho':>8}  {'mu':>10}  {'Pr':>8}")
    for T in (20, 40, 60, 80):
        p = water_properties_polynomial(T)
        from wavy_channel_cfd.solver.fluid_properties import FluidProperties as FP
        fp = FP(rho=p["rho"], mu=p["mu"], k_f=p["k_f"], Cp=p["Cp"])
        print(f"  {T:>8}  {fp.rho:>8.2f}  {fp.mu:>10.3e}  {fp.Pr:>8.4f}")

    # Backward-compat dict API
    d = get_properties()
    assert "nu" in d and "Pr" in d and "alpha_th" in d
    print("\nget_properties() dict keys:", sorted(d.keys()))
    sys.exit(0)
