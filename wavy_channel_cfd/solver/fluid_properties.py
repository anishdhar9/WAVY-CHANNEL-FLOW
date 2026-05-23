"""
solver/fluid_properties.py

Defines and validates fluid (coolant) thermophysical properties.

Responsibilities:
  - Load fluid properties from the config dict or explicit arguments.
  - Provide a single `get_properties` function that returns a validated
    property dict used by the SIMPLE solver and post-processing modules.
  - Compute derived properties: kinematic viscosity (nu), thermal
    diffusivity (alpha), Prandtl number (Pr), and hydraulic-diameter-based
    Nusselt number scaling.
  - Optionally support simple temperature-dependent correlations for water
    in the range 10–80 °C (polynomial fits to NIST data).
"""


# Reference property table for liquid water (approximate, 20 °C)
_WATER_20C = {
    "rho": 998.0,    # kg/m³
    "mu":  0.001,    # Pa·s
    "k_f": 0.6,      # W/(m·K)
    "Cp":  4182.0,   # J/(kg·K)
}


def get_properties(config: dict | None = None, **kwargs) -> dict:
    """Return a validated fluid-property dict.

    Property values are taken from *config* (a dict or sub-dict of the
    parsed YAML config) with individual keyword arguments taking precedence.
    Missing keys fall back to 20 °C water defaults.

    Returns
    -------
    dict with keys: rho, mu, k_f, Cp, nu, alpha_th, Pr
    """
    base = dict(_WATER_20C)
    if config is not None:
        for key in ("rho", "mu", "k_f", "Cp"):
            if key in config:
                base[key] = float(config[key])
    base.update({k: v for k, v in kwargs.items() if k in base})

    _validate(base)
    base["nu"]       = base["mu"]  / base["rho"]
    base["alpha_th"] = base["k_f"] / (base["rho"] * base["Cp"])
    base["Pr"]       = base["mu"]  * base["Cp"] / base["k_f"]
    return base


def _validate(props: dict) -> None:
    """Raise ValueError for non-physical property values."""
    for key in ("rho", "mu", "k_f", "Cp"):
        if props[key] <= 0:
            raise ValueError(
                f"Fluid property '{key}' must be positive, got {props[key]}")


def print_properties(fluid: dict) -> None:
    """Print a formatted summary of the fluid properties."""
    print("Fluid properties:")
    print(f"  rho    = {fluid['rho']:.2f}   kg/m³")
    print(f"  mu     = {fluid['mu']:.6f} Pa·s")
    print(f"  k_f    = {fluid['k_f']:.4f} W/(m·K)")
    print(f"  Cp     = {fluid['Cp']:.2f}   J/(kg·K)")
    print(f"  nu     = {fluid['nu']:.2e} m²/s")
    print(f"  alpha  = {fluid['alpha_th']:.2e} m²/s")
    print(f"  Pr     = {fluid['Pr']:.4f}")


def water_properties_polynomial(T_C: float) -> dict:
    """Approximate temperature-dependent properties of water [10–80 °C].

    Simple polynomial curve-fits to NIST data.  Intended for sensitivity
    studies; the default solver uses constant properties.

    Parameters
    ----------
    T_C : temperature [°C]
    """
    if not (10.0 <= T_C <= 80.0):
        raise ValueError(f"Temperature {T_C} °C is outside the valid range [10, 80].")

    t = T_C
    rho  = 999.84 - 0.0643 * t - 0.00366 * t**2
    mu   = 1.787e-3 * 10 ** (-0.01913 * (t - 20.0))
    k_f  = 0.5715 + 0.00193 * t - 9.0e-6 * t**2
    Cp   = 4217.6 - 3.83 * t + 0.0388 * t**2

    return get_properties(rho=rho, mu=mu, k_f=k_f, Cp=Cp)
