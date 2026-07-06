"""
main.py

Entry point for the wavy-channel CFD simulation framework.

Responsibilities:
  - Parse command-line arguments and load config.yaml.
  - Dispatch to one of three execution modes:
      single  – run a single case with user-specified Re, AR, phi
      sweep   – run the full (Re × AR × phi) parameter sweep
      validate – run straight-channel validation against analytical solutions
  - Print a run summary and elapsed time on completion.

Usage
-----
  python main.py                          # full sweep (default)
  python main.py --mode single --Re 400 --AR 0.10 --phi 90
  python main.py --mode validate
  python main.py --mode sweep --parallel
"""

import argparse
import time
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def run_single(config: dict, Re: float, AR: float, phi: float) -> None:
    """Run a single CFD case and print key metrics."""
    from wavy_channel_cfd.geometry.channel_gen import ChannelGeometry
    from wavy_channel_cfd.mesh.mesh_gen import StructuredMesh
    from wavy_channel_cfd.solver.boundary_conditions import BoundaryConditions
    from wavy_channel_cfd.solver.fluid_properties import FluidProperties
    from wavy_channel_cfd.solver.simple import solve
    from wavy_channel_cfd.postprocess.extract_metrics import extract_all

    fluid = FluidProperties(rho=config["rho"], mu=config["mu"],
                            k_f=config["k_f"], Cp=config["Cp"])
    geom  = ChannelGeometry(L=config["L"], H=config["H"], lam=config["lambda"],
                            AR=AR, phi_deg=phi)
    geom.check_validity()
    mesh  = StructuredMesh.for_yplus(geom, Re=Re, rho=fluid.rho, mu=fluid.mu,
                                     y_plus=1.0, nx=80, ny=40)
    bc    = BoundaryConditions(fluid, Re=Re, q_flux=config["q_flux"],
                               Dh=2.0 * config["H"])

    fluid.print_summary()
    bc.summary()

    u, v, p, T, monitor = solve(mesh, bc, fluid,
                                 {"max_iter": 1000, "tol": 1e-6},
                                 verbose=True)
    fields  = {"u": u, "v": v, "p": p, "T": T}
    metrics = extract_all(fields, mesh, fluid, bc, geom)

    print("\nResults:")
    for k, val in metrics.items():
        print(f"  {k:15s} = {val:.4g}")


def run_sweep_mode(config: dict, parallel: bool = False) -> None:
    """Run the full parameter sweep."""
    from wavy_channel_cfd.sweep.run_sweep import run_sweep
    run_sweep(config, parallel=parallel)


def run_validate(config: dict) -> None:
    """Run straight-channel validation."""
    from wavy_channel_cfd.validation.validate_straight import run_validation
    print("Running straight-channel validation …")
    df = run_validation(config)
    n_pass = df["passed"].sum()
    print(f"\n{n_pass}/{len(df)} cases PASSED.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wavy-channel CFD parameter sweep")
    parser.add_argument("--config",   default=str(CONFIG_PATH),
                        help="Path to config.yaml")
    parser.add_argument("--mode",     choices=["single", "sweep", "validate"],
                        default="sweep")
    parser.add_argument("--Re",       type=float, default=400)
    parser.add_argument("--AR",       type=float, default=0.10)
    parser.add_argument("--phi",      type=float, default=0.0)
    parser.add_argument("--parallel", action="store_true",
                        help="Parallelise sweep with ProcessPoolExecutor")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    t0 = time.perf_counter()
    if args.mode == "single":
        run_single(config, args.Re, args.AR, args.phi)
    elif args.mode == "sweep":
        run_sweep_mode(config, parallel=args.parallel)
    elif args.mode == "validate":
        run_validate(config)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f} s.")


if __name__ == "__main__":
    main()
