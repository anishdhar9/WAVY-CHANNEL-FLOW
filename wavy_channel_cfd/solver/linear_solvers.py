"""
solver/linear_solvers.py

Thin wrapper around scipy.sparse.linalg so the direct-vs-iterative
choice for the per-field linear solves (u, v, p', T) inside the SIMPLE
outer loop is a parameter, not a code fork.

Default is a direct sparse LU factorization (scipy.sparse.linalg.spsolve),
appropriate for the mesh sizes used in this project (a few thousand to a
few tens of thousands of cells): it cleanly separates "does this
discretization satisfy its own equation" (answered to round-off by a
direct solve) from "did the outer SIMPLE loop converge" — useful while
bringing up a correctness-first discretization. If sweep performance
becomes a bottleneck, switch to method="iterative" (BiCGSTAB with an
ILU preconditioner and a warm start from the previous outer iteration)
without touching any assembly code.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve_linear(A: sp.spmatrix, b: np.ndarray,
                 x0: np.ndarray | None = None,
                 method: str = "direct",
                 tol: float = 1e-8,
                 maxiter: int = 200) -> np.ndarray:
    """Solve A x = b.

    Parameters
    ----------
    A       : sparse matrix (any format; converted to CSC for direct solves)
    b       : right-hand side
    x0      : initial guess, used only by iterative methods (warm start)
    method  : "direct" (scipy spsolve / SuperLU) or "iterative" (bicgstab
              with spilu preconditioning)
    tol, maxiter : iterative-solver controls (ignored for "direct")
    """
    if method == "direct":
        return spla.spsolve(A.tocsc(), b)

    if method == "iterative":
        A_csc = A.tocsc()
        try:
            ilu = spla.spilu(A_csc)
            M = spla.LinearOperator(A_csc.shape, ilu.solve)
        except RuntimeError:
            M = None
        x, info = spla.bicgstab(A_csc, b, x0=x0, rtol=tol, maxiter=maxiter, M=M)
        if info != 0:
            # Fall back to a direct solve rather than silently returning a
            # non-converged iterate.
            return spla.spsolve(A_csc, b)
        return x

    raise ValueError(f"Unknown method {method!r}; expected 'direct' or 'iterative'")


if __name__ == "__main__":
    import sys

    rng = np.random.default_rng(0)
    n = 200
    diag = 4.0 + rng.uniform(0, 1, n)
    off = -1.0 * np.ones(n - 1)
    A = sp.diags([off, diag, off], offsets=[-1, 0, 1], format="csr")
    x_true = rng.uniform(-1, 1, n)
    b = A @ x_true

    x_direct = solve_linear(A, b, method="direct")
    assert np.allclose(x_direct, x_true, atol=1e-8), "direct solve mismatch"
    print("Direct solve matches ground truth. OK")

    x_iter = solve_linear(A, b, method="iterative", x0=np.zeros(n))
    assert np.allclose(x_iter, x_true, atol=1e-6), "iterative solve mismatch"
    print("Iterative (bicgstab+ILU) solve matches ground truth. OK")

    sys.exit(0)
