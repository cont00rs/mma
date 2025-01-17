"""
GCMMA-MMA-Python

This file is part of GCMMA-MMA-Python. GCMMA-MMA-Python is licensed under the terms of GNU
General Public License as published by the Free Software Foundation. For more information and
the LICENSE file, see <https://github.com/arjendeetman/GCMMA-MMA-Python>.

The orginal work is written by Krister Svanberg in MATLAB. This is the Python implementation
of the code written by Arjen Deetman.

Functionality:
- `mmasub`: Solves the MMA subproblem.
- `subsolv`: Performs a primal-dual Newton method to solve subproblems.
- `kktcheck`: Checks the Karush-Kuhn-Tucker (KKT) conditions for the solution.

Dependencies:
- numpy: Numerical operations and array handling.
- scipy: Sparse matrix operations and linear algebra.

To use this module, import the desired functions and provide the necessary arguments
according to the specific problem being solved.
"""

# Loading modules
from __future__ import division

from typing import Tuple

import numpy as np

from mma.approximations import Approximations
from mma.bounds import Bounds, MMABounds
from mma.options import Coefficients, Options
from mma.subsolve import State, subsolv


def mma(
    x: np.ndarray,
    func: callable,
    bounds: Bounds,
    options: Options,
    coeff: Coefficients | None = None,
):
    """Driver of the MMA optimization.

    Reference material is available from:
    - https://people.kth.se/~krille/mmagcmma.pdf.
    """
    # Count constriants.
    _, _, fval, _ = func(x)
    m = 1 if isinstance(fval, float) else len(fval)
    n = len(x)

    # Initialisation.
    xval = x.copy()

    # Lower, upper bounds
    mma_bounds = MMABounds(bounds, options)
    coeff = coeff if coeff else Coefficients.from_defaults(m)

    kkttol = 0

    # Test output
    outvector1s = []
    outvector2s = []
    kktnorms = []

    # The iterations start
    kktnorm = kkttol + 10

    subproblem = SubProblem(options)

    for _ in range(options.iteration_count):
        if kktnorm <= kkttol:
            break

        f0val, df0dx, fval, dfdx = func(xval)

        # The MMA subproblem is solved at the point xval:
        state = subproblem.mmasub(
            m,
            n,
            xval,
            mma_bounds,
            coeff,
            df0dx,
            fval,
            dfdx,
        )

        # Some vectors are updated:
        xval = state.x.copy()

        # Re-calculate function values, gradients
        f0val, df0dx, fval, dfdx = func(xval)

        # The residual vector of the KKT conditions is calculated
        residu, kktnorm, residumax = kktcheck(
            m,
            n,
            state,
            mma_bounds.bounds,
            df0dx,
            fval,
            dfdx,
            coeff,
        )

        outvector1 = np.concatenate((f0val, fval))
        outvector2 = xval.flatten()
        outvector1s += [outvector1.flatten()]
        outvector2s += [outvector2]
        kktnorms += [kktnorm]
    else:
        count = options.iteration_count
        msg = f"MMA did not converge within iteration limit ({count})"
        print(msg)

    return np.array(outvector1s), np.array(outvector2s), np.array(kktnorms)


class SubProblem:
    def __init__(self, options: Options):
        self.options: Options = options

        # xold1 (np.ndarray): Design variables from one iteration ago.
        self.xold1 = None
        # xold2 (np.ndarray): Design variables from two iterations ago.
        self.xold2 = None

    def mmasub(
        self,
        m: int,
        n: int,
        xval: np.ndarray,
        bounds: MMABounds,
        coeff: Coefficients,
        df0dx: np.ndarray,
        fval: np.ndarray,
        dfdx: np.ndarray,
    ) -> State:
        """
        Solve the MMA (Method of Moving Asymptotes) subproblem for optimization.

        Minimize:
            f_0(x) + a_0 * z + sum(c_i * y_i + 0.5 * d_i * (y_i)^2)

        Subject to:
            f_i(x) - a_i * z - y_i <= 0,    i = 1,...,m
            xmin_j <= x_j <= xmax_j,        j = 1,...,n
            z >= 0, y_i >= 0,               i = 1,...,m

        Args:
            m (int): Number of constraints.
            n (int): Number of variables.
            xval (np.ndarray): Current values of the design variables.
            bounds (Bounds): Lower (xmin_j) and upper (xmax_j) bounds of the design variables.
            f0val (float): Objective function value at xval.
            df0dx (np.ndarray): Gradient of the objective function at xval.
            fval (np.ndarray): Constraint function values at xval.
            dfdx (np.ndarray): Gradient of the constraint functions at xval.
            low (np.ndarray): Lower bounds for the variables from the previous iteration (provided that iter > 1).
            upp (np.ndarray): Upper bounds for the variables from the previous iteration (provided that iter > 1).

        Returns:
            state (State)
        """
        # Calculation of the asymptotes low and upp.
        bounds.update_asymptotes(xval, self.xold1, self.xold2)

        # Calculation of the bounds alfa and beta.
        bounds.calculate_alpha_beta(xval)

        # Calculations approximating functions: P, Q.
        approx = Approximations(xval, df0dx, dfdx, bounds, self.options.raa0)

        # Negative residual.
        b = approx.residual(bounds, xval, fval)

        # Solving the subproblem using the primal-dual Newton method
        # FIXME: Move options for Newton method into dataclass.
        # b (np.ndarray): Right-hand side constants in the constraints.
        state = subsolv(m, n, bounds, approx, coeff, b)

        # Store design variables of last two iterations.
        self.xold2 = None if self.xold1 is None else self.xold1.copy()
        self.xold1 = xval.copy()

        return state


def kktcheck(
    m: int,
    n: int,
    state: State,
    bounds: Bounds,
    df0dx: np.ndarray,
    fval: np.ndarray,
    dfdx: np.ndarray,
    coeff: Coefficients,
) -> Tuple[np.ndarray, float, float]:
    """
    Evaluate the residuals for the Karush-Kuhn-Tucker (KKT) conditions of a nonlinear programming problem.

    The KKT conditions are necessary for optimality in constrained optimization problems. This function computes
    the residuals for these conditions based on the current values of the variables, constraints, and Lagrange multipliers.

    Args:
        m (int): Number of general constraints.
        n (int): Number of variables.
        state (State): Current state of the optimization problem.
        bounds (Bounds): Lower and upper bounds for the variables.
        df0dx (np.ndarray): Gradient of the objective function with respect to the variables.
        fval (np.ndarray): Values of the constraint functions.
        dfdx (np.ndarray): Jacobian matrix of the constraint functions.
        a0 (float): Coefficient for the term involving z in the objective function.
        a (np.ndarray): Coefficients for the terms involving z in the constraints.
        c (np.ndarray): Coefficients for the terms involving y in the constraints.
        d (np.ndarray): Coefficients for the quadratic terms involving y in the objective function.

    Returns:
        Tuple[np.ndarray, float, float]:
            - residu (np.ndarray): Residual vector for the KKT conditions.
            - residunorm (float): Norm of the residual vector.
            - residumax (float): Maximum absolute value among the residuals.
    """

    # Compute residuals for the KKT conditions
    rex = df0dx + np.dot(dfdx.T, state.lam) - state.xsi + state.eta
    rey = coeff.c + coeff.d * state.y - state.mu - state.lam
    rez = coeff.a0 - state.zet - np.dot(coeff.a.T, state.lam)
    relam = fval - coeff.a * state.z - state.y + state.s
    rexsi = state.xsi * (state.x - bounds.lower())
    reeta = state.eta * (bounds.upper() - state.x)
    remu = state.mu * state.y
    rezet = state.zet * state.z
    res = state.lam * state.s

    # Concatenate residuals into a single vector
    residu1 = np.concatenate((rex, rey, rez), axis=0)
    residu2 = np.concatenate((relam, rexsi, reeta, remu, rezet, res), axis=0)
    residu = np.concatenate((residu1, residu2), axis=0)

    # Calculate norm and maximum value of the residual vector
    residunorm = np.sqrt(np.dot(residu.T, residu).item())
    residumax = np.max(np.abs(residu))

    return residu, residunorm, residumax
