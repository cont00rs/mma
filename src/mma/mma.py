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

import numpy as np

from mma.approximations import Approximations
from mma.bounds import Bounds, MMABounds
from mma.options import Coefficients, Options
from mma.subsolve import State, subsolv
from mma.target_function import TargetFunction


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
    # Initialise at first iterate.
    target_function = TargetFunction(func, x.copy())
    coeff = coeff if coeff else Coefficients.from_defaults(target_function.m)

    # Lower, upper bounds
    mma_bounds = MMABounds(bounds, options)

    kkttol = 0

    # Test output
    outvector1s = []
    outvector2s = []
    kktnorms = []

    # The iterations start
    kktnorm = kkttol + 10

    for _ in range(options.iteration_count):
        if kktnorm <= kkttol:
            break

        # The MMA subproblem is solved at the current point (`xval`).
        state = mmasub(target_function, mma_bounds, coeff, options)

        # Re-calculate function values, gradients at next iterate.
        target_function.evaluate(state.x)

        # The residual vector of the KKT conditions is calculated
        kktnorm = kktcheck(state, bounds, target_function, coeff)

        outvector1 = np.concatenate((target_function.f0, target_function.f))
        outvector2 = target_function.x.flatten()
        outvector1s += [outvector1.flatten()]
        outvector2s += [outvector2]
        kktnorms += [kktnorm]
    else:
        count = options.iteration_count
        msg = f"MMA did not converge within iteration limit ({count})"
        print(msg)

    return np.array(outvector1s), np.array(outvector2s), np.array(kktnorms)


def mmasub(
    target_function: TargetFunction,
    bounds: MMABounds,
    coeff: Coefficients,
    options: Options,
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
        bounds (Bounds)
        coeff (Coefficients)
        target_function (TargetFunction)

    Returns:
        state (State)
    """
    # Calculation of the asymptotes low and upp.
    bounds.update_asymptotes(target_function)

    # Calculation of the bounds alfa and beta.
    bounds.calculate_alpha_beta(target_function.x)

    # Calculations approximating functions: P, Q.
    # TODO: Consider caching approximations and only updating its contents?
    approx = Approximations(target_function, bounds, options.raa0)

    # Solving the subproblem using the primal-dual Newton method
    state = subsolv(target_function, bounds, approx, coeff, options)

    # Store design variables of last two iterations.
    target_function.store()

    return state


def kktcheck(
    state: State,
    bounds: Bounds,
    target_function: TargetFunction,
    coeff: Coefficients,
) -> float:
    """
    Evaluate the residuals for the Karush-Kuhn-Tucker (KKT) conditions of a nonlinear programming problem.

    The KKT conditions are necessary for optimality in constrained optimization problems. This function computes
    the residuals for these conditions based on the current values of the variables, constraints, and Lagrange multipliers.

    Args:
        state (State): Current state of the optimization problem.
        bounds (Bounds): Lower and upper bounds for the variables.
        target_function (TargetFunction)
        coeff (Coefficients)

    Returns:
        float:
            - residunorm (float): Norm of the residual vector.
    """

    # Compute residuals for the KKT conditions
    df0dx = target_function.df0dx
    dfdx = target_function.dfdx

    state = State.from_variables(
        df0dx + dfdx.T @ state.lam - state.xsi + state.eta,
        coeff.c + coeff.d * state.y - state.mu - state.lam,
        coeff.a0 - state.zet - coeff.a.T @ state.lam,
        target_function.f - coeff.a * state.z - state.y + state.s,
        state.xsi * (state.x - bounds.lower()),
        state.eta * (bounds.upper() - state.x),
        state.mu * state.y,
        state.zet * state.z,
        state.lam * state.s,
    )

    residunorm, _ = state.residual()
    return residunorm
