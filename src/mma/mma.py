"""The core MMA implementation."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from mma.approximations import Approximations
from mma.bounds import Bounds, MMABounds
from mma.options import Coefficients, Options
from mma.subsolve import State, subsolv
from mma.target_function import TargetFunction


@dataclass
class OptimizationResult:
    """Result of a single MMA iteration."""

    state: State
    target_function: TargetFunction
    kktnorm: float


def mma(
    x: np.ndarray,
    func: Callable,
    bounds: Bounds,
    options: Options,
    coeff: Coefficients | None = None,
    callback: Callable | None = None,
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
    result = None

    # The iterations start
    kktnorm: float = 10.0

    for _ in range(options.iteration_count):
        if kktnorm <= 0:
            break

        # The MMA subproblem is solved at the current point (`xval`).
        state = mmasub(target_function, mma_bounds, coeff, options)

        # Re-calculate function values, gradients at next iterate.
        target_function.evaluate(state.x)

        # The residual vector of the KKT conditions is calculated
        kktnorm = kktcheck(state, bounds, target_function, coeff)

        # Collect (intermediate) results.
        result = OptimizationResult(state, target_function, kktnorm)

        # Evaluate user-defined callback functions.
        if callback:
            callback(result)

    else:
        count = options.iteration_count
        msg = f"MMA did not converge within iteration limit ({count})"
        print(msg)

    return result


def mmasub(
    target_function: TargetFunction,
    bounds: MMABounds,
    coeff: Coefficients,
    options: Options,
) -> State:
    """Solve the MMA (Method of Moving Asymptotes) subproblem for optimization.

    Minimize:
        f_0(x) + a_0 * z + sum(c_i * y_i + 0.5 * d_i * (y_i)^2)

    Subject to:
        f_i(x) - a_i * z - y_i <= 0,    i = 1,...,m
        xmin_j <= x_j <= xmax_j,        j = 1,...,n
        z >= 0, y_i >= 0,               i = 1,...,m

    Parameters
    ----------
        target_function: TargetFunction
        bounds: Bounds
        coeff: Coefficients
        options: Options

    Returns
    -------
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
    """Evaluate the residuals for the Karush-Kuhn-Tucker (KKT) conditions.

    The KKT conditions are necessary for optimality in constrained
    optimization problems. This function computes the residuals for
    these conditions based on the current values of the variables,
    constraints, and Lagrange multipliers.

    Args:
        state (State): Current state of the optimization problem.
        bounds (Bounds): Lower and upper bounds for the variables.
        target_function (TargetFunction)
        coeff (Coefficients)

    Returns
    -------
        float:
            residunorm (float): Norm of the residual vector.

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
