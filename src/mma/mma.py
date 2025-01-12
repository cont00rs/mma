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
from scipy.sparse import diags_array

from mma.bounds import Bounds, MMABounds
from mma.options import Options
from mma.subsolve import State, subsolv


def mma(
    x: np.ndarray,
    func: callable,
    bounds: Bounds,
    options: Options,
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
    bounds = MMABounds(bounds, options)

    c = 1000 * np.ones((m, 1))

    a0 = 1

    # This implementations assumes `a_i = 0` and `d_i = 1`
    # for all i to match the basic problem formulation as
    # defined in equation (1.2) in mmagcmma.pdf.
    a = np.zeros((m, 1))
    d = np.ones((m, 1))

    kkttol = 0

    # Test output
    outvector1s = []
    outvector2s = []
    kktnorms = []

    # The iterations start
    kktnorm = kkttol + 10

    # FIXME: Should c, a0, a, d also be considered "options"?
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
            bounds,
            f0val,
            df0dx,
            fval,
            dfdx,
            a0,
            a,
            c,
            d,
        )

        # Some vectors are updated:
        xval = state.x.copy()

        # Re-calculate function values, gradients
        f0val, df0dx, fval, dfdx = func(xval)

        # The residual vector of the KKT conditions is calculated
        residu, kktnorm, residumax = kktcheck(
            m,
            n,
            state.x,
            state.y,
            state.z,
            state.lam,
            state.xsi,
            state.eta,
            state.mu,
            state.zet,
            state.s,
            bounds.bounds,
            df0dx,
            fval,
            dfdx,
            a0,
            a,
            c,
            d,
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
        self.options = options

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
        f0val: float,
        df0dx: np.ndarray,
        fval: np.ndarray,
        dfdx: np.ndarray,
        a0: float,
        a: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
    ) -> Tuple[
        State,
        np.ndarray,
        np.ndarray,
    ]:
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
            a0 (float): Constant in the term a_0 * z.
            a (np.ndarray): Coefficients for the term a_i * z.
            c (np.ndarray): Coefficients for the term c_i * y_i.
            d (np.ndarray): Coefficients for the term 0.5 * d_i * (y_i)^2.

        Returns:
            Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
                - xmma (np.ndarray): Optimal values of the design variables.
                - ymma (np.ndarray): Optimal values of the slack variables for constraints.
                - zmma (float): Optimal value of the regularization variable z.
                - lam (np.ndarray): Lagrange multipliers for the constraints.
                - xsi (np.ndarray): Lagrange multipliers for the lower bounds on design variables.
                - eta (np.ndarray): Lagrange multipliers for the upper bounds on design variables.
                - mu (np.ndarray): Lagrange multipliers for the slack variables of the constraints.
                - zet (float): Lagrange multiplier for the regularization term z.
                - s (np.ndarray): Slack variables for the general constraints.
                - low (np.ndarray): Updated lower bounds for the design variables.
                - upp (np.ndarray): Updated upper bounds for the design variables.
        """
        # Calculation of the asymptotes low and upp.
        bounds.update_asymptotes(xval, self.xold1, self.xold2)

        # Calculation of the bounds alfa and beta.
        bounds.calculate_alpha_beta(xval)

        # Calculations approximating functions: P, Q.
        p0, q0 = self.approximating_functions(
            xval, df0dx, bounds, objective=True
        )

        P, Q = self.approximating_functions(
            xval, dfdx, bounds, objective=False
        )

        # Negative residual between approximating functions and objective
        # as described in beginning of Section 5.
        # TODO: Move this into the State class. It can be computed on the fly?
        b = (
            P @ (1 / (bounds.upp - xval))
            + Q @ (1 / (xval - bounds.low))
            - fval
        )

        # Solving the subproblem using the primal-dual Newton method
        # FIXME: Move options for Newton method into dataclass.
        state = subsolv(m, n, bounds, p0, q0, P, Q, a0, a, b, c, d)

        # Store design variables of last two iterations.
        self.xold2 = None if self.xold1 is None else self.xold1.copy()
        self.xold1 = xval.copy()

        return state

    def approximating_functions(
        self, xval, dfdx, bounds: MMABounds, objective=True
    ):
        """Calculate approximating functions "P" and "Q".

        Build components for approximation of objective and
        constraint functions from lower/upper asymptotes and
        current derivative information.

        The routine calculations Equations 3.2 - 3.5 for the
        objective function (f_0) and constraints (f_1 ... f_n).

        Due to the layout of the constraint derivatives an
        additional transpose is needed, controlled through the
        objective keyword argument.
        """

        factor = 0.001

        # Inverse bounds with eps to avoid divide by zero.
        # Last component of equations 3.3 and 3.4.
        eps_delta = 0.00001
        delta_inv = 1 / np.maximum(bounds.bounds.delta(), eps_delta)

        if objective:
            df_plus = np.maximum(dfdx, 0)
            df_minus = np.maximum(-dfdx, 0)
        else:
            df_plus = np.maximum(dfdx.T, 0)
            df_minus = np.maximum(-dfdx.T, 0)

        # Equation 3.3.
        p0 = (1 + factor) * df_plus + factor * df_minus
        p0 += self.options.raa0 * delta_inv
        p0 = diags_array(((bounds.upp - xval) ** 2).squeeze(axis=1)) @ p0

        # Equation 3.4.
        q0 = factor * df_plus + (1 + factor) * df_minus
        q0 += self.options.raa0 * delta_inv
        q0 = diags_array(((xval - bounds.low) ** 2).squeeze(axis=1)) @ q0

        if objective:
            return p0, q0
        else:
            return p0.T, q0.T


def kktcheck(
    m: int,
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    z: float,
    lam: np.ndarray,
    xsi: np.ndarray,
    eta: np.ndarray,
    mu: np.ndarray,
    zet: float,
    s: np.ndarray,
    bounds: Bounds,
    df0dx: np.ndarray,
    fval: np.ndarray,
    dfdx: np.ndarray,
    a0: float,
    a: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """
    Evaluate the residuals for the Karush-Kuhn-Tucker (KKT) conditions of a nonlinear programming problem.

    The KKT conditions are necessary for optimality in constrained optimization problems. This function computes
    the residuals for these conditions based on the current values of the variables, constraints, and Lagrange multipliers.

    Args:
        m (int): Number of general constraints.
        n (int): Number of variables.
        x (np.ndarray): Current values of the variables.
        y (np.ndarray): Current values of the general constraints' slack variables.
        z (float): Current value of the single variable in the problem.
        lam (np.ndarray): Lagrange multipliers for the general constraints.
        xsi (np.ndarray): Lagrange multipliers for the lower bound constraints on variables.
        eta (np.ndarray): Lagrange multipliers for the upper bound constraints on variables.
        mu (np.ndarray): Lagrange multipliers for the non-negativity constraints on slack variables.
        zet (float): Lagrange multiplier for the non-negativity constraint on z.
        s (np.ndarray): Slack variables for the general constraints.
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
    rex = df0dx + np.dot(dfdx.T, lam) - xsi + eta
    rey = c + d * y - mu - lam
    rez = a0 - zet - np.dot(a.T, lam)
    relam = fval - a * z - y + s
    rexsi = xsi * (x - bounds.lower())
    reeta = eta * (bounds.upper() - x)
    remu = mu * y
    rezet = zet * z
    res = lam * s

    # Concatenate residuals into a single vector
    residu1 = np.concatenate((rex, rey, rez), axis=0)
    residu2 = np.concatenate((relam, rexsi, reeta, remu, rezet, res), axis=0)
    residu = np.concatenate((residu1, residu2), axis=0)

    # Calculate norm and maximum value of the residual vector
    residunorm = np.sqrt(np.dot(residu.T, residu).item())
    residumax = np.max(np.abs(residu))

    return residu, residunorm, residumax
