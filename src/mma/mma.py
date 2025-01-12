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

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.linalg import solve  # or use numpy: from numpy.linalg import solve
from scipy.sparse import (
    diags,
    diags_array,
    issparse,
)


class Bounds:
    """Container for the variable bounds."""

    def __init__(self, lb=-np.inf, ub=np.inf):
        if issparse(lb) or issparse(ub):
            message = "Lower and upper bounds must be dense arrays."
            raise ValueError(message)

        self.lb = np.atleast_1d(lb)
        self.ub = np.atleast_1d(ub)

    def lower(self):
        return self.lb

    def upper(self):
        return self.ub

    def delta(self):
        return self.ub - self.lb


@dataclass
class Options:
    """
    MMA Algorithm options

    Attributes:
        iteration_count: Maximum number of outer iterations.
        move_limit: Move limit for the design variables.
        asyinit: Factor to calculate the initial distance of the asymptotes.
        asydecr: Factor by which the asymptotes distance is decreased.
        asyincr: Factor by which the asymptotes distance is increased.
        asymin: Factor to calculate the minimum distance of the asymptotes.
        asymax: Factor to calculate the maximum distance of the asymptotes.
        raa0: Parameter representing the function approximation's accuracy.
        alpha_factor: Factor to calculate the bounds alpha.
        beta_factor: Factor to calculate the bounds beta.
    """

    iteration_count: int
    move_limit: float = 0.5
    asyinit: float = 0.5
    asydecr: float = 0.7
    asyincr: float = 1.2
    asymin: float = 0.01
    asymax: float = 10
    raa0: float = 0.00001
    beta_factor: float = 0.1
    alpha_factor: float = 0.1


class MMABounds:
    def __init__(self, bounds: Bounds, options: Options):
        self.bounds = bounds
        self.options = options

        # Initialise (low, upp) to the bounds of the problem.
        self.low = bounds.lower()
        self.upp = bounds.upper()

    def update_asymptotes(self, xval, xold1, xold2):
        """Calculation of the asymptotes low and upp.

        This represents equations 3.11 to 3.14.
        """

        # The difference between upper and lower bounds.
        delta = self.bounds.delta()

        if xold1 is None or xold2 is None:
            # Equation 3.11.
            self.low = xval - self.options.asyinit * delta
            self.upp = xval + self.options.asyinit * delta
            return

        # Extract sign of variable change from previous iterates.
        signs = (xval - xold1) * (xold1 - xold2)

        # Assign increase/decrease factoring depending on signs. Equation 3.13.
        factor = np.ones_like(xval, dtype=float)
        factor[signs > 0] = self.options.asyincr
        factor[signs < 0] = self.options.asydecr

        # Equation 3.12.
        self.low = xval - factor * (xold1 - self.low)
        self.upp = xval + factor * (self.upp - xold1)

        # Limit asymptote change to maximum increase/decrease. Equation 3.14.
        np.clip(
            self.low,
            a_min=xval - self.options.asymax * delta,
            a_max=xval - self.options.asymin * delta,
            out=self.low,
        )

        np.clip(
            self.upp,
            a_min=xval + self.options.asymin * delta,
            a_max=xval + self.options.asymax * delta,
            out=self.upp,
        )

    def calculate_alpha_beta(self, xval):
        """Calculation of the bounds alpha and beta.

        Equations 3.6 and 3.7.
        """

        # Restrict lower bound with move limit.
        lower_bound = np.maximum(
            self.low + self.options.alpha_factor * (xval - self.low),
            xval - self.options.move_limit * self.bounds.delta(),
        )

        # Restrict upper bound with move limit.
        upper_bound = np.minimum(
            self.upp - self.options.beta_factor * (self.upp - xval),
            xval + self.options.move_limit * self.bounds.delta(),
        )

        # Restrict bounds with variable bounds.
        self.alpha = np.maximum(lower_bound, self.bounds.lb)
        self.beta = np.minimum(upper_bound, self.bounds.ub)


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


class State:
    """State representation for the Newton problem.

    Section 5.3.
        - xmma (np.ndarray): Optimal values of the variables x_j.
        - ymma (np.ndarray): Optimal values of the variables y_i.
        - zmma (float): Optimal value of the variable z.
        - slack (np.ndarray): Slack variables for the general MMA constraints.
        - lagrange (np.ndarray): Lagrange multipliers for the constraints.
    """

    def __init__(self, x, y, z, lam, xsi, eta, mu, zet, s):
        self.x = x
        self.y = y
        self.z = z
        self.lam = lam
        self.xsi = xsi
        self.eta = eta
        self.mu = mu
        self.zet = zet
        self.s = s

    @classmethod
    def from_alpha_beta(cls, n, m, bounds, c):
        x = (bounds.alpha + bounds.beta) / 2
        y = np.ones((m, 1))
        z = np.array([[1.0]])
        lam = np.ones((m, 1))
        xsi = np.maximum(1 / (x - bounds.alpha), 1)
        eta = np.maximum(1 / (bounds.beta - x), 1)
        mu = np.maximum(np.ones((m, 1)), 0.5 * c)
        zet = np.array([[1.0]])
        s = np.ones((m, 1))

        return State(x, y, z, lam, xsi, eta, mu, zet, s)

    def copy(self):
        return State(
            self.x.copy(),
            self.y.copy(),
            self.z.copy(),
            self.lam.copy(),
            self.xsi.copy(),
            self.eta.copy(),
            self.mu.copy(),
            self.zet.copy(),
            self.s.copy(),
        )

    def __add__(self, other):
        assert isinstance(other, State)

        return State(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.lam + other.lam,
            self.xsi + other.xsi,
            self.eta + other.eta,
            self.mu + other.mu,
            self.zet + other.zet,
            self.s + other.s,
        )

    def scale(self, value: int | float):
        """Scale all state variables."""
        self.x *= value
        self.y *= value
        self.z *= value
        self.lam *= value
        self.xsi *= value
        self.eta *= value
        self.mu *= value
        self.zet *= value
        self.s *= value
        return self

    def relaxed_residual(self, a0, a, b, c, d, p0, q0, P, Q, bounds, epsi):
        """Calculate residuals of the relaxed equations.

        The state equations are converted to their "relaxed" form,
        see Equations 5.9*, and the residuals are obtained from the
        full state vector of all equations.
        """

        plam = p0 + P.T @ self.lam
        qlam = q0 + Q.T @ self.lam
        gvec = P @ (1 / (bounds.upp - self.x)) + Q @ (
            1 / (self.x - bounds.low)
        )
        dpsidx = (
            plam / (bounds.upp - self.x) ** 2
            - qlam / (self.x - bounds.low) ** 2
        )

        relaxed = State(
            dpsidx - self.xsi + self.eta,
            c + d * self.y - self.mu - self.lam,
            a0 - self.zet - a.T @ self.lam,
            gvec - a * self.z - self.y + self.s - b,
            self.xsi * (self.x - bounds.alpha) - epsi,
            self.eta * (bounds.beta - self.x) - epsi,
            self.mu * self.y - epsi,
            self.zet * self.z - epsi,
            self.lam * self.s - epsi,
        )

        residual = np.concatenate(
            (
                relaxed.x,
                relaxed.y,
                relaxed.z,
                relaxed.lam,
                relaxed.xsi,
                relaxed.eta,
                relaxed.mu,
                relaxed.zet,
                relaxed.s,
            ),
            axis=0,
        )

        norm = np.sqrt(np.dot(residual.T, residual).item())
        norm_max = np.max(np.abs(residual))

        return norm, norm_max


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


def subsolv(
    m: int,
    n: int,
    bounds: MMABounds,
    p0: np.ndarray,
    q0: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    a0: float,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> State:
    """
    Solve the MMA (Method of Moving Asymptotes) subproblem for optimization.

    Minimize:
        SUM[p0j/(uppj-xj) + q0j/(xj-lowj)] + a0*z + SUM[ci*yi + 0.5*di*(yi)^2]

    Subject to:
        SUM[pij/(uppj-xj) + qij/(xj-lowj)] - ai*z - yi <= bi,
        alfa_j <= xj <= beta_j, yi >= 0, z >= 0.

    Args:
        m (int): Number of constraints.
        n (int): Number of variables.
        low (np.ndarray): Lower bounds for the variables x_j.
        upp (np.ndarray): Upper bounds for the variables x_j.
        alfa (np.ndarray): Lower asymptotes for the variables.
        beta (np.ndarray): Upper asymptotes for the variables.
        p0 (np.ndarray): Coefficients for the lower bound terms.
        q0 (np.ndarray): Coefficients for the upper bound terms.
        P (np.ndarray): Matrix of coefficients for the lower bound terms in the constraints.
        Q (np.ndarray): Matrix of coefficients for the upper bound terms in the constraints.
        a0 (float): Constant term in the objective function.
        a (np.ndarray): Coefficients for the constraints involving z.
        b (np.ndarray): Right-hand side constants in the constraints.
        c (np.ndarray): Coefficients for the terms involving y in the constraints.
        d (np.ndarray): Coefficients for the quadratic terms involving y in the objective function.

    Returns:
        State
    """

    # Initial problem state as given in Section 5.5 beginning.
    epsi = 1

    state = State.from_alpha_beta(n, m, bounds, c)

    # A small positive number to ensure numerical stability.
    epsimin = 1e-7
    iteration_count = 200

    # Start while loop for numerical stability
    while epsi > epsimin:
        # Start inner while loop for optimization
        for _ in range(iteration_count):
            # Compute relaxed optimality conditions, Section 5.2.
            residunorm, residumax = state.relaxed_residual(
                a0, a, b, c, d, p0, q0, P, Q, bounds, epsi
            )

            if residumax <= 0.9 * epsi:
                break

            d_state = solve_newton_step(
                state,
                # bounds
                bounds,
                # approximations
                p0,
                q0,
                P,
                Q,
                # parameters
                a0,
                a,
                b,
                c,
                d,
                epsi,
            )

            state = line_search(
                # bounds
                bounds,
                # approximations
                p0,
                q0,
                P,
                Q,
                state,
                d_state,
                # parameters
                a0,
                a,
                b,
                c,
                d,
                epsi,
            )

        epsi = 0.1 * epsi

    return state


# FIXME: Match this to interface of `line_search`.
def solve_newton_step(
    state,
    bounds: MMABounds,
    # approximations
    p0,
    q0,
    P,
    Q,
    # parameters
    a0,
    a,
    b,
    c,
    d,
    epsi,
):
    ux1 = bounds.upp - state.x
    xl1 = state.x - bounds.low
    ux2 = ux1 * ux1
    xl2 = xl1 * xl1
    ux3 = ux1 * ux2
    xl3 = xl1 * xl2
    uxinv1 = 1 / ux1
    xlinv1 = 1 / xl1
    uxinv2 = 1 / ux2
    xlinv2 = 1 / xl2
    plam = p0 + np.dot(P.T, state.lam)
    qlam = q0 + np.dot(Q.T, state.lam)
    gvec = np.dot(P, uxinv1) + np.dot(Q, xlinv1)
    GG = (diags(uxinv2.flatten(), 0).dot(P.T)).T - (
        diags(xlinv2.flatten(), 0).dot(Q.T)
    ).T
    dpsidx = plam / ux2 - qlam / xl2
    delx = (
        dpsidx
        - epsi / (state.x - bounds.alpha)
        + epsi / (bounds.beta - state.x)
    )
    dely = c + d * state.y - state.lam - epsi / state.y
    delz = a0 - np.dot(a.T, state.lam) - epsi / state.z
    dellam = gvec - a * state.z - state.y - b + epsi / state.lam
    diagx = plam / ux3 + qlam / xl3
    diagx = (
        2 * diagx
        + state.xsi / (state.x - bounds.alpha)
        + state.eta / (bounds.beta - state.x)
    )
    diagxinv = 1 / diagx
    diagy = d + state.mu / state.y
    diagyinv = 1 / diagy
    diaglam = state.s / state.lam
    diaglamyi = diaglam + diagyinv

    # Solve system of equations
    # The size of design variables and constraint functions play
    # a role in determining how to solve the primal-dual problem.
    # The paper expands on this at the end of Section 5.3.
    # TODO: Consider to move more of that discussion here.

    n, m = len(state.x), len(state.y)

    if n > m:
        # Delta x is eliminated (Equation 5.19) and the system
        # of equations in delta lambda, delta z is constructed.
        # This represents Equation 5.20.
        blam = dellam + dely / diagy - np.dot(GG, (delx / diagx))
        bb = np.concatenate((blam, delz), axis=0)
        Alam = np.asarray(
            diags(diaglamyi.flatten(), 0)
            + (diags(diagxinv.flatten(), 0).dot(GG.T).T).dot(GG.T)
        )
        AAr1 = np.concatenate((Alam, a), axis=1)
        AAr2 = np.concatenate((a, -state.zet / state.z), axis=0).T
        AA = np.concatenate((AAr1, AAr2), axis=0)
        solut = solve(AA, bb)
        dlam = solut[0:m]

        dz = solut[m : m + 1]
        dx = -delx / diagx - np.dot(GG.T, dlam) / diagx
    else:
        # Delta lambda is eliminated (Equation 5.21) and the system
        # of equations in delta x, delta z is constructed. This
        # represents Equation 5.22.
        diaglamyiinv = 1 / diaglamyi
        dellamyi = dellam + dely / diagy
        Axx = np.asarray(
            diags(diagx.flatten(), 0)
            + (diags(diaglamyiinv.flatten(), 0).dot(GG).T).dot(GG)
        )
        azz = state.zet / state.z + np.dot(a.T, (a / diaglamyi))
        axz = np.dot(-GG.T, (a / diaglamyi))
        bx = delx + np.dot(GG.T, (dellamyi / diaglamyi))
        bz = delz - np.dot(a.T, (dellamyi / diaglamyi))
        AAr1 = np.concatenate((Axx, axz), axis=1)
        AAr2 = np.concatenate((axz.T, azz), axis=1)
        AA = np.concatenate((AAr1, AAr2), axis=0)
        bb = np.concatenate((-bx, -bz), axis=0)
        solut = solve(AA, bb)
        dx = solut[0:n]
        dz = solut[n : n + 1]
        dlam = (
            np.dot(GG, dx) / diaglamyi
            - dz * (a / diaglamyi)
            + dellamyi / diaglamyi
        )

    # Back substitute the solutions found to reconstruct the full
    # solutions of delta's. This is the solution of a Newton step
    # as specified at the start of Section 5.3.
    dy = -dely / diagy + dlam / diagy
    dxsi = (
        -state.xsi
        + epsi / (state.x - bounds.alpha)
        - (state.xsi * dx) / (state.x - bounds.alpha)
    )
    deta = (
        -state.eta
        + epsi / (bounds.beta - state.x)
        + (state.eta * dx) / (bounds.beta - state.x)
    )
    dmu = -state.mu + epsi / state.y - (state.mu * dy) / state.y
    dzet = -state.zet + epsi / state.z - state.zet * dz / state.z
    ds = -state.s + epsi / state.lam - (state.s * dlam) / state.lam

    return State(dx, dy, dz, dlam, dxsi, deta, dmu, dzet, ds)


def line_search(
    # bounds
    bounds: MMABounds,
    # approximations
    p0,
    q0,
    P,
    Q,
    state,
    d_state,
    # parameters
    a0,
    a,
    b,
    c,
    d,
    epsi,
):
    """Line search along Newton descent direction.


    Line search into the Newton direction, Section 5.4.
    This aims to find a step into the Newton direction without
    violating any of the constraints, which might happen when taking
    the full Newton step. The specifications of a "good" step are
    indicated in Section 5.4.
    """

    xx = np.concatenate(
        (
            state.y,
            state.z,
            state.lam,
            state.xsi,
            state.eta,
            state.mu,
            state.zet,
            state.s,
        ),
        axis=0,
    )

    dxx = np.concatenate(
        (
            d_state.y,
            d_state.z,
            d_state.lam,
            d_state.xsi,
            d_state.eta,
            d_state.mu,
            d_state.zet,
            d_state.s,
        ),
        axis=0,
    )

    # Step length determination
    stepxx = -1.01 * dxx / xx
    stmxx = np.max(stepxx)
    stepalfa = -1.01 * d_state.x / (state.x - bounds.alpha)
    stmalfa = np.max(stepalfa)
    stepbeta = 1.01 * d_state.x / (bounds.beta - state.x)
    stmbeta = np.max(stepbeta)
    stmalbe = np.maximum(stmalfa, stmbeta)
    stmalbexx = np.maximum(stmalbe, stmxx)
    stminv = np.maximum(stmalbexx, 1.0)
    steg = 1.0 / stminv

    # Keep current state without addition of any Newton step.
    old = state.copy()

    # Initial residual to be improved up on.
    residunorm, _ = state.relaxed_residual(
        a0, a, b, c, d, p0, q0, P, Q, bounds, epsi
    )

    # Find largest step sizes that decreases the residual.
    # Since the direction is a descent direction, a reduction will be found.
    # It can be found for fairly small step sizes though.
    resinew = np.inf
    iteration_count = 50

    for iteration in range(iteration_count):
        if resinew <= residunorm:
            break

        # Step along the search direction.
        scaling = steg / (2**iteration)
        state = old + d_state.scale(scaling)

        # Compute relaxed optimality conditions, Section 5.2 (Equations 5.9*).
        resinew, residumax = state.relaxed_residual(
            a0, a, b, c, d, p0, q0, P, Q, bounds, epsi
        )

    return state


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
