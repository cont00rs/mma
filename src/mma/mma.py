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
    low = bounds.lb.copy()
    upp = bounds.ub.copy()

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
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = (
            subproblem.mmasub(
                m,
                n,
                xval,
                bounds,
                f0val,
                df0dx,
                fval,
                dfdx,
                low,
                upp,
                a0,
                a,
                c,
                d,
            )
        )

        # Some vectors are updated:
        xval = xmma.copy()

        # Re-calculate function values, gradients
        f0val, df0dx, fval, dfdx = func(xval)

        # The residual vector of the KKT conditions is calculated
        residu, kktnorm, residumax = kktcheck(
            m,
            n,
            xmma,
            ymma,
            zmma,
            lam,
            xsi,
            eta,
            mu,
            zet,
            s,
            bounds,
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
        bounds: Bounds,
        f0val: float,
        df0dx: np.ndarray,
        fval: np.ndarray,
        dfdx: np.ndarray,
        low: np.ndarray,
        upp: np.ndarray,
        a0: float,
        a: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
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
        low, upp = self.update_asymptotes(xval, bounds, low, upp)

        # Calculation of the bounds alfa and beta.
        alfa, beta = self.calculate_alpha_beta(xval, bounds, low, upp)

        # Calculations approximating functions: P, Q.
        p0, q0 = self.approximating_functions(
            xval, df0dx, bounds, low, upp, objective=True
        )

        P, Q = self.approximating_functions(
            xval, dfdx, bounds, low, upp, objective=False
        )

        # Negative residual between approximating functions and objective
        # as described in beginning of Section 5.
        b = P @ (1 / (upp - xval)) + Q @ (1 / (xval - low)) - fval

        # Solving the subproblem using the primal-dual Newton method
        # FIXME: Move options for Newton method into dataclass.
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(
            m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d
        )

        # Store design variables of last two iterations.
        self.xold2 = None if self.xold1 is None else self.xold1.copy()
        self.xold1 = xval.copy()

        # Return values
        return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp

    def update_asymptotes(self, xval, bounds, low, upp):
        """Calculation of the asymptotes low and upp.

        This represents equations 3.11 to 3.14.
        """

        # The difference between upper and lower bounds.
        delta = bounds.delta()

        if self.xold1 is None or self.xold2 is None:
            # Equation 3.11.
            low = xval - self.options.asyinit * delta
            upp = xval + self.options.asyinit * delta
            return low, upp

        # Extract sign of variable change from previous iterates.
        signs = (xval - self.xold1) * (self.xold1 - self.xold2)

        # Assign increase/decrease factoring depending on signs. Equation 3.13.
        factor = np.ones_like(xval, dtype=float)
        factor[signs > 0] = self.options.asyincr
        factor[signs < 0] = self.options.asydecr

        # Equation 3.12.
        low = xval - factor * (self.xold1 - low)
        upp = xval + factor * (upp - self.xold1)

        # Limit asymptote change to maximum increase/decrease. Equation 3.14.
        np.clip(
            low,
            a_min=xval - self.options.asymax * delta,
            a_max=xval - self.options.asymin * delta,
            out=low,
        )

        np.clip(
            upp,
            a_min=xval + self.options.asymin * delta,
            a_max=xval + self.options.asymax * delta,
            out=upp,
        )

        return low, upp

    def calculate_alpha_beta(self, xval, bounds, low, upp):
        """Calculation of the bounds alpha and beta.

        Equations 3.6 and 3.7.
        """

        # Restrict lower bound with move limit.
        lower_bound = np.maximum(
            low + self.options.alpha_factor * (xval - low),
            xval - self.options.move_limit * bounds.delta(),
        )

        # Restrict upper bound with move limit.
        upper_bound = np.minimum(
            upp - self.options.beta_factor * (upp - xval),
            xval + self.options.move_limit * bounds.delta(),
        )

        # Restrict bounds with variable bounds.
        alpha = np.maximum(lower_bound, bounds.lb)
        beta = np.minimum(upper_bound, bounds.ub)

        return alpha, beta

    def approximating_functions(
        self, xval, dfdx, bounds, low, upp, objective=True
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
        delta_inv = 1 / np.maximum(bounds.delta(), eps_delta)

        if objective:
            df_plus = np.maximum(dfdx, 0)
            df_minus = np.maximum(-dfdx, 0)
        else:
            df_plus = np.maximum(dfdx.T, 0)
            df_minus = np.maximum(-dfdx.T, 0)

        # Equation 3.3.
        p0 = (1 + factor) * df_plus + factor * df_minus
        p0 += self.options.raa0 * delta_inv
        p0 = diags_array(((upp - xval) ** 2).squeeze(axis=1)) @ p0

        # Equation 3.4.
        q0 = factor * df_plus + (1 + factor) * df_minus
        q0 += self.options.raa0 * delta_inv
        q0 = diags_array(((xval - low) ** 2).squeeze(axis=1)) @ q0

        if objective:
            return p0, q0
        else:
            return p0.T, q0.T


def subsolv(
    m: int,
    n: int,
    low: np.ndarray,
    upp: np.ndarray,
    alfa: np.ndarray,
    beta: np.ndarray,
    p0: np.ndarray,
    q0: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    a0: float,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
]:
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
        Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
            - xmma (np.ndarray): Optimal values of the variables x_j.
            - ymma (np.ndarray): Optimal values of the variables y_i.
            - zmma (float): Optimal value of the variable z.
            - slack (np.ndarray): Slack variables for the general MMA constraints.
            - lagrange (np.ndarray): Lagrange multipliers for the constraints.
    """

    # Initial problem state as given in Section 5.5 beginning.
    een = np.ones((n, 1))
    eem = np.ones((m, 1))
    epsi = 1
    x = 0.5 * (alfa + beta)
    y = eem.copy()
    z = np.array([[1.0]])
    lam = eem.copy()
    xsi = een / (x - alfa)
    xsi = np.maximum(xsi, een)
    eta = een / (beta - x)
    eta = np.maximum(eta, een)
    mu = np.maximum(eem, 0.5 * c)
    zet = np.array([[1.0]])
    s = eem.copy()

    # A small positive number to ensure numerical stability.
    epsimin = 1e-7

    # Start while loop for numerical stability
    while epsi > epsimin:
        # Compute relaxed optimality conditions, Section 5.2 (Equations 5.9*).
        plam = p0 + np.dot(P.T, lam)
        qlam = q0 + np.dot(Q.T, lam)
        gvec = P @ (1 / (upp - x)) + Q @ (1 / (x - low))
        dpsidx = plam / (upp - x) ** 2 - qlam / (x - low) ** 2
        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - a.T @ lam
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsi
        reeta = eta * (beta - x) - epsi
        remu = mu * y - epsi
        rezet = zet * z - epsi
        res = lam * s - epsi

        # Form residual vector, i.e. the right-hand-side at top Section 5.3.
        residu1 = np.concatenate((rex, rey, rez), axis=0)
        residu2 = np.concatenate(
            (relam, rexsi, reeta, remu, rezet, res), axis=0
        )
        residu = np.concatenate((residu1, residu2), axis=0)
        residunorm = np.sqrt(np.dot(residu.T, residu).item())
        residumax = np.max(np.abs(residu))

        # Start inner while loop for optimization
        for ittt in range(200):
            if residumax <= 0.9 * epsi:
                break

            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            uxinv1 = een / ux1
            xlinv1 = een / xl1
            uxinv2 = een / ux2
            xlinv2 = een / xl2
            plam = p0 + np.dot(P.T, lam)
            qlam = q0 + np.dot(Q.T, lam)
            gvec = np.dot(P, uxinv1) + np.dot(Q, xlinv1)
            GG = (diags(uxinv2.flatten(), 0).dot(P.T)).T - (
                diags(xlinv2.flatten(), 0).dot(Q.T)
            ).T
            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsi / (x - alfa) + epsi / (beta - x)
            dely = c + d * y - lam - epsi / y
            delz = a0 - np.dot(a.T, lam) - epsi / z
            dellam = gvec - a * z - y - b + epsi / lam
            diagx = plam / ux3 + qlam / xl3
            diagx = 2 * diagx + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = een / diagx
            diagy = d + mu / y
            diagyinv = eem / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv

            # Solve system of equations
            # The size of design variables and constraint functions play
            # a role in determining how to solve the primal-dual problem.
            # The paper expands on this at the end of Section 5.3.
            # TODO: Consider to move more of that discussion here.

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
                AAr2 = np.concatenate((a, -zet / z), axis=0).T
                AA = np.concatenate((AAr1, AAr2), axis=0)
                solut = solve(AA, bb)
                dlam = solut[0:m]

                dz = solut[m : m + 1]
                dx = -delx / diagx - np.dot(GG.T, dlam) / diagx
            else:
                # Delta lambda is eliminated (Equation 5.21) and the system
                # of equations in delta x, delta z is constructed. This
                # represents Equation 5.22.
                diaglamyiinv = eem / diaglamyi
                dellamyi = dellam + dely / diagy
                Axx = np.asarray(
                    diags(diagx.flatten(), 0)
                    + (diags(diaglamyiinv.flatten(), 0).dot(GG).T).dot(GG)
                )
                azz = zet / z + np.dot(a.T, (a / diaglamyi))
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
            dxsi = -xsi + epsi / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsi / (beta - x) + (eta * dx) / (beta - x)
            dmu = -mu + epsi / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsi / lam - (s * dlam) / lam
            xx = np.concatenate((y, z, lam, xsi, eta, mu, zet, s), axis=0)
            dxx = np.concatenate(
                (dy, dz, dlam, dxsi, deta, dmu, dzet, ds), axis=0
            )

            x, y, z, lam, xsi, eta, mu, zet, s, residumax = line_search(
                # newton solution
                xx,
                dxx,
                # bounds
                low,
                upp,
                alfa,
                beta,
                # approximations
                p0,
                q0,
                P,
                Q,
                # current state
                x,
                y,
                z,
                lam,
                xsi,
                eta,
                mu,
                zet,
                s,
                # derivative
                dx,
                dy,
                dz,
                dlam,
                dxsi,
                deta,
                dmu,
                dzet,
                ds,
                # parameters
                a0,
                a,
                b,
                c,
                d,
                epsi,
                residunorm,
                n,
                m,
            )

        epsi = 0.1 * epsi

    xmma = x.copy()
    ymma = y.copy()
    zmma = z.copy()
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu
    zetmma = zet
    smma = s

    return xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma


def line_search(
    # newton solution
    xx,
    dxx,
    # bounds
    low,
    upp,
    alfa,
    beta,
    # approximations
    p0,
    q0,
    P,
    Q,
    # current state
    x,
    y,
    z,
    lam,
    xsi,
    eta,
    mu,
    zet,
    s,
    # derivative
    dx,
    dy,
    dz,
    dlam,
    dxsi,
    deta,
    dmu,
    dzet,
    ds,
    # parameters
    a0,
    a,
    b,
    c,
    d,
    epsi,
    residunorm,
    n,
    m,
):
    """Line search along Newton descent direction.


    Line search into the Newton direction, Section 5.4.
    This aims to find a step into the Newton direction without
    violating any of the constraints, which might happen when taking
    the full Newton step. The specifications of a "good" step are
    indicated in Section 5.4.
    """
    een = np.ones((n, 1))
    eem = np.ones((m, 1))

    # Step length determination
    stepxx = -1.01 * dxx / xx
    stmxx = np.max(stepxx)
    stepalfa = -1.01 * dx / (x - alfa)
    stmalfa = np.max(stepalfa)
    stepbeta = 1.01 * dx / (beta - x)
    stmbeta = np.max(stepbeta)
    stmalbe = np.maximum(stmalfa, stmbeta)
    stmalbexx = np.maximum(stmalbe, stmxx)
    stminv = np.maximum(stmalbexx, 1.0)
    steg = 1.0 / stminv

    # Update variables
    xold = x.copy()
    yold = y.copy()
    zold = z.copy()
    lamold = lam.copy()
    xsiold = xsi.copy()
    etaold = eta.copy()
    muold = mu.copy()
    zetold = zet.copy()
    sold = s.copy()

    itto = 0
    resinew = 2 * residunorm

    for itto in range(50):
        if resinew <= residunorm:
            break

        x = xold + steg * dx
        y = yold + steg * dy
        z = zold + steg * dz
        lam = lamold + steg * dlam
        xsi = xsiold + steg * dxsi
        eta = etaold + steg * deta
        mu = muold + steg * dmu
        zet = zetold + steg * dzet
        s = sold + steg * ds
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = een / ux1
        xlinv1 = een / xl1
        plam = p0 + np.dot(P.T, lam)
        qlam = q0 + np.dot(Q.T, lam)
        gvec = np.dot(P, uxinv1) + np.dot(Q, xlinv1)
        dpsidx = plam / ux2 - qlam / xl2
        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - np.dot(a.T, lam)
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsi
        reeta = eta * (beta - x) - epsi
        remu = mu * y - epsi
        rezet = zet * z - epsi
        res = lam * s - epsi
        residu1 = np.concatenate((rex, rey, rez), axis=0)
        residu2 = np.concatenate(
            (relam, rexsi, reeta, remu, rezet, res), axis=0
        )
        residu = np.concatenate((residu1, residu2), axis=0)
        resinew = np.sqrt(np.dot(residu.T, residu))
        steg = steg / 2

    residumax = np.max(np.abs(residu))
    steg = 2 * steg

    return x, y, z, lam, xsi, eta, mu, zet, s, residumax


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
    rexsi = xsi * (x - bounds.lb)
    reeta = eta * (bounds.ub - x)
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
