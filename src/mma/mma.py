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
        asyinit: Factor to calculate the initial distance of the asymptotes.
        asydecr: Factor by which the asymptotes distance is decreased.
        asyincr: Factor by which the asymptotes distance is increased.
        asymin: Factor to calculate the minimum distance of the asymptotes.
        asymax: Factor to calculate the maximum distance of the asymptotes.
        raa0: Parameter representing the function approximation's accuracy.
        albefa: Factor to calculate the bounds alfa and beta..
    """

    asyinit: float = 0.5
    asydecr: float = 0.7
    asyincr: float = 1.2
    asymin: float = 0.01
    asymax: float = 10
    raa0: float = 0.00001
    albefa: float = 0.1


def mma(
    x: np.ndarray,
    func: callable,
    bounds: Bounds,
    iteration_count: int,
    move_limit: float = 0.5,
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
    options = Options()
    subproblem = SubProblem(options)

    for _ in range(iteration_count):
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
                move_limit,
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
        msg = (
            f"MMA did not converge within iteration limit ({iteration_count})"
        )
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
        move_limit: float,
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
            move_limit (float): Move limit for the design variables.

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

        epsimin = 0.0000001
        eeen = np.ones((n, 1), dtype=float)
        eeem = np.ones((m, 1), dtype=float)
        zeron = np.zeros((n, 1), dtype=float)

        # Calculation of the asymptotes low and upp
        if self.xold1 is None or self.xold2 is None:
            low = xval - self.options.asyinit * bounds.delta()
            upp = xval + self.options.asyinit * bounds.delta()
        else:
            zzz = (xval - self.xold1) * (self.xold1 - self.xold2)
            factor = eeen.copy()
            factor[zzz > 0] = self.options.asyincr
            factor[zzz < 0] = self.options.asydecr
            low = xval - factor * (self.xold1 - low)
            upp = xval + factor * (upp - self.xold1)
            lowmin = xval - self.options.asymax * bounds.delta()
            lowmax = xval - self.options.asymin * bounds.delta()
            uppmin = xval + self.options.asymin * bounds.delta()
            uppmax = xval + self.options.asymax * bounds.delta()
            low = np.maximum(low, lowmin)
            low = np.minimum(low, lowmax)
            upp = np.minimum(upp, uppmax)
            upp = np.maximum(upp, uppmin)

        # Calculation of the bounds alfa and beta
        zzz1 = low + self.options.albefa * (xval - low)
        zzz2 = xval - move_limit * bounds.delta()
        zzz = np.maximum(zzz1, zzz2)
        alfa = np.maximum(zzz, bounds.lb)
        zzz1 = upp - self.options.albefa * (upp - xval)
        zzz2 = xval + move_limit * bounds.delta()
        zzz = np.minimum(zzz1, zzz2)
        beta = np.minimum(zzz, bounds.ub)

        # Calculations of p0, q0, P, Q and b
        xmami_eps = 0.00001 * eeen
        xmami = np.maximum(bounds.delta(), xmami_eps)
        xmami_inv = eeen / xmami
        ux1 = upp - xval
        ux2 = ux1 * ux1
        xl1 = xval - low
        xl2 = xl1 * xl1
        ux_inv = eeen / ux1
        xl_inv = eeen / xl1
        p0 = zeron.copy()
        q0 = zeron.copy()
        p0 = np.maximum(df0dx, 0)
        q0 = np.maximum(-df0dx, 0)
        pq0 = 0.001 * (p0 + q0) + self.options.raa0 * xmami_inv
        p0 = p0 + pq0
        q0 = q0 + pq0
        p0 = p0 * ux2
        q0 = q0 * xl2
        P = np.zeros((m, n), dtype=float)
        Q = np.zeros((m, n), dtype=float)
        P = np.maximum(dfdx, 0)
        Q = np.maximum(-dfdx, 0)
        PQ = 0.001 * (P + Q) + self.options.raa0 * np.dot(eeem, xmami_inv.T)
        P = P + PQ
        Q = Q + PQ
        P = (diags(ux2.flatten(), 0).dot(P.T)).T
        Q = (diags(xl2.flatten(), 0).dot(Q.T)).T
        b = np.dot(P, ux_inv) + np.dot(Q, xl_inv) - fval

        # Solving the subproblem using the primal-dual Newton method
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(
            m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d
        )

        # Store design variables of last two iterations.
        self.xold2 = None if self.xold1 is None else self.xold1.copy()
        self.xold1 = xval.copy()

        # Return values
        return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp


def subsolv(
    m: int,
    n: int,
    epsimin: float,
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
        epsimin (float): A small positive number to ensure numerical stability.
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

    een = np.ones((n, 1))
    eem = np.ones((m, 1))
    epsi = 1
    epsvecn = epsi * een
    epsvecm = epsi * eem
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
    itera = 0

    # Start while loop for numerical stability
    while epsi > epsimin:
        epsvecn = epsi * een
        epsvecm = epsi * eem
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
        rexsi = xsi * (x - alfa) - epsvecn
        reeta = eta * (beta - x) - epsvecn
        remu = mu * y - epsvecm
        rezet = zet * z - epsi
        res = lam * s - epsvecm
        residu1 = np.concatenate((rex, rey, rez), axis=0)
        residu2 = np.concatenate(
            (relam, rexsi, reeta, remu, rezet, res), axis=0
        )
        residu = np.concatenate((residu1, residu2), axis=0)
        residunorm = np.sqrt(np.dot(residu.T, residu).item())
        residumax = np.max(np.abs(residu))
        ittt = 0

        # Start inner while loop for optimization
        while (residumax > 0.9 * epsi) and (ittt < 200):
            ittt += 1
            itera += 1
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
            delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x)
            dely = c + d * y - lam - epsvecm / y
            delz = a0 - np.dot(a.T, lam) - epsi / z
            dellam = gvec - a * z - y - b + epsvecm / lam
            diagx = plam / ux3 + qlam / xl3
            diagx = 2 * diagx + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = een / diagx
            diagy = d + mu / y
            diagyinv = eem / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv

            # Solve system of equations
            if m < n:
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

            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsvecn / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsvecn / (beta - x) + (eta * dx) / (beta - x)
            dmu = -mu + epsvecm / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsvecm / lam - (s * dlam) / lam
            xx = np.concatenate((y, z, lam, xsi, eta, mu, zet, s), axis=0)
            dxx = np.concatenate(
                (dy, dz, dlam, dxsi, deta, dmu, dzet, ds), axis=0
            )

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

            while (resinew > residunorm) and (itto < 50):
                itto += 1
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
                rexsi = xsi * (x - alfa) - epsvecn
                reeta = eta * (beta - x) - epsvecn
                remu = mu * y - epsvecm
                rezet = zet * z - epsi
                res = lam * s - epsvecm
                residu1 = np.concatenate((rex, rey, rez), axis=0)
                residu2 = np.concatenate(
                    (relam, rexsi, reeta, remu, rezet, res), axis=0
                )
                residu = np.concatenate((residu1, residu2), axis=0)
                resinew = np.sqrt(np.dot(residu.T, residu))
                steg = steg / 2
            residunorm = resinew.copy()
            residumax = np.max(np.abs(residu))
            steg = 2 * steg

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
