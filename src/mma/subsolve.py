import numpy as np
from scipy.linalg import solve
from scipy.sparse import diags

from mma.bounds import MMABounds


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

    def relaxed_residual(
        self, a0, a, b, c, d, p0, q0, P, Q, bounds: MMABounds, epsi
    ):
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
