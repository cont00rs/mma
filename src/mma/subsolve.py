import numpy as np
from scipy.linalg import solve
from scipy.sparse import diags_array

from mma.approximations import Approximations
from mma.bounds import MMABounds
from mma.options import Coefficients, Options
from mma.target_function import TargetFunction


class State:
    """State representation for the Newton problem.

    Section 5.3.
        - xmma (np.ndarray): Optimal values of the variables x_j.
        - ymma (np.ndarray): Optimal values of the variables y_i.
        - zmma (float): Optimal value of the variable z.
        - lam (np.ndarray): Lagrange multipliers for the constraints.
        - xsi (np.ndarray):
            Lagrange multipliers for the lower bounds on design variables.
        - eta (np.ndarray):
            Lagrange multipliers for the upper bounds on design variables.
        - mu (np.ndarray):
            Lagrange multipliers for the slack variables of the constraints.
        - zet (float): Lagrange multiplier for the regularization term z.
        - s (np.ndarray): Slack variables for the general constraints.
    """

    def __init__(self, state: np.ndarray, offsets: dict[str, tuple[int, int]]):
        self.state = state
        self.offsets = offsets

    @classmethod
    def from_alpha_beta(cls, m: int, bounds: MMABounds, c: np.ndarray):
        x = (bounds.alpha + bounds.beta) / 2
        y = np.ones((m, 1))
        z = np.array([[1.0]])
        lam = np.ones((m, 1))
        xsi = np.maximum(1 / (x - bounds.alpha), 1)
        eta = np.maximum(1 / (bounds.beta - x), 1)
        mu = np.maximum(np.ones((m, 1)), 0.5 * c)
        zet = np.array([[1.0]])
        s = np.ones((m, 1))
        return State.from_variables(x, y, z, lam, xsi, eta, mu, zet, s)

    @classmethod
    def from_variables(cls, x, y, z, lam, xsi, eta, mu, zet, s):
        attributes = [
            "x",
            "y",
            "z",
            "lam",
            "xsi",
            "eta",
            "mu",
            "zet",
            "s",
        ]

        arguments = [
            x,
            y,
            z,
            lam,
            xsi,
            eta,
            mu,
            zet,
            s,
        ]

        offset = 0
        offsets = dict()
        for attr, arg in zip(attributes, arguments):
            offsets[attr] = (offset, offset + len(arg))
            offset += len(arg)

        return State(np.concatenate(arguments, axis=0), offsets)

    # TODO: Can these properties be generated?

    @property
    def x(self):
        start, end = self.offsets["x"]
        return self.state[start:end]

    @property
    def y(self):
        start, end = self.offsets["y"]
        return self.state[start:end]

    @property
    def z(self):
        start, end = self.offsets["z"]
        return self.state[start:end]

    @property
    def lam(self):
        start, end = self.offsets["lam"]
        return self.state[start:end]

    @property
    def xsi(self):
        start, end = self.offsets["xsi"]
        return self.state[start:end]

    @property
    def eta(self):
        start, end = self.offsets["eta"]
        return self.state[start:end]

    @property
    def mu(self):
        start, end = self.offsets["mu"]
        return self.state[start:end]

    @property
    def zet(self):
        start, end = self.offsets["zet"]
        return self.state[start:end]

    @property
    def s(self):
        start, end = self.offsets["s"]
        return self.state[start:end]

    def copy(self):
        return State(self.state.copy(), self.offsets)

    def __add__(self, other):
        assert isinstance(other, State)
        assert len(self.state) == len(other.state)
        return State(self.state + other.state, self.offsets)

    def scale(self, value: int | float):
        """Scale all state variables."""
        self.state *= value
        return self

    def residual(self) -> tuple[float, float]:
        """Return the 2-norm of the state and maximum absolute state value."""
        return np.linalg.norm(self.state).item(), np.max(np.abs(self.state))

    def relaxed_residual(
        self,
        coeff: Coefficients,
        b: np.ndarray,
        approx: Approximations,
        bounds: MMABounds,
        epsi,
    ) -> tuple[float, float]:
        """Calculate residuals of the relaxed equations.

        The state equations are converted to their "relaxed" form,
        see Equations 5.9*, and the residuals are obtained from the
        full state vector of all equations.
        """

        plam = approx.p0 + approx.P.T @ self.lam
        qlam = approx.q0 + approx.Q.T @ self.lam
        gvec = approx.P @ (1 / (bounds.upp - self.x)) + approx.Q @ (
            1 / (self.x - bounds.low)
        )
        dpsidx = (
            plam / (bounds.upp - self.x) ** 2
            - qlam / (self.x - bounds.low) ** 2
        )

        return State.from_variables(
            dpsidx - self.xsi + self.eta,
            coeff.c + coeff.d * self.y - self.mu - self.lam,
            coeff.a0 - self.zet - coeff.a.T @ self.lam,
            gvec - coeff.a * self.z - self.y + self.s - b,
            self.xsi * (self.x - bounds.alpha) - epsi,
            self.eta * (bounds.beta - self.x) - epsi,
            self.mu * self.y - epsi,
            self.zet * self.z - epsi,
            self.lam * self.s - epsi,
        ).residual()


def subsolv(
    target_function: TargetFunction,
    bounds: MMABounds,
    approx: Approximations,
    coeff: Coefficients,
    options: Options,
) -> State:
    """
    Solve the MMA (Method of Moving Asymptotes) subproblem for optimization.

    Minimize:
        SUM[p0j/(uppj-xj) + q0j/(xj-lowj)] + a0*z + SUM[ci*yi + 0.5*di*(yi)^2]

    Subject to:
        SUM[pij/(uppj-xj) + qij/(xj-lowj)] - ai*z - yi <= bi,
        alfa_j <= xj <= beta_j, yi >= 0, z >= 0.

    Args:
        target_function (TargetFunction)
        bounds (Bounds)
        approx (Approximations)
        coeff (Coefficients)
        options (Options)

    Returns:
        State
    """

    # Initial problem state as given in Section 5.5 beginning.
    epsi = 1

    state = State.from_alpha_beta(target_function.m, bounds, coeff.c)

    # Negative residual.
    b = approx.residual(bounds, target_function)

    # Start while loop for numerical stability
    while epsi > options.epsimin:
        # Start inner while loop for optimization
        for _ in range(options.iteration_count):
            # Compute relaxed optimality conditions, Section 5.2.
            _, residual = state.relaxed_residual(
                coeff, b, approx, bounds, epsi
            )

            if residual <= 0.9 * epsi:
                break

            state = line_search(bounds, approx, state, coeff, b, options, epsi)

        epsi = 0.1 * epsi

    return state


def solve_newton_step(
    state: State,
    bounds: MMABounds,
    approx: Approximations,
    coeff: Coefficients,
    b: np.ndarray,
    epsi: float,
):
    ux = bounds.upp - state.x
    xl = state.x - bounds.low

    # Equation 5.5
    plam = approx.p0 + approx.P.T @ state.lam
    qlam = approx.q0 + approx.Q.T @ state.lam

    # Equation 5.2.
    gvec = approx.P @ np.reciprocal(ux) + approx.Q @ np.reciprocal(xl)

    # Equation 5.12.
    GG = approx.P @ diags_array(np.reciprocal(ux**2).squeeze(axis=1))
    GG -= approx.Q @ diags_array(np.reciprocal(xl**2).squeeze(axis=1))

    # Equation 5.8 (dpsi/dxj) and Equation 5.15d.
    delx = plam / ux**2
    delx -= qlam / xl**2
    delx -= epsi / (state.x - bounds.alpha)
    delx += epsi / (bounds.beta - state.x)

    # Equation 5.15e.
    dely = coeff.c + coeff.d * state.y - state.lam - epsi / state.y

    # Equation 5.15f.
    delz = coeff.a0 - coeff.a.T @ state.lam - epsi / state.z

    # Equation 5.15g.
    dellam = gvec - coeff.a * state.z - state.y - b + epsi / state.lam

    # Equation 5.15a.
    diagx = 2 * plam / ux**3
    diagx += 2 * qlam / xl**3
    diagx += state.xsi / (state.x - bounds.alpha)
    diagx += state.eta / (bounds.beta - state.x)

    # Equation 5.15b.
    diagy = coeff.d + state.mu / state.y

    # Equation 5.15c.
    diaglamy = state.s / state.lam + np.reciprocal(diagy)

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
        Alam = diags_array(diaglamy.squeeze(axis=1))
        Alam += GG @ diags_array(np.reciprocal(diagx).squeeze(axis=1)) @ GG.T
        AA = np.block([[Alam, coeff.a], [coeff.a.T, -state.zet / state.z]])

        bb = np.vstack([dellam + dely / diagy - GG @ (delx / diagx), delz])
        dlam_dz = solve(AA, bb)
        dlam, dz = dlam_dz[0:m], dlam_dz[m:]
        dx = -(delx + GG.T @ dlam) / diagx
    else:
        # Delta lambda is eliminated (Equation 5.21) and the system
        # of equations in delta x, delta z is constructed. This
        # represents Equation 5.22.
        dellamyi = dellam + dely / diagy

        Axx = diags_array(diagx.squeeze(axis=1))
        Axx += GG @ diags_array(np.reciprocal(diaglamy).squeeze(axis=1)) @ GG.T
        axz = -GG.T @ (coeff.a / diaglamy)
        azz = state.zet / state.z + coeff.a.T @ (coeff.a / diaglamy)
        AA = np.block([[Axx, axz], [axz.T, azz]])

        bx = delx + GG.T @ (dellamyi / diaglamy)
        bz = delz - coeff.a.T @ (dellamyi / diaglamy)
        bb = np.vstack([-bx, -bz])

        dx_dz = solve(AA, bb)
        dx, dz = dx_dz[0:n], dx_dz[n:]

        dlam = np.dot(GG, dx) / diaglamy
        dlam -= -dz * (coeff.a / diaglamy)
        dlam += +dellamyi / diaglamy

    # Back substitute the solutions found to reconstruct the full
    # solutions of delta's. This is the solution of a Newton step
    # as specified at the start of Section 5.3.

    # Equation 5.13a.
    dxsi = -state.xsi
    dxsi += epsi / (state.x - bounds.alpha)
    dxsi -= (state.xsi * dx) / (state.x - bounds.alpha)

    # Equation 5.13b.
    deta = -state.eta
    deta += epsi / (bounds.beta - state.x)
    deta += (state.eta * dx) / (bounds.beta - state.x)

    # Equation 5.16.
    dy = -dely / diagy + dlam / diagy

    # Equation 5.13c, 5.13d, 5.13e.
    dmu = -state.mu + epsi / state.y - (state.mu * dy) / state.y
    dzet = -state.zet + epsi / state.z - state.zet * dz / state.z
    ds = -state.s + epsi / state.lam - (state.s * dlam) / state.lam

    return State.from_variables(dx, dy, dz, dlam, dxsi, deta, dmu, dzet, ds)


def line_search(
    bounds: MMABounds,
    approx: Approximations,
    state: State,
    coeff: Coefficients,
    b: np.ndarray,
    options: Options,
    epsi,
):
    """Line search along Newton descent direction.


    Line search into the Newton direction, Section 5.4.
    This aims to find a step into the Newton direction without
    violating any of the constraints, which might happen when taking
    the full Newton step. The specifications of a "good" step are
    indicated in Section 5.4.
    """

    # Obtain the Newton direction to search along.
    d_state = solve_newton_step(state, bounds, approx, coeff, b, epsi)

    # A smudge factor used in the step length decision.
    # This is defined in Section 5.4.
    factor = 1.01

    # For the step length in x all attributes except x are used. This finds
    # the starting index of the other attributes and computes the maximum
    # step size using the remainder of the array only.
    start, _ = state.offsets["y"]
    step_max_x = -factor * np.min(d_state.state[start:] / state.state[start:])

    step_max_alpha = np.max(-factor * d_state.x / (state.x - bounds.alpha))
    step_max_beta = np.max(factor * d_state.x / (bounds.beta - state.x))
    step_max_alpha_beta = np.maximum(step_max_alpha, step_max_beta)
    step_max_x_alpha_beta = np.maximum(step_max_alpha_beta, step_max_x)
    step_max = 1.0 / np.maximum(step_max_x_alpha_beta, 1.0)

    # Keep current state without addition of any Newton step.
    old = state.copy()

    # Initial residual to be improved up on.
    residunorm, _ = state.relaxed_residual(coeff, b, approx, bounds, epsi)

    # Find largest step sizes that decreases the residual.
    # Since the direction is a descent direction, a reduction will be found.
    # It can be found for fairly small step sizes though.
    resinew = np.inf

    for iteration in range(options.line_search_iteration_count):
        # Step along the search direction.
        scaling = step_max / (2**iteration)
        state = old + d_state.scale(scaling)

        # Compute relaxed optimality conditions, Section 5.2 (Equations 5.9*).
        resinew, _ = state.relaxed_residual(coeff, b, approx, bounds, epsi)

        if resinew <= residunorm:
            break

    return state
