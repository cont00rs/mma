import numpy as np
from scipy.sparse import diags_array

from mma.bounds import MMABounds
from mma.target_function import TargetFunction


class Approximations:
    """Container for the function approximations."""

    def __init__(
        self,
        xval,
        target_function: TargetFunction,
        bounds: MMABounds,
        raa0: float,
    ):
        """
        p0: Coefficients for the lower bound terms.
        q0: Coefficients for the upper bound terms.
        P: Matrix of coefficients for the lower bound terms in the constraints.
        Q: Matrix of coefficients for the upper bound terms in the constraints.
        """
        self.factor = 0.001
        self.eps_delta = 0.00001
        self.raa0 = raa0

        self.p0, self.q0 = self.approximating_functions(
            xval, target_function.df0dx, bounds, objective=True
        )

        self.P, self.Q = self.approximating_functions(
            xval, target_function.dfdx, bounds, objective=False
        )

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

        # Inverse bounds with eps to avoid divide by zero.
        # Last component of equations 3.3 and 3.4.
        delta_inv = 1 / np.maximum(bounds.bounds.delta(), self.eps_delta)

        if objective:
            df_plus = np.maximum(dfdx, 0)
            df_minus = np.maximum(-dfdx, 0)
        else:
            df_plus = np.maximum(dfdx.T, 0)
            df_minus = np.maximum(-dfdx.T, 0)

        # Equation 3.3.
        p0 = (1 + self.factor) * df_plus + self.factor * df_minus
        p0 += self.raa0 * delta_inv
        p0 = diags_array(((bounds.upp - xval) ** 2).squeeze(axis=1)) @ p0

        # Equation 3.4.
        q0 = self.factor * df_plus + (1 + self.factor) * df_minus
        q0 += self.raa0 * delta_inv
        q0 = diags_array(((xval - bounds.low) ** 2).squeeze(axis=1)) @ q0

        if objective:
            return p0, q0
        else:
            return p0.T, q0.T

    def residual(
        self, bounds: MMABounds, xval, target_function: TargetFunction
    ) -> np.ndarray:
        """Negative residual between approximating functions and objective.

        Described in beginning of Section 5.
        """
        p = self.P @ (1 / (bounds.upp - xval))
        q = self.Q @ (1 / (xval - bounds.low))
        return p + q - target_function.f
