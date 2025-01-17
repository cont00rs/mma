import numpy as np
from scipy.sparse import issparse

from mma.options import Options
from mma.target_function import TargetFunction


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


class MMABounds:
    def __init__(self, bounds: Bounds, options: Options):
        self.bounds = bounds
        self.options = options

        # Initialise (low, upp) to the bounds of the problem.
        self.low = bounds.lower()
        self.upp = bounds.upper()

    def update_asymptotes(self, target_function: TargetFunction):
        """Calculation of the asymptotes low and upp.

        This represents equations 3.11 to 3.14.
        """

        # The difference between upper and lower bounds.
        delta = self.bounds.delta()

        if target_function.xold1 is None or target_function.xold2 is None:
            # Equation 3.11.
            self.low = target_function.x - self.options.asyinit * delta
            self.upp = target_function.x + self.options.asyinit * delta
            return

        # Extract sign of change between iterates.
        signs = (target_function.x - target_function.xold1) * (
            target_function.xold1 - target_function.xold2
        )

        # Assign increase/decrease factoring depending on signs. Equation 3.13.
        factor = np.ones_like(target_function.x, dtype=float)
        factor[signs > 0] = self.options.asyincr
        factor[signs < 0] = self.options.asydecr

        # Equation 3.12.
        self.low = target_function.x - factor * (
            target_function.xold1 - self.low
        )
        self.upp = target_function.x + factor * (
            self.upp - target_function.xold1
        )

        # Limit asymptote change to maximum increase/decrease. Equation 3.14.
        np.clip(
            self.low,
            a_min=target_function.x - self.options.asymax * delta,
            a_max=target_function.x - self.options.asymin * delta,
            out=self.low,
        )

        np.clip(
            self.upp,
            a_min=target_function.x + self.options.asymin * delta,
            a_max=target_function.x + self.options.asymax * delta,
            out=self.upp,
        )

    def calculate_alpha_beta(self, x: np.ndarray):
        """Calculation of the bounds alpha and beta.

        Equations 3.6 and 3.7.
        """

        # Restrict lower bound with move limit.
        lower_bound = np.maximum(
            self.low + self.options.alpha_factor * (x - self.low),
            x - self.options.move_limit * self.bounds.delta(),
        )

        # Restrict upper bound with move limit.
        upper_bound = np.minimum(
            self.upp - self.options.beta_factor * (self.upp - x),
            x + self.options.move_limit * self.bounds.delta(),
        )

        # Restrict bounds with variable bounds.
        self.alpha = np.maximum(lower_bound, self.bounds.lb)
        self.beta = np.minimum(upper_bound, self.bounds.ub)
