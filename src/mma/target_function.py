"""Wrapper class for objective and constraint functions."""

from typing import Optional

import numpy as np


class TargetFunction:
    """Wrapper class containing the function and point of evaluation."""

    def __init__(self, func, x: np.ndarray):
        self.func = func
        self.x = x
        self.f0, self.df0dx, self.f, self.dfdx = func(x)

        # Store the problem sizes.
        self.n = len(x)
        self.m = 1 if isinstance(self.f, float) else len(self.f)

        # Track state of last two evaluations.
        self.xold1: Optional[np.ndarray] = None
        self.xold2: Optional[np.ndarray] = None

    def evaluate(self, x):
        """Evaluate the objective and constraints at x."""
        self.x = x
        self.f0, self.df0dx, self.f, self.dfdx = self.func(x)

    def store(self):
        """Persist current state as a previous iterate state."""
        self.xold2 = None if self.xold1 is None else self.xold1.copy()
        self.xold1 = self.x.copy()
