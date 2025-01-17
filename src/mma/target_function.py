import numpy as np


class TargetFunction:
    def __init__(self, func, x: np.ndarray):
        self.func = func
        self.x = x
        self.f0, self.df0dx, self.f, self.dfdx = func(x)

    def evaluate(self, x):
        self.x = x
        self.f0, self.df0dx, self.f, self.dfdx = self.func(x)
