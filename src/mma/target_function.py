import numpy as np


class TargetFunction:
    def __init__(self, func, xval: np.ndarray):
        self.func = func
        self.f0, self.df0dx, self.f, self.dfdx = func(xval)

    def evaluate(self, xval):
        self.f0, self.df0dx, self.f, self.dfdx = self.func(xval)
