from dataclasses import dataclass

import numpy as np


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
        epsimin: A small positive number to ensure numerical stability of the subsolver.
        subsolver_iteration_count: Maximum number of iterations of the subsolver.
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
    epsimin: float = 1e-7
    subsolver_iteration_count: int = 200


@dataclass
class Coefficients:
    # a0 (float): Constant in the term a_0 * z.
    a0: float
    # a (np.ndarray): Coefficients for the term a_i * z.
    a: np.ndarray
    # c (np.ndarray): Coefficients for the term c_i * y_i.
    c: np.ndarray
    # d (np.ndarray): Coefficients for the term 0.5 * d_i * (y_i)^2.
    d: np.ndarray

    @classmethod
    def from_defaults(cls, m: int):
        """
        A collection of coeffcients within the problem formulation.

        This implementations assumes `a_i = 0` and `d_i = 1`
        for all i to match the basic problem formulation as
        defined in equation (1.2) in mmagcmma.pdf.
        """
        a0 = 1
        a = np.zeros((m, 1))
        c = 1000 * np.ones((m, 1))
        d = np.ones((m, 1))
        return Coefficients(a0, a, c, d)
