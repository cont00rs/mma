from dataclasses import dataclass


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
