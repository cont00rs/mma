"""Toy problem test case."""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
from matplotlib.widgets import Button


from mma import Bounds, Options, OptimizationResult, mma, target_function

"""

Testing functionality of the solver with a simple parabola
gives rather strange results. If the minimum is contained
on a specific point, e.g. a line constraint cuts off the
actual minimum of the parabola, the solution converges fast
and consistently: it moves onto the constraint and is done.

If this line constraint is then shifted away, such that the
minimum of the constraint is actually the minimum of the
parabola, the solver has quite some difficulties to converge
onto this point. It keeps oscillating around 0.0, but will
not properly truncate the step sizes to terminate onto the
minimum. Tweaking various parameters seems to help, e.g. by
dropping `asymin` or `raa0` the problem reaches much lower
values and stops continuous oscillations.

To analyse this better, it would be nice to create debug
functionality that is able to draw the target function,
the current evaluation point, and the constructed asymptotes
for that position.

"""


def parabola(x):
    f0val = (x[0][0]) ** 2
    df0dx = np.array([[2 * x[0][0]]])
    fval = np.array([[-0.5 - x[0][0]]])
    dfdx = np.array([[-1]]).T
    return f0val, df0dx, fval, dfdx


class Plotter:
    def __init__(self):
        self.iter = 0
        self.sol_xs = []
        self.sol_fs = []

    def callback(self, res: OptimizationResult):
        bounds = res.mma_bounds.bounds

        _, ax = plt.subplots()

        xs = np.linspace(bounds.lb, bounds.ub, num=50)
        fs = [parabola(np.atleast_2d(x))[0] for x in xs]
        gs = [parabola(np.atleast_2d(x))[2][0] < 0 for x in xs]

        ax.plot(xs, fs, label="f")
        ax.plot(xs, gs, label="g")

        marker_size = mpl.rcParams["lines.markersize"] ** 2 / 2

        ymin, ymax = min(fs), max(fs)
        ax.vlines(
            res.mma_bounds.alpha.item(),
            ymin,
            ymax,
            label="alpha",
            colors="green",
            linestyles="dashed",
        )
        ax.vlines(
            res.mma_bounds.beta.item(),
            ymin,
            ymax,
            label="beta",
            colors="red",
            linestyles="dashed",
        )

        ax.scatter(
            res.target_function.x,
            res.target_function.f0,
            s=marker_size,
            color="magenta",
            label="x",
            zorder=10,
            marker="*",
        )

        if res.target_function.xold1:
            ax.scatter(
                res.target_function.xold1,
                res.target_function.func(res.target_function.xold1)[0].item(),
                s=marker_size / 2,
                color="red",
                label="xold1",
                zorder=10,
            )

        if self.sol_xs:
            ax.scatter(
                self.sol_xs,
                self.sol_fs,
                color="black",
                s=marker_size / 2,
            )

        self.sol_xs += [res.target_function.x]
        self.sol_fs += [res.target_function.f0]

        ax.legend()

        plt.title(f"Iteration: {self.iter}.")

        plt.draw()
        print("Press a button to continue...")
        plt.waitforbuttonpress(0)
        plt.close()

        self.iter += 1


def debug():
    bounds = Bounds(-1.0, 1.0)
    plotter = Plotter()

    result = mma(
        np.array([[+1.0]]),
        parabola,
        bounds,
        Options(
            iteration_count=50,
            move_limit=0.2,
            asymin=0.001,
            # raa0=0.01,
        ),
        callback=plotter.callback,
    )

    assert np.allclose(result.state.x, 1.0)


if __name__ == "__main__":
    debug()
