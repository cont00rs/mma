"""Toy problem test case."""

import numpy as np
from case import Case

from mma import mma, Bounds, Options


def parabola(x):
    f0val = (x[0][0]) ** 2
    df0dx = np.array([[2 * x[0][0]]])
    fval = np.array([[-100 - x[0][0]]])
    dfdx = np.array([[-1]]).T
    return f0val, df0dx, fval, dfdx


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


def test_parabola():
    result = mma(
        np.array([[+1.0]]),
        parabola,
        Bounds(-1.0, 1.0),
        Options(
            iteration_count=100,
            move_limit=0.2,
            asymin=0.0001,
            # raa0=0.01,
        ),
    )
    assert np.allclose(result.state.x, 1.0)
