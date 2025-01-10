import pathlib

import numpy as np
import pytest

from mma import Bounds, mma


def funct(
    xval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple function with one design variable and bounds:

    Minimize:
        (x - 50)^2 + 25

    Subject to:
        1 <= x <= 100
    """
    f0val = (xval - 50) ** 2 + 25
    df0dx = 2 * (xval - 50)
    fval = np.zeros_like(xval)
    dfdx = np.zeros_like(xval)
    return f0val, df0dx, fval, dfdx


def funct2(
    xval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple function with two variables and bounds:

    Minimize:
        (x1 - 50)^2 + (x2 - 25)^2 + 25

    Subject to:
        1 <= x(j) <= 100, for j = 1, 2
    """
    f0val = np.array([(xval[0][0] - 50) ** 2 + (xval[1][0] - 25) ** 2 + 25])
    df0dx = np.array([2 * (xval[0] - 50), 2 * (xval[1] - 25)])
    fval = np.zeros((1,))
    dfdx = np.zeros_like(xval.T)
    return f0val, df0dx, fval, dfdx


def toy(
    xval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A toy problem defined as:

    Minimize:
         x(1)^2 + x(2)^2 + x(3)^2

    Subject to:
        (x(1)-5)^2 + (x(2)-2)^2 + (x(3)-1)^2 <= 9
        (x(1)-3)^2 + (x(2)-4)^2 + (x(3)-3)^2 <= 9
        0 <= x(j) <= 5, for j=1,2,3.
    """
    f0val = np.sum(xval**2, keepdims=True)
    df0dx = 2 * xval
    fval1 = np.sum((xval.T - np.array([[5, 2, 1]])) ** 2) - 9
    fval2 = np.sum((xval.T - np.array([[3, 4, 3]])) ** 2) - 9
    fval = np.array([[fval1, fval2]]).T
    dfdx1 = 2 * (xval.T - np.array([[5, 2, 1]]))
    dfdx2 = 2 * (xval.T - np.array([[3, 4, 3]]))
    dfdx = np.concatenate((dfdx1, dfdx2))
    return f0val, df0dx, fval, dfdx


def beam(
    xval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The beam problem from the MMA paper of Svanberg.

    Minimize:
        c1 * (x(1) + x(2) + x(3) + x(4) + x(5))
        c1 = 0.0624

    Subject to:
        a1/x(1)^3 + a2/x(2)^3 + a3/x(3)^3 + a4/x(4)^3 + a5/x(5)^3 <= 1
        a1 = 61, a2 = 37, a3 = 19, a4 = 7, a5 = 1
        1 <= x(j) <= 10, for j = 1, ..., 5.
    """
    c1 = 0.0624
    ai = np.array([[61.0, 37.0, 19.0, 7.0, 1.0]]).T
    f0val = c1 * np.sum(xval, keepdims=True)
    df0dx = c1 * np.ones_like(xval)
    fval = np.sum(ai / xval**3, keepdims=True) - 1
    dfdx = -3 * (ai / xval**4).T
    return f0val, df0dx, fval, dfdx


@pytest.mark.parametrize(
    "target_function, name, x, bounds, maxoutit, move",
    [
        (toy, "toy", np.array([[4, 3, 2]]).T, Bounds(0, 5), 11, 1),
        (beam, "beam", 5 * np.ones((5, 1)), Bounds(1, 10), 11, 1),
        (funct, "funct", np.ones((1, 1)), Bounds(1, 100), 20, 1),
        (funct2, "funct2", np.ones((2, 1)), Bounds(1, 100), 20, 0.2),
    ],
    ids=["toy", "beam", "funct", "funct2"],
)
def test_mma_toy(target_function, name, x, bounds, maxoutit, move):
    outvector1s, outvector2s, kktnorms = mma(
        x, target_function, bounds, maxoutit, move
    )

    reference_dir = pathlib.Path("test/reference")

    ref_outvector1s = np.loadtxt(
        reference_dir / f"test_mma_{name}_vec1.txt", delimiter=","
    )

    ref_outvector2s = np.loadtxt(
        reference_dir / f"test_mma_{name}_vec2.txt", delimiter=","
    )

    # FIXME: Align reference shapes with mma output
    if ref_outvector2s.ndim == 1:
        ref_outvector2s = np.atleast_2d(ref_outvector2s).T

    ref_kktnorms = np.loadtxt(reference_dir / f"test_mma_{name}_kkt.txt")

    msg = "Unexpected outvector 1."
    assert ref_outvector1s.shape == outvector1s.shape
    assert np.allclose(ref_outvector1s, np.array(outvector1s)), msg
    msg = "Unexpected outvector 2."
    assert ref_outvector2s.shape == outvector2s.shape
    assert np.allclose(ref_outvector2s, outvector2s), msg
    msg = "Unexpected kktnorms."
    assert ref_kktnorms.shape == kktnorms.shape
    assert np.allclose(ref_kktnorms, kktnorms), msg
