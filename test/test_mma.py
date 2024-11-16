import pathlib

import numpy as np
import pytest

from mma import mma


def funct(xval: np.ndarray) -> tuple[float, np.ndarray, float, np.ndarray]:
    """Simple function with one design variable and no constraints:

    Minimize:
        (x - 50)^2 + 25

    Subject to:
        1 <= x <= 100
    """
    eeen = np.ones((len(xval), 1))
    zeron = np.zeros((len(xval), 1))
    f0val = (xval.item() - 50) ** 2 + 25
    df0dx = eeen * (2 * (xval.item() - 50))
    fval = 0.0
    dfdx = zeron
    return f0val, df0dx, fval, dfdx


def funct2(xval: np.ndarray) -> tuple[float, np.ndarray, float, np.ndarray]:
    """Simple function with two variables and one constraint:

    Minimize:
        (x1 - 50)^2 + (x2 - 25)^2 + 25

    Subject to:
        1 <= x(j) <= 100, for j = 1, 2
    """
    zeron = np.zeros((len(xval), 1))
    f0val = (xval[0][0] - 50) ** 2 + (xval[1][0] - 25) ** 2 + 25
    df0dx1 = 2 * (xval[0] - 50)
    df0dx2 = 2 * (xval[1] - 25)
    df0dx = zeron
    df0dx[0] = df0dx1
    df0dx[1] = df0dx2
    fval = 0.0
    dfdx = zeron.T
    return f0val, df0dx, fval, dfdx


def toy(xval: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """A toy problem defined as:

    Minimize:
         x(1)^2 + x(2)^2 + x(3)^2

    Subject to:
        (x(1)-5)^2 + (x(2)-2)^2 + (x(3)-1)^2 <= 9
        (x(1)-3)^2 + (x(2)-4)^2 + (x(3)-3)^2 <= 9
        0 <= x(j) <= 5, for j=1,2,3.
    """
    f0val = xval[0][0] ** 2 + xval[1][0] ** 2 + xval[2][0] ** 2
    df0dx = 2 * xval
    fval1 = ((xval.T - np.array([[5, 2, 1]])) ** 2).sum() - 9
    fval2 = ((xval.T - np.array([[3, 4, 3]])) ** 2).sum() - 9
    fval = np.array([[fval1, fval2]]).T
    dfdx1 = 2 * (xval.T - np.array([[5, 2, 1]]))
    dfdx2 = 2 * (xval.T - np.array([[3, 4, 3]]))
    dfdx = np.concatenate((dfdx1, dfdx2))
    return f0val, df0dx, fval, dfdx


def beam(xval: np.ndarray) -> tuple[float, np.ndarray, float, np.ndarray]:
    """The beam problem from the MMA paper of Svanberg.

    Minimize:
        0.0624*(x(1) + x(2) + x(3) + x(4) + x(5))

    Subject to:
        61/(x(1)^3) + 37/(x(2)^3) + 19/(x(3)^3) + 7/(x(4)^3) + 1/(x(5)^3) <= 1
        1 <= x(j) <= 10, for j = 1, ..., 5.
    """
    nx = 5
    eeen = np.ones((nx, 1))
    c1 = 0.0624
    c2 = 1
    aaa = np.array([[61.0, 37.0, 19.0, 7.0, 1.0]]).T
    xval2 = xval * xval
    xval3 = xval2 * xval
    xval4 = xval2 * xval2
    xinv3 = eeen / xval3
    xinv4 = eeen / xval4
    f0val = c1 * np.dot(eeen.T, xval).item()
    df0dx = c1 * eeen
    fval = (np.dot(aaa.T, xinv3) - c2).item()
    dfdx = -3 * (aaa * xinv4).T
    return f0val, df0dx, fval, dfdx


@pytest.mark.parametrize(
    "target_function, name, x, lower_bound, upper_bound, maxoutit, move, a, d",
    [
        (toy, "toy", np.array([[4, 3, 2]]).T, 0, 5, 11, 1, None, None),
        (beam, "beam", 5 * np.ones((5, 1)), 1, 10, 11, 1, None, None),
        (funct, "funct", np.ones((1, 1)), 1, 100, 20, 1, None, None),
        (funct2, "funct2", np.ones((2, 1)), 1, 100, 20, 0.2, None, None),
    ],
    ids=["toy", "beam", "funct", "funct2"],
)
def test_mma_toy(
    target_function, name, x, lower_bound, upper_bound, maxoutit, move, a, d
):
    outvector1s, outvector2s, kktnorms = mma(
        x, target_function, lower_bound, upper_bound, maxoutit, move, d=d
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
