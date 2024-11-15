import pathlib

import numpy as np
import pytest
from scipy.linalg import solve

from mma import kktcheck, mmasub


def truss2(
    xval: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    This script solves the "three bar truss problem", formulated as follows:

    Minimize:
        max{f1(x), f2(x), f3(x)}

    Subject to:
        x1 + x2 + x3 <= V
        0.001 <= xj <= V, for j = 1, 2, 3

    Where:
        - xj: Volume of the j-th bar
        - fi(x): Compliance for the i-th load case, calculated as pi' * ui
        - V: Upper bound on the total volume (V = 3)

    Problem Description:
        - Bar 1 connects nodes 1 and 4.
        - Bar 2 connects nodes 2 and 4.
        - Bar 3 connects nodes 3 and 4.
        - Nodes coordinates:
            Node 1: (-1, 0)
            Node 2: (-1/sqrt(2), -1/sqrt(2))
            Node 3: (0, -1)
            Node 4: (0, 0)
        - Nodes 1, 2, 3 are fixed.
        - Load vectors at node 4:
            Load vector p1 = (1, 0)'
            Load vector p2 = (1, 1)'
            Load vector p3 = (0, 1)'

    Displacement vector ui is obtained from the system
        K(x) * ui = pi,
    where K(x) is the stiffness matrix and pi is the load vector.
    The stiffness matrix is given by
        K(x) = R * D(x) * R',
    where D(x) is a diagonal matrix with diagonal elements x1, x2, x3.
    The derivatives of the functions fi(x) are given by:
        dfi/dxj = -(rj' * ui)^2,
    where rj is the j-th column of the matrix R.

    MMA Formulation:

    Minimize:
        z + 1000 * (y1 + y2 + y3 + y4)

    Subject to:
        f1(x) - z - y1 <= 0
        f2(x) - z - y2 <= 0
        f3(x) - z - y3 <= 0
        x1 + x2 + x3 - 3 - y4 <= 0
        0 <= xj <= 3, for j = 1, 2, 3
        yi >= 0, for i = 1, 2, 3, 4
        z >= 0
    """

    e = np.array([[1, 1, 1]]).T
    f0val = 0
    df0dx = 0 * e
    D = np.diag(xval.flatten())
    sq2 = 1.0 / np.sqrt(2.0)
    R = np.array([[1, sq2, 0], [0, sq2, 1]])
    p1 = np.array([[1, 0]]).T
    p2 = np.array([[1, 1]]).T
    p3 = np.array([[0, 1]]).T
    K = np.dot(R, D).dot(R.T)
    u1 = solve(K, p1)
    u2 = solve(K, p2)
    u3 = solve(K, p3)
    compl1 = np.dot(p1.T, u1)
    compl2 = np.dot(p2.T, u2)
    compl3 = np.dot(p3.T, u3)
    volume = np.dot(e.T, xval)
    V = 3.0
    vol1 = volume - V
    fval = np.concatenate((compl1, compl2, compl3, vol1))
    rtu1 = np.dot(R.T, u1)
    rtu2 = np.dot(R.T, u2)
    rtu3 = np.dot(R.T, u3)
    dcompl1 = -rtu1 * rtu1
    dcompl2 = -rtu2 * rtu2
    dcompl3 = -rtu3 * rtu3
    dfdx = np.concatenate((dcompl1.T, dcompl2.T, dcompl3.T, e.T))
    return f0val, df0dx, fval, dfdx


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


def minimize(
    x: np.ndarray,
    func: callable,
    lower_bound,
    upper_bound,
    maxoutit,
    move,
    d=None,
    a=None,
):
    # Count constriants.
    _, _, fval, _ = func(x)
    m = 1 if isinstance(fval, float) else len(fval)
    n = len(x)

    # Initialisation.
    xval = x.copy()
    xold1 = xval.copy()
    xold2 = xval.copy()

    # Lower, upper bounds
    xmin = lower_bound * np.ones((n, 1))
    xmax = upper_bound * np.ones((n, 1))
    low = xmin.copy()
    upp = xmax.copy()

    c = 1000 * np.ones((m, 1))

    if d is None:
        d = np.ones((m, 1))

    a0 = 1

    if a is None:
        a = np.zeros((m, 1))

    outeriter = 0
    kkttol = 0

    # Test output
    outvector1s = []
    outvector2s = []
    kktnorms = []

    # The iterations start
    kktnorm = kkttol + 10
    outit = 0

    while kktnorm > kkttol and outit < maxoutit:
        outit += 1
        outeriter += 1

        f0val, df0dx, fval, dfdx = func(xval)

        # The MMA subproblem is solved at the point xval:
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
            m,
            n,
            outeriter,
            xval,
            xmin,
            xmax,
            xold1,
            xold2,
            f0val,
            df0dx,
            fval,
            dfdx,
            low,
            upp,
            a0,
            a,
            c,
            d,
            move,
        )

        # Some vectors are updated:
        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()

        # Re-calculate function values, gradients
        f0val, df0dx, fval, dfdx = func(xval)

        # The residual vector of the KKT conditions is calculated
        residu, kktnorm, residumax = kktcheck(
            m,
            n,
            xmma,
            ymma,
            zmma,
            lam,
            xsi,
            eta,
            mu,
            zet,
            s,
            xmin,
            xmax,
            df0dx,
            fval,
            dfdx,
            a0,
            a,
            c,
            d,
        )

        # TODO: Align sizes and shapes between test problems.
        if isinstance(fval, float):
            outvector1 = np.array([f0val, fval])
        else:
            outvector1 = np.concatenate((np.array([f0val]), fval.flatten()))

        outvector2 = xval.flatten()
        outvector1s += [outvector1]
        outvector2s += [outvector2]
        kktnorms += [kktnorm]

    return np.array(outvector1s), np.array(outvector2s), np.array(kktnorms)


@pytest.mark.parametrize(
    "target_function, name, x, lower_bound, upper_bound, maxoutit, move, a, d",
    [
        (toy, "toy", np.array([[4, 3, 2]]).T, 0, 5, 11, 1, None, None),
        (beam, "beam", 5 * np.ones((5, 1)), 1, 10, 11, 1, None, None),
        (funct, "funct", np.ones((1, 1)), 1, 100, 20, 1, None, None),
        (funct2, "funct2", np.ones((2, 1)), 1, 100, 20, 0.2, None, None),
        (
            truss2,
            "truss",
            np.ones((3, 1)),
            0.001,
            3,
            6,
            1.0,
            np.array([[1, 1, 1, 0]]).T,
            np.zeros((4, 1)),
        ),
    ],
    ids=["toy", "beam", "funct", "funct2", "truss"],
)
def test_mma_toy(
    target_function, name, x, lower_bound, upper_bound, maxoutit, move, a, d
):
    outvector1s, outvector2s, kktnorms = minimize(
        x, target_function, lower_bound, upper_bound, maxoutit, move, a=a, d=d
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
