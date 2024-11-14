import numpy as np
import pytest

from mma import kktcheck, mmasub


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


def minimize(x: np.ndarray, func: callable):
    xval = x.copy()

    # Count constriants.
    _, _, fval, _ = func(xval)
    m = len(fval)
    n = len(xval)

    # Initial settings
    eeen = np.ones((n, 1))
    eeem = np.ones((m, 1))
    zeron = np.zeros((n, 1))
    zerom = np.zeros((m, 1))
    xold1 = xval.copy()
    xold2 = xval.copy()
    xmin = zeron.copy()
    xmax = 5 * eeen
    low = xmin.copy()
    upp = xmax.copy()
    move = 1.0
    c = 1000 * eeem
    d = eeem.copy()
    a0 = 1
    a = zerom.copy()
    outeriter = 0
    maxoutit = 11
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

        outvector1 = np.concatenate((np.array([f0val]), fval.flatten()))
        outvector2 = xval.flatten()
        outvector1s += [outvector1]
        outvector2s += [outvector2]
        kktnorms += [kktnorm]

    return outvector1s, outvector2s, kktnorms


@pytest.mark.parametrize(
    "target_function, name, x", [(toy, "toy", np.array([[4, 3, 2]]).T)]
)
def test_mma_toy(target_function, name, x):
    outvector1s, outvector2s, kktnorms = minimize(x, target_function)
    ref_outvector1s = np.loadtxt(
        f"test/test_mma_{name}_vec1.txt", delimiter=","
    )
    ref_outvector2s = np.loadtxt(
        f"test/test_mma_{name}_vec2.txt", delimiter=","
    )
    ref_kktnorms = np.loadtxt(f"test/test_mma_{name}_kkt.txt")

    msg = "Unexpected outvector 1."
    assert np.allclose(ref_outvector1s, outvector1s), msg
    msg = "Unexpected outvector 2."
    assert np.allclose(ref_outvector2s, outvector2s), msg
    msg = "Unexpected kktnorms."
    assert np.allclose(ref_kktnorms, kktnorms), msg
