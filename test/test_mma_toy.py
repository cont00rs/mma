import numpy as np

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


def minimize_toy():
    # Initial settings
    m, n = 2, 3
    eeen = np.ones((n, 1))
    eeem = np.ones((m, 1))
    zeron = np.zeros((n, 1))
    zerom = np.zeros((m, 1))
    xval = np.array([[4, 3, 2]]).T
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

    # Calculate function values and gradients
    if outeriter == 0:
        f0val, df0dx, fval, dfdx = toy(xval)
        outvector1 = np.concatenate((np.array([f0val]), fval.flatten()))
        outvector2 = xval.flatten()

        outvector1s += [outvector1]
        outvector2s += [outvector2]

    # The iterations start
    kktnorm = kkttol + 10
    outit = 0

    while kktnorm > kkttol and outit < maxoutit:
        outit += 1
        outeriter += 1

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
        f0val, df0dx, fval, dfdx = toy(xval)

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


def test_mma_toy():
    outvector1s, outvector2s, kktnorms = minimize_toy()

    ref_outvector1s = np.array(
        [
            [29, -6, -6],
            [9.9599287, -2.15166048, 0.21519475],
            [8.80303134, -0.11433839, 0.02320731],
            [
                8.77032876e00,
                -1.98047257e-04,
                1.64412280e-05,
            ],
            [
                8.77024960e00,
                7.88739252e-07,
                -2.46903902e-06,
            ],
            [
                8.77024658e00,
                8.20107875e-08,
                -6.17503177e-07,
            ],
            [
                8.77024613e00,
                -1.80068287e-07,
                -1.85263591e-07,
            ],
            [
                8.77024610e00,
                -2.31151580e-07,
                -1.33492204e-07,
            ],
            [
                8.77024610e00,
                -2.34501032e-07,
                -1.31555208e-07,
            ],
            [
                8.77024610e00,
                -2.34509775e-07,
                -1.31552747e-07,
            ],
            [
                8.77024610e00,
                -2.34510143e-07,
                -1.31552799e-07,
            ],
            [
                8.77024610e00,
                -2.34510360e-07,
                -1.31552859e-07,
            ],
        ]
    )

    ref_outvector2s = np.array(
        [
            [4, 3, 2],
            [2.39029817, 1.8057194, 0.99286496],
            [2.03845206, 1.76235892, 1.24170671],
            [2.01779333, 1.77855703, 1.23918268],
            [2.01762573, 1.77936919, 1.23825735],
            [2.01755448, 1.7797962, 1.23775845],
            [2.01752587, 1.77996796, 1.2375579],
            [2.01751893, 1.78000961, 1.2375093],
            [2.01751858, 1.78001172, 1.23750683],
            [2.01751864, 1.78001134, 1.23750728],
            [2.01751863, 1.78001144, 1.23750716],
            [2.01751862, 1.78001145, 1.23750715],
        ]
    )

    ref_kktnorms = np.array(
        [
            4.307115201941678,
            0.6221210867839285,
            0.02221729585724501,
            0.004514848168456325,
            0.0015626630207070617,
            0.00035034849134041524,
            4.046232679602292e-05,
            2.494070630787751e-06,
            8.220819329845235e-07,
            3.844920078255069e-07,
            4.3359496454658805e-07,
        ]
    )

    msg = "Unexpected outvector 1."
    assert np.allclose(ref_outvector1s, outvector1s), msg
    msg = "Unexpected outvector 2."
    assert np.allclose(ref_outvector2s, outvector2s), msg
    msg = "Unexpected kktnorms."
    assert np.allclose(ref_kktnorms, kktnorms), msg
