"""Rudimentary test functionality for MMA implementation."""

import numpy as np
import pytest
from case_beam import case_beam
from case_funct import case_funct
from case_funct_2 import case_funct_2
from case_toy import case_toy

from mma import mma


@pytest.mark.parametrize(
    "test_case",
    [
        case_beam,
        case_funct,
        case_funct_2,
        case_toy,
    ],
)
def test_mma(test_case):
    """Run through problem cases and assert expected KKT outputs are found."""
    outvector1s, outvector2s, kktnorms = mma(
        test_case.x0, test_case.func, test_case.bounds, test_case.options
    )

    msg = "Unexpected outvector 1."
    assert test_case.vec1.shape == outvector1s.shape
    assert np.allclose(test_case.vec1, np.array(outvector1s)), msg

    msg = "Unexpected outvector 2."
    assert test_case.vec2.shape == outvector2s.shape
    assert np.allclose(test_case.vec2, outvector2s), msg

    msg = "Unexpected kktnorms."
    assert test_case.kktnorms.shape == kktnorms.shape
    assert np.allclose(test_case.kktnorms, kktnorms), msg
