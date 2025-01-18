"""Test case container definition."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from mma import Bounds, Options


@dataclass
class Case:
    """Problem definition and reference solutions."""

    func: Callable
    x0: np.ndarray
    bounds: Bounds
    options: Options
    kktnorms: np.ndarray
    vec1: np.ndarray
    vec2: np.ndarray
