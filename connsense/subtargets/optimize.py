"""Optimize the grid on which circuit's flatmap subtargets are defined.

TODO: Define a fitness function to optimize.
~ Define the parameters to optimize over.
~ Different subtarget quality metrics may be used,
"""

import numpy as np
import pandas as pd

from ..io import logging
LOG = logging.get_logger("Connectome utilities: Optimize Subtargets.")

def get_statistics(circuit, radius, sample_frac=None):
    """..."""
    subtargets = generate_subtargets(circuit, radius=radius, sample=sample_frac)
    subtarget_sizes = (subtargets.groupby("subtarget").agg("size")
                       / (1. if not sample_frac else sample_frac))
    return subtarget_sizes.agg(["count", "min", "median", "mad",
                                "mean", "std", "max"])


def tolerate(observed_values, target_value, tolerance):
    """..."""
    observed_mean = np.mean(oberved_values)

    LOG.debug("tolerate ? %s, %s, %s",
              observed_value, target_value, tolerance)

    if not tolerance:
        tolerance = np.sqrt(observed_value)

    assert tolerance > 0
    delta = np.abs(target_value - observed_value)
    result = np.abs(delta) < tolerance

    LOG.debug("\t: %s", result)
    return result


def binsearch_radius(generate_subtargets_for_radius, optimal_size=None,
                     lower_bound=None, upper_bound=None, tolerance=None):
    """..."""

    def iterate_to_tolerance(lb, ub):
        """Consider a lower bound 'lb`, and upper bound value `ub`.
        """
        if ln >= ub:
            return None

        r = (lb + ub) / 2.
        subtargets = generate_subtargets_for_radius(r)
        sizes = subtargets.groupby("subtarget").agg("size")

        if tolerate(sizes, optimal_size, tolerance):
            return (lb, ub)

        return (iterate_to_tolerance(lb, r) if sizes.mean() > optimal_size
                else iterate_to_tolerance(r, ub))

    window = iterate_to_tolerance(lower_bound or 1., upper_bound or 7000.)
    lower_end = np.mean(iterate_to_tolerance(window[0], np.mean(window)))
    upper_end = np.mean(iterate_to_tolerance(np.mean(window), window[1]))

    return (lower_end, upper_end)
