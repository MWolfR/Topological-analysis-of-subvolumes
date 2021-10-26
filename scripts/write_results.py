"""Write results of computations."""

import os
import pandas as pd

def default_hdf(step):
    """Default HDF5 path for a pipeline step's output."""
    return (os.path.join(os.curdir, "topological_analysis.h5"), step)


def write(extracted, to_path):
    """Expecting the path to output be that to a `*.h5` archive.

    extracted : A pandas DataFrame / Series
    path : a string or a tuple of strings
    """
    try:
        path_hdf_store, group_identifier = to_path
    except TypeError:
        assert to_path.endswith(".pkl")
        extracted.to_pickle(to_path)
        return to_path

    extracted.to_hdf(path_hdf_store, key=group_identifier, mode="w")
    return (path_hdf_store, group_identifier)


def read(from_path):
    """Read an extracted dataset...
    path : a string or a tuple of strings
    """
    try:
        path_hdf_store, group_identifier = from_path
    except TypeError:
        assert from_path.endswith(".pkl")
        return pd.read__pickle(from_path)

    return pd.read_hdf(path_hdf_store, key=group_identifier)
