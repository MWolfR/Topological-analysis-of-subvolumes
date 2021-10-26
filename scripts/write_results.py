"""Write results of computations."""

import os
import pandas as pd

def default_hdf(step):
    """Default HDF5 path for a pipeline step's output."""
    return (os.path.join(os.curdir, "topological_analysis.h5"), step)


def write(extracted, to_path, format=None):
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

    extracted.to_hdf(path_hdf_store, key=group_identifier,
                     mode="a", format=(format or "fixed"))
    return (path_hdf_store, group_identifier)


def read(path, for_step):
    """Read dataset extracted for a pipeline step from path to the dataset.

    path : a string or a tuple of strings
    for_step : string that names a pipeline step.
    """
    try:
        path_hdf_store, group_identifier = path
    except TypeError:
        assert path.endswith(".pkl")
        if not os.path.isfile(path):
            raise RuntimeError(f"Missing pickled data for step {for_step} at {path}.\n"
                               f"Run {for_step} with config that sets outputs to pickels first.")
        return pd.read_pickle(path)

    if not os.path.isfile(path_hdf_store):
        raise RuntimeError(f"Missing HDF data for step {for_step} at path {path_hdf_store}\n"
                           f"Run {for_step} step with config that sets outputs to HDF first.")

    return pd.read_hdf(path_hdf_store, key=group_identifier)
