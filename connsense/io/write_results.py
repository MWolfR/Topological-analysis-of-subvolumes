"""Write results of computations."""

import os
from pathlib import Path

import io

import h5py
import numpy
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


def write_sparse_matrix_payload(hdf_group, dset_pattern="matrix_{0}"):
    def write_dset(mat):
        dset_name = dset_pattern.format(len(hdf_group.keys()))
        from scipy import sparse
        bio = io.BytesIO()
        sparse.save_npz(bio, mat)
        bio.seek(0)
        matrix_bytes = list(bio.read())
        hdf_group.create_dataset(dset_name, data=matrix_bytes)
        return hdf_group.name + "/" + dset_name
    return write_dset


def read_sparse_matrix_payload(hdf_dset):
    from scipy import sparse
    raw_data = bytes(hdf_dset[:].astype(numpy.uint8))
    bio = io.BytesIO(raw_data)
    mat = sparse.load_npz(bio)
    return mat


def write_toc_plus_payload(extracted, to_path, format=None):
    path_hdf_store, group_identifier = to_path
    group_identifier_toc = group_identifier + "/toc"
    group_identifier_mat = group_identifier + "/payload"

    h5_file = h5py.File(path_hdf_store, "a")
    h5_grp_mat = h5_file.require_group(group_identifier_mat)
    toc = extracted.apply(write_sparse_matrix_payload(h5_grp_mat))
    h5_file.close()

    write(toc, (path_hdf_store, group_identifier_toc), format=format)


class LazyMatrix:
    """..."""
    from lazy import lazy
    def __init__(self, path_hdf, path_dset):
        """..."""
        self._hdf = path_hdf
        self._dset = path_dset

    @lazy
    def matrix(self):
        """..."""
        with h5py.File(self._hdf, 'r') as hdf:
            dset = hdf[self._dset]
            matrix = read_sparse_matrix_payload(dset)
        return matrix


def read_toc_plus_payload(path, for_step):
    path_hdf_store, group_identifier = path
    group_identifier_toc = group_identifier + "/toc"

    if not os.path.isfile(path_hdf_store):
        raise RuntimeError(f"Missing HDF data for step {for_step} at path {path_hdf_store}\n"
                           f"Run {for_step} step with config that sets outputs to HDF first.")

    toc = pd.read_hdf(path_hdf_store, key=group_identifier_toc)

    return toc.apply(lambda dset_path: LazyMatrix(path_hdf_store, dset_path))


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


def read_subtargets(from_object):
    """..."""
    try:
        path = Path(from_object)
    except TypeError:
        root, group = path
    else:
        root = path; group = "subtargets"

    return read((root, group), for_step="define-subtargets")


def read_node_properties(path):
    """
    TODO: allow flat_x, flat_y in index.
    """
    try:
        path = Path(from_object)
    except TypeError:
        root, group = path
    else:
        root = path; group = "neurons"

    return (read((root, group), "extract-neurons")
            .droplevel(["flat_x", "flat_y"])
            .reset_index()
            .set_index(["circuit", "subtarget"]))


def resolve_output_between(argued, and_configured):
    """
    If a valid value is passed in arguments from CLI, use that one instead
    of the value provided in a configuration.

    Only the folder is configurable from CLI --
    HDF keys are fixed to follow those set in a config.
    """
    if not argued:
        return and_configured

    try:
        _, __ = argued
    except TypeError:
        pass
    else:
        return argued

    try:
        _, key_hdf = and_configured
    except TypeError:
        return argued

    return (argued, key_hdf)
