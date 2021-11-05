from collections.abc import Mapping
import os
from pathlib import Path
import json



def adjust_hdf_paths(dict_paths, root):
    """..."""
    try:
        hdf_keys = dict_paths["keys"]
    except KeyError as keys:
        missing = ValueError("Missing entry for keys needed to read / write a HDF store.\n"
                             "Please provide a Mapping pipeline-step --> HDF key.")
        raise missing from keys

    if not hdf_keys:
        raise ValueError("Empty keys needed to read / write a HDF store.")

    return {step: (root, hdf_key) for step, hdf_key in hdf_keys.items()}


def mark_path_results(dict_paths):
    """Mark a path to results"""
    try:
        specified = dict_paths["root"]
    except KeyError:
        root = (Path(__file__).parent.parent.parent / "results"
                / "topological_sampling.h5")
    else:
        specified = Path(specified)
        root = (specified if specified.is_absolute()
                else Path(__file__).parent.parent.parent / specified)

    root = dict_paths.get("root", '.')
    if not os.path.isabse(root):
        root = os.path.abspath()


def adjust_root(in_a_dict_paths):
    """..."""
    if not in_a_dict_paths:
        return {}

    try:
        root = in_a_dict_paths["root"]
    except KeyError:
        root = None
    else:
        root = Path(root)

    def absolute(filepath):
        if os.path.isabs(filepath):
            return filepath
        return (root / filepath).as_posix()

    try:
        files_dict = in_a_dict_paths["files"]
    except KeyError:
        return {}
    else:
        files = files_dict.items()

    def nest(filename):
        """..."""
        try:
            subfiles = filename.items
        except AttributeError:
            return absolute(filename)
        return {label: absolute(name) for label, name in subfiles}

    return {label: nest(specified) for label, specified in files}


def read(fn, raw=False):
    with open(fn, "r") as fid:
        cfg = json.load(fid)
    assert "paths" in cfg,\
        "Configuration file must specify 'paths' to input/output files!"
    assert "parameters" in cfg,\
        "Configuration file must specify 'parameters' for pipeline steps!"
    if raw:
        return cfg

    steps = cfg["paths"]["steps"]
    hdf5_root = steps["root"]
    hdf5_groups = steps["groups"].items()

    paths = {}
    paths["circuit"] = adjust_root(cfg["paths"]["circuit"])
    paths["flatmap"] = adjust_root(cfg["paths"].get("flatmap", None))
    for step, group in hdf5_groups:
        paths[step] = (hdf5_root, group)

    parameters = cfg["parameters"]

    return {"paths": paths, "parameters": parameters}
