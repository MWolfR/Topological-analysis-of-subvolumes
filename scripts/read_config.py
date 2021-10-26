import os
import json


def join_if_not_absolute(root, fn):
    if os.path.isabs(fn):
        return fn
    return os.path.join(root, fn)


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


def adjust_paths(dict_paths):
    root = dict_paths.get("root", ".")
    if not os.path.isabs(root):
        root = os.path.abspath(
            os.path.join(os.path.split(__file__)[0], root)
        )
    if root.endswith(".h5"):
        return adjust_hdf_paths(dict_paths, root)
    out = {}
    for name, fn in dict_paths.get("files", {}).items():
        if isinstance(fn, dict):
            for subname, subfn in fn.items():
                out.setdefault(name, {})[subname] = join_if_not_absolute(root, subfn)
        else:
            out[name] = join_if_not_absolute(root, fn)
    return out


def read(fn):
    with open(fn, "r") as fid:
        cfg = json.load(fid)
    assert "paths" in cfg, "Configuration file must specify 'paths' to input/output files!"
    assert "parameters" in cfg, "Configuration file must specify 'parameters' for pipeline steps!"
    paths = {}
    for entry in cfg["paths"]:
        paths.update(adjust_paths(entry))
    cfg["paths"] = paths
    return cfg
