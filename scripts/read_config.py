import os
import json


def join_if_not_absolute(root, fn):
    if os.path.isabs(fn):
        return fn
    return os.path.join(root, fn)


def adjust_paths(dict_paths):
    root = dict_paths.get("root", ".")
    if not os.path.isabs(root):
        root = os.path.abspath(
            os.path.join(os.path.split(__file__)[0], root)
        )
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
    paths = {}
    for entry in cfg["paths"]:
        paths.update(adjust_paths(entry))
    cfg["paths"] = paths
    return cfg
