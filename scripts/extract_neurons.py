import os
import importlib
import pandas as pd
import bluepy
import logging

from write_results import read as read_results, write, default_hdf

read_cfg = importlib.import_module("read_config")


LOG = logging.getLogger("Generate flatmap subtargets")
LOG.setLevel(os.environ.get("LOGLEVEL", "INFO"))


def run_extractions(circuits, subtargets, params):
    if len(params) == 0:
        print("Warning: No properties to extract given. This step will do nothing!")

    circuits = dict([(k, bluepy.Circuit(v)) for k, v in circuits.items()])

    #circuit_frame = subtargets.index.to_frame().apply(lambda x: circuits[x["circuit"]], axis=1)

    def get_props(index, gids):
        circuit = circuits[index[0]]
        props = circuit.cells.get(gids, properties=params)
        props.index = pd.MultiIndex.from_tuples([index + (gid,) for gid in gids],
                                                names=["circuit", "subtarget",
                                                       "flat_x", "flat_y", "gid"])
        return props

    return pd.concat([get_props(index, gids)
                      for index, gids in subtargets.iteritems()])


def main(fn_cfg):
    cfg = read_cfg.read(fn_cfg)
    paths = cfg["paths"]
    if "circuit" not in paths:
        raise RuntimeError("No circuits defined in config!")
    if "defined_columns" not in paths:
        raise RuntimeError("No defined columns in config!")
    if "neurons" not in paths:
        raise RuntimeError("No neurons in config!")

    circuits = cfg["paths"]["circuit"]

    path_targets = cfg["paths"]["defined_columns"]

    LOG.warning("Read targets from path %s", path_targets)
    targets = read_results(path_targets, for_step="subtargets")
    LOG.warning("Number of targets read: %s", targets.shape[0])

    cfg = cfg["parameters"].get("extract_neurons", {})
    params = cfg.get("properties", [])
    extracted = run_extractions(circuits, targets, params)
    write(extracted, to_path=paths.get("neurons", default_hdf("neurons")), format="table")


if __name__ == "__main__":
    import sys
    print(main(sys.argv[1]))
