"""Extract neuron properties for each of the subtargets generated
by `define-subtargets`
"""
import os
import importlib
import pandas as pd
from bluepy import Circuit

from write_results import read as read_results, write, default_hdf

from .io import read_config as read_cfg
from .io.logging import get_logger

STEP = "extract-neurons"
LOG = get_logger(STEP)


def run_extractions(circuits, subtargets, params):
    """Run the extractions."""
    LOG.warning("RUN neuron properties extractions")
    if len(params) == 0:
        LOG.warning("No properties to extract given. This step will do nothing!")

    circuits = dict([(k, Circuit(v)) for k, v in circuits.items()])

    #circuit_frame = subtargets.index.to_frame().apply(lambda x: circuits[x["circuit"]], axis=1)

    def get_props(index, gids):
        """Get properties for one circuit."""
        LOG.info("Get properties for %s-th circuit", index)
        circuit = circuits[index[0]]
        props = circuit.cells.get(gids, properties=params)
        props.index = pd.MultiIndex.from_tuples([index + (gid,) for gid in gids],
                                                names=["circuit", "subtarget",
                                                       "flat_x", "flat_y", "gid"])
        LOG.info("DONE %s properties for %s-th circuit", props.shape, index)
        return props

    neuron_properties = pd.concat([get_props(index, gids)
                                   for index, gids in subtargets.iteritems()])
    LOG.warning("DONE neuron properties extractions: %s", neuron_properties.shape)

    return neuron_properties


def extract_configurations(fn_cfg):
    """Launch extraction of  neurons."""
    LOG.warning("Extract neurons for subtargets.")
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

    LOG.info("READ targets from path %s", path_targets)
    targets = read_results(path_targets, for_step="subtargets")
    LOG.info("DONE read number of targets read: %s", targets.shape[0])

    cfg = cfg["parameters"].get("extract_neurons", {})
    params = cfg.get("properties", [])

    extracted = run_extractions(circuits, targets, params)

    LOG.info("WRITE neuron properties to archive.")
    write(extracted, to_path=paths.get("neurons", default_hdf("neurons")), format="table")
    LOG.info("DONE neuron properties to archive.")

    LOG.warning("DONE extract neurons for %s subtargets", targets.shape[0])


if __name__ == "__main__":
    import sys
    print(main(sys.argv[1]))
