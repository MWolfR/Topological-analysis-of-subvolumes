"""Extract neuron properties for each of the subtargets generated
by `define-subtargets`
"""

import os
import importlib
import pandas as pd
from bluepy import Circuit

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
