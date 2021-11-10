"""Randomize subtarget connectivity."""

from collections.abc import Mapping
from pathlib import Path
from argparse import ArgumentParser

from randomization import Algorithm


from ..io.write_results import (read as read_results,
                               read_toc_plus_payload,
                               write_toc_plus_payload,
                               default_hdf)

from ..io import read_config
from ..io import logging

from .randomize import get_neuron_properties, randomize_table_of_contents

STEP = "randomize-connectivity"
LOG = logging.get_logger(STEP)


def read(config):
    """..."""
    try:
        path = Path(config)
    except TypeError:
        assert isinstance(config, Mapping)
        return config
    return  read_config.read(path)


def run(config, *args, output=None, batch_size=None, sample=None,  dry_run=None,
        **kwargs):
    """..."""
    config = read(config)
    paths = config["paths"]

    if "circuit" not in paths:
        raise RuntimeError("No circuits defined in config!")
    if "define-subtargets" not in paths:
        raise RuntimeError("No defined columns in config!")
    if "extract-neurons" not in paths:
        raise RuntimeError("No neurons in config!")
    if "extract-connectivity" not in paths:
        raise RuntimeError("No connection matrices in config!")
    if STEP not in paths:
        raise RuntimeError("No randomized matrices in config!")

    hdf_path, hdf_group = paths["extract-neurons"]
    LOG.info("Load extracted neuron properties from %s\n\t, group %s",
             hdf_path, hdf_group)
    if dry_run:
        LOG.info("TEST pipeline plumbing")
    else:
        neurons = get_neuron_properties(hdf_path, hdf_group)
        LOG.info("Done loading extracted neuron properties: %s", neurons.shape)

    hdf_path, hdf_group = paths["extract-connectivity"]
    LOG.info("Load extracted connectivity from %s\n\t, group %s",
             hdf_path, hdf_group)
    if dry_run:
        LOG.info("TEST pipeline plumbing")
    else:
        toc = read_toc_plus_payload((hdf_path, hdf_group), STEP).rename("matrix")
        LOG.info("Done reading %s table of contents for connectivity matrices",
                 toc.shape)
        if sample:
            S = np.float(sample)
            toc = toc.sample(frac=S) if S < 1 else toc.sample(n=int(S))
        parameters = config["parameters"].get("randomize_connectivity", {})
        algorithms = {Algorithm.from_config(description)
                      for _, description in parameters["algorithms"].items()}

    LOG.info("DISPATCH randomization of connecivity matrices.")
    if dry_run:
        LOG.info("TEST pipeline plumbing.")
    else:
        randomized = randomize_table_of_contents(toc, neurons, algorithms,
                                                 batch_size)
        LOG.info("Done, randomizing %s matrices.", randomized.shape)

    hdf_path, hdf_key = paths.get(STEP, default_hdf(STEP))
    if output:
        hdf_path = output

    output = (hdf_path, hdf_key)
    LOG.info("Write randomized matrices to path %s.",  output)
    if dry_run:
        LOG.info("TEST pipeline plumbing")
    else:
        output = write_toc_plus_payload(randomized, to_path=output, format="table",)
        LOG.info("Done writing %s randomized matrices: to %s", randomized.shape, output)

    LOG.warning("DONE randomizing: %s", config)
    return f"Result saved {output}"
