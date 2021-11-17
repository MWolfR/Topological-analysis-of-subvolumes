"""Analyze connectivity."""


from collections.abc import Mapping
from pathlib import Path
from argparse import ArgumentParser


import pandas as pd
import numpy as np
from scipy import sparse

from ..io.write_results import (read as read_results,
                                read_toc_plus_payload,
                                write as write_dataframe,
                                write_toc_plus_payload,
                                default_hdf)

from ..io import read_config
from ..io import logging

from ..randomize_connectivity.randomize import get_neuron_properties
from .analysis import SingleMethodAnalysisFromSource
from .analyze import analyze_table_of_contents

STEP = "analyze-connectivity"
LOG = logging.get_logger(STEP)


def read(config):
    """..."""
    try:
        path = Path(config)
    except TypeError:
        assert isinstance(config, Mapping)
        return config
    return  read_config.read(path)

def write(analyzed, to_path, format="table"):
    """..."""
    hdf_path, hdf_group = to_path
    for i, g in analyzed.groupby("analysis"):
        LOG.info("Write analysis %s to %s/%s", i, hdf_group, i)
        write_dataframe(g, to_path=(hdf_path, f"{hdf_group}/{i}"),
                        format=format)
    return hdf_path

def subset_subtargets(original, randomized, sample):
    """..."""
    if not sample:
        return (original, randomized)

    all_matrices = pd.concat([original, randomized]).rename('matrix')

    if not sample:
        return all_matrices

    S = np.float(sample)
    if S > 1:
        subset = all_matrices.sample(n=int(S))
    elif S > 0:
        subset = all_matrices.sample(frac=S)
    else:
        raise ValueError(f"Illegal sample={sample}")

    return subset

def get_analyses(config):
    """..."""
    all_parameters = config["parameters"]

    analyze_params = all_parameters[STEP]

    configured = analyze_params["analyses"]

    LOG.warning("configured analyses %s", configured )

    return [SingleMethodAnalysisFromSource(name, description)
            for name, description in configured.items()]


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
    if "randomize-connectivity" not in paths:
        raise RuntimeError("No randomized matrices in config paths: {list(paths.keys()}!")
    if STEP not in paths:
        raise RuntimeError(f"No {STEP} in config!")

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
        toc_orig = pd.concat([read_toc_plus_payload((hdf_path, hdf_group),
                                                    STEP).rename("matrix")],
                              keys=["original"], names=["algorithm"])
        LOG.info("Done loading  %s table of contents of original connectivity matrices",
                 toc_orig.shape)

    hdf_path, hdf_group = paths["randomize-connectivity"]
    LOG.info("Load randomized connectivity from %s\n\t, group %s",
             hdf_path, hdf_group)
    if dry_run:
        LOG.info("Test pipeline plumbing")
    else:
        toc_rand = read_toc_plus_payload((hdf_path, hdf_group), STEP).rename("matrix")
        LOG.info("Done loading  %s table of contents of randomized connevitiy matrices",
                 toc_rand.shape)

    LOG.info("DISPATCH analyses of connectivity.")
    if dry_run:
        LOG.info("TEST pipeline plumbing")
    else:
        parameters = config["parameters"].get("analyze_connectivity", {})
        analyses = get_analyses(config)

        toc_dispatch = subset_subtargets(toc_orig, toc_rand, sample)
        analyzed = analyze_table_of_contents(toc_dispatch, neurons, analyses,
                                             batch_size)
        LOG.info("Done, analyzing %s matrices.", analyzed.shape[0])


    hdf_path, hdf_group = paths.get(STEP, default_hdf(STEP))
    if output:
        hdf_path = output

    output = (hdf_path, hdf_group)

    LOG.info("Write analyses to %s", output)
    if dry_run:
        LOG.info("TEST pipeline plumbing")
    else:
        for i, g in analyzed.groupby("analysis"):
            LOG.info("Write analysis %s to %s/%s", i, hdf_group, i)
            write_dataframe(g, to_path=(hdf_path, f"{hdf_group}/{i}"),
                            format="table")
        output = hdf_path
        #output = write(analyzed, to_path=output, format="table")
        LOG.info("Done writing %s analyzed matrices: to %s", analyzed.shape, output)

    LOG.warning("DONE analyzing: %s", config)
    return f"Result saved {output}"
