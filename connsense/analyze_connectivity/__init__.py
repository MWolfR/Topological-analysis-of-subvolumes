"""Analyze connectivity."""


from collections.abc import Mapping
from pathlib import Path
from argparse import ArgumentParser

from randomization import Algorithm


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
    is_matrix = analyzed.apply(lambda r: isinstance(r, (sparse.csc_matrix,
                                                        sparse.csr_matrix,
                                                        sparse.coo_matrix)))
    matrices = analyzed[is_matrix]

    is_dataframe = analyzed.apply(lambda r: isinstance(r, (pd.DataFrame,
                                                           pd.Series)))
    dataframes = analyzed[is_dataframe]

    hdf_path, hdf_group = to_path
    return (write_dataframe(dataframes, to_path=(hdf_path, hdf_group), format=format),
            write_toc_plus_payload(matrices, to_path(hdf_path, f"{hdf_group}/matrices")))

def subset_subtargets(original, randomized, sample):
    """..."""
    if not sample:
        return (original, randomized)

    S = np.float(sample)
    if S > 1:
        subset = original.sample(n=int(S))
    elif S > 0:
        subset = original.sample(n=S)
    else:
        raise ValueError(f"Illegal sample={sample}")

    selection = subset.index

    def get_one(algorithm, randmats):
        """..."""
        randmats = randmats.droplevel("algorithm")
        return pd.concat([randmats.droplevel("algorithm").loc[selection]],
                         keys=[algorithm], names=["algorithm"])
    return (subset,
            pd.concat([get_one(a, r) for a, r in randomized.groupby("algorithm")]))


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
        toc_orig = read_toc_plus_payload((hdf_path, hdf_group), STEP).rename("matrix")
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
        analyses = [Analysis.from_config(description)
                    for _, description in parameters[STEP].items()]

        toc_orig_dispatch, toc_rand_dispatch = subset_subtargets(toc_orig, toc_rand, sample)
        analyzed = analyze_table_of_contents(toc_orig_dispatch, toc_rand_dispatch, neurons,
                                             analyses, sample,
                                             batch_size)
        LOG.info("Done, analyzing %s matrices.", sample.shape[0])


    hdf_path, hdf_group = paths.get(STEP, default_hdf(STEP))
    if output:
        hdf_path = output

    output = (hdf_path, hdf_group)

    LOG.info("Write analyses to %s", output)
    if dry_run:
        LOG.info("TEST pipeline plumbing")
    else:
        output = write(analyzed, to_path=output, format="table")
        LOG.info("Done writing %s randomized matrices: to %s", analyzed.shape, output)


    LOG.warning("DONE analyzing: %s", config)
    return f"Result saved {output}"
