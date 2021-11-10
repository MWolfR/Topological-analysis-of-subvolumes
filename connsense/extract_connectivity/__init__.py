"""Extract subtargets' connectivity.
"""
from argparse import ArgumentParser

from ..io.write_results import (read as read_results,
                                write_toc_plus_payload as write,
                                default_hdf)
from ..io import logging
from .extract import run_extraction_from_full_matrix

STEP = "extract-connectivity"
LOG = logging.get_logger(STEP)


def run(config, *args, dry_run=False, **kwargs):
    """..."""
    paths = config["paths"]
    if "circuit" not in paths:
        raise RuntimeError("No circuits defined in config!")
    if "define-subtargets" not in paths:
        raise RuntimeError("No defined columns in config!")
    if STEP not in paths:
        raise RuntimeError("No connection matrices in config!")

    config = config["parameters"].get("extract_connectivity", {})

    circuits = paths["circuit"]
    path_targets = paths["define-subtargets"]

    LOG.info("Read targets from path %s", path_targets)
    if dry_run:
        LOG.info("TEST pipeline plumbing")
    else:
        targets = read_results(path_targets, for_step="subtargets")
        LOG.info("Number of targets read: %s", targets.shape[0])

    connectomes = config.get("connectomes", [])
    LOG.info("Extract connevtivity from connectomes: %s", connectomes)
    if dry_run:
        LOG.info("TEST pipeline plumbing.")
    else:
        extracted = run_extraction_from_full_matrix(circuits, targets, connectomes)
        LOG.info("Done, extraction of %s connectivity matrices.", extracted.shape)

    output = paths.get(STEP, default_hdf(STEP))
    LOG.info("Write extracted matrices to %s\n\t group %s", output[0], output[1])
    if dry_run:
        LOG.info("TEST pipeline plumbing.")
    else:
        output = write(extracted, to_path=output, format="table")
        LOG.info("Done, writing %s connectivity matrices.", extracted)

    LOG.warning("DONE, extraction of matrices")
    return f"Output saved at {output}"
