"""Extract subtargets' connectivity.
"""
import importlib
from argparse import ArgumentParser

from .io.write_results import (read as read_results,
                               write_toc_plus_payload as write,
                               default_hdf)
from .io import read_config as read_cfg
from .io import logging
from .connectivity import run_extraction_from_full_matrix

STEP = "extract-connectivity"
LOG = logging.get_logger(STEP)


def main(fn_cfg):
    """..."""
    LOG.warning("Run extractions using the config at %s", fn_cfg)

    cfg = read_cfg.read(fn_cfg)
    paths = cfg["paths"]
    if "circuit" not in paths:
        raise RuntimeError("No circuits defined in config!")
    if "defined_columns" not in paths:
        raise RuntimeError("No defined columns in config!")
    if "connection_matrices" not in paths:
        raise RuntimeError("No connection matrices in config!")

    cfg = cfg["parameters"].get("extract_connectivity", {})

    circuits = paths["circuit"]
    path_targets = paths["defined_columns"]

    LOG.info("Read targets from path %s", path_targets)
    if args.test:
        LOG.info("TEST pipeline plumbing")
    else:
        targets = read_results(path_targets, for_step="subtargets")
        LOG.info("Number of targets read: %s", targets.shape[0])

    connectomes = cfg.get("connectomes", [])
    LOG.info("Extract connevtivity from connectomes: %s", connectomes)
    if args.test:
        LOG.info("TEST pipeline plumbing.")
    else:
        extracted = run_extraction_from_full_matrix(circuits, targets, connectomes)
        LOG.info("Done, extraction of %s connectivity matrices.", extracted.shape)

    output = paths.get("connection_matrices", default_hdf("connection_matrices"))
    LOG.info("Write extracted matrices to %s\n\t group %s", output[0], output[1])
    if args.test:
        LOG.info("TEST pipeline plumbing.")
    else:
        output = write(extracted, to_path=output, format="table")
        LOG.info("Done, writing %s connectivity matrices.", extracted)

    LOG.warning("DONE, extraction of matrices")
    return output


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract connectivity for sub-targets.")

    parser.add_argument("config",
                        help="Path to the configuration to run the pipeline.")

    parser.add_argument("-o", "--output",
                        help="Path to the directory to output in.", default=None)

    parser.add_argument("--dry-run", dest="test",  action="store_true"
                        help=("Use this to test the pipeline's plumbing "
                              "before running any juices through it."))
    parser.set_default(test=False)

    args = parser.parse_args()

    LOG.warning("Run %s: %s", STEP, args)

    main(args)
