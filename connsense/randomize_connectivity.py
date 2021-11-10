"""Randomize subtarget connectivity."""

from argparse import ArgumentParser

from randomization import Algorithm

from .io.write_results import (read as read_results,
                               read_toc_plus_payload,
                               write_toc_plus_payload,
                               default_hdf)
from .randomization import get_neuron_properties, randomize_table_of_contents

from .io import read_config
from .io import logging
from .connectivity import run_extraction_from_full_matrix

STEP = "randomize-connectivity"
LOG = logging.get_logger(STEP)


def main(args):
    """..."""
    config = read_config.read(args.config)
    paths = config["paths"]


    if "circuit" not in paths:
        raise RuntimeError("No circuits defined in config!")
    if "defined_columns" not in paths:
        raise RuntimeError("No defined columns in config!")
    if "neurons" not in paths:
        raise RuntimeError("No neurons in config!")
    if "connection_matrices" not in paths:
        raise RuntimeError("No connection matrices in config!")
    if "randomized_matrices" not in paths:
        raise RuntimeError("No randomized matrices in config!")

    hdf_path, hdf_group = paths["randomize-conenctivity"]
    LOG.info("Load extracted neuron properties from %s\n\t, group %s",
             hdf_path, hdf_group)
    if args.test:
        LOG.info("TEST pipeline plumbing")
    else:
        neurons = get_neuron_properties(hdf_path, hdf_group)
        LOG.info("Done loading extracted neuron properties: %s", neurons.shape)

    hdf_path, hdf_group = paths["extract-connectivity"]
    LOG.info("Load extracted connectivity from %s\n\t, group %s",
             hdf_path, hdf_group)
    if args.test:
        LOG.info("TEST pipeline plumbing")
    else:
        toc = read_toc_plus_payload((hdf_path, hdf_group), STEP).rename("matrix")
        LOG.info("Done reading %s table of contents for connectivity matrices",
                 toc.shape)
        if args.sample:
            S = np.float(args.sample)
            toc = toc.sample(frac=S) if S < 1 else toc.sample(n=int(S))
        parameters = config["parameters"].get("randomize_connectivity", {})
        algorithms = {Algorithm.from_config(description)
                      for _, description in parameters["algorithms"].items()}

    LOG.info("DISPATCH randomization of connecivity matrices.")
    if args.test:
        LOG.info("TEST pipeline plumbing.")
    else:
        randomized = randomize_table_of_contents(toc, neurons, algorithms,
                                                 args.batch_size)
        LOG.info("Done, randomizing %s matrices.", randomized.shape)

    hdf_path, hdf_key = paths.get(STEP, default_hdf(STEP))
    if args.output:
        hdf_path = args.output

    output = (hdf_path, hdf_key)
    LOG.info("Write %s randomized matrices to path %s.", randomized.shape, output)
    if args.test:
        LOG.info("TEST pipeline plumbing")
    else:
        output = write_toc_plus_payload(randomized, to_path=output, format="table",)
        LOG.info("Done writing %s randomized matrices.", randomized.shape)

    LOG.warning("DONE randomizing: %s", args)
    return output


if __name__ == "__main__":
    parser = ArgumentParser(description="Randomize connectivity.")

    parser.add_argument("config",
                        help="Path to the configuration to run the pipeline.")

    parser.add_argument("-s", "--sample",
                        help="A float to sample with, to be used for testing",
                        default=None)

    parser.add_argument("-b", "--batch-size",
                        help="Number of targets to process in a single thread.",
                        default=None)

    parser.add_argument("-o", "--output",
                        help="Path to the directory to output to override the config.",
                        default=None)

    parser.add_argument("--dry-run", dest="test",  action="store_true"
                        help=("Use this to test the pipeline's plumbing "
                              "before running any juices through it."))
    parser.set_default(test=False)

    args = parser.parse_args()

    LOG.warning("Launch %s: %s", STEP, str(args))
    main(args)
