"""Define flatmap column subtargets for a circuit.
"""
import os
from argparse import ArgumentParser
import logging

from .io import read_config
from .io.write_results import write, default_hdf
from .io.logging import get_logger
from .subtargets.config import SubtargetsConfig
from .subtargets import define as define_subtargets

STEP = "define-subtargets"
LOG = get_logger(STEP)


def main(args):
    """Interpret `args` and launch."""

    LOG.warning("Get subtargets for %s", args)
    LOG.info("Load the config %s", args.config)

    if args.sample:
        LOG.info("Sample a fraction %s of the cells", args.sample)
        sample = float(args.sample)
    else:
        sample = None

    config = SubtargetsConfig(args.config)

    output = config.output
    if args.output:
        try:
            _, hdf_key = output
        except TypeError:
            output = args.output
        else:
            output = (args.output, hdf_key)
    LOG.info("Output in %s\n\t, group %s", output[0], output[1])

    LOG.info("DISPATCH the definition of subtargets.")
    if args.test:
        LOG.info("TEST pipeline plumbing.")
    else:
        subtargets = define_subtargets(config, sample=sample, fmt="wide")
        LOG.info("Done defining %s subtargets.", subtargets.shape)

    LOG.info("Write result to %s", output)
    if args.test:
        LOG.info("TEST pipeline plumbing.")
    else:
        output = write(subtargets, to_path=(output or default_hdf("subtargets")))
        LOG.info("Done writing results to %s", output)

    LOG.warning("DONE, defining subtargets.")
    return output


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate flatmap columnar sub-targets.")

    parser.add_argument("config",
                        help="Path to the configuration to run the pipeline.")

    parser.add_argument("-s", "--sample",
                        help="A float to sample with", default=None)

    parser.add_argument("-o", "--output",
                        help="Path to the directory to output in.", default=None)

    parser.add_argument("--dry-run", dest="test",  action="store_true"
                        help=("Use this to test the pipeline's plumbing "
                              "before running any juices through it."))
    parser.set_default(test=False)

    args = parser.parse_args()

    LOG.warning(str(args))
    main(args)
