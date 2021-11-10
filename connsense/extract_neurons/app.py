"""Extract neuron properties for each of the subtargets generated
by `define-subtargets`
"""
from argparse import ArgumentParser

from ..io import read_config as read_cfg
from ..io.write_results import read as read_results, write, default_hdf
from ..io import logging

from .import run

STEP = "extract-neurons"
LOG = get_logger(STEP)


def main(args):
    """Launch extraction of  neurons."""
    LOG.warning("Extract neurons for subtargets.")

    return run(args.config, output=args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract neuron properties.")

    parser.add_argument("config",
                        help="Path to the configuration to run the pipeline.")

    parser.add_argument("-o", "--output",
                        help="Path to the directory to output in.", default=None)

    parser.add_argument("--dry-run", dest="dry_run",  action="store_true"
                        help=("Use this to test the pipeline's plumbing "
                              "before running any juices through it."))
    parser.set_default(test=False)

    args = parser.parse_args()

    LOG.warning("Run %s: %s", STEP, args)
    main(args)
