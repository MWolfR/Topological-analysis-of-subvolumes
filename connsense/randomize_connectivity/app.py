"""Randomize subtarget connectivity."""

from argparse import ArgumentParser

from randomization import Algorithm

from ..io import logging

from .import run

STEP = "randomize-connectivity"
LOG = logging.get_logger(STEP)


def main(args):
    """..."""
    return run(args.config, output=args.output,  batch_size=args.batch_size,
               sample=args.sample, dry_run=args.dry_run)


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
