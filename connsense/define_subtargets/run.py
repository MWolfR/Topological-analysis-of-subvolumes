"""Define flatmap column subtargets for a circuit.
"""
from argparse import ArgumentParser
from .io.logging import get_logger
from .import run

STEP = "define-subtargets"
LOG = get_logger(STEP)



if __name__ == "__main__":

    LOG.warning("RUN %s", STEP)

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

    run(args.config, args.output, args.sample, args.dry_run)

    LOG.warning("DONE running %s", STEP)
