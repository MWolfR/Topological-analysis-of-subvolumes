"""ConnSense: An app to run connectome utility pipeline.
"""
from argparse import ArgumentParser

from connsense import pipeline
from connsense.io import logging

LOG = logging.get_logger("Toplogical analysis of flatmapped subtargets.")


def main(args):
    """..."""
    LOG.info("Initialize the topological analysis pipeline.")
    topaz = pipeline.TopologicalAnalysis(args.config, mode="run")

    LOG.info("Run the pipeline.")
    result = topaz.run(args.steps, sample=args.sample, output=args.output,
                       dry_run=args.test)
    LOG.info("DONE running pipeline: %s", result)

    return result


if __name__ == "__main__":
    LOG.warning("Parse arguments.")
    parser = ArgumentParser(description="Topological analysis of flatmapped subtargets")

    parser.add_argument("config",
                        help="Path to the configuration to run the pipeline.")

    parser.add_argument("-s", "--steps", type=str, nargs='+',
                        help="Subset of steps to run.", default=None)

    parser.add_argument("--output",
                        help="Path to the directory to output in.", default=None)

    parser.add_argument("--sample",
                        help="A float to sample subtargets with", default=None)

    parser.add_argument("--dry-run", dest="test",  action="store_true",
                        help=("Use this to test the pipeline's plumbing "
                              "before running any juices through it."))
    parser.set_defaults(test=False)

    args = parser.parse_args()

    LOG.warning(str(args))
    main(args)
