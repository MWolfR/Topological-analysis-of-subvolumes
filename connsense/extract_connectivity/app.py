
"""Extract subtargets' connectivity.
"""
import importlib
from argparse import ArgumentParser

from ..io.write_results import (read as read_results,
                                write_toc_plus_payload as write,
                                default_hdf)
from ..io import read_config as read_cfg
from ..io import logging

STEP = "extract-connectivity"
LOG = logging.get_logger(STEP)


def main(args):
    """..."""
    LOG.warning("Run extractions using the config at %s", fn_cfg)

    cfg = read_cfg.read(args.config)
    return run(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract connectivity for sub-targets.")

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
