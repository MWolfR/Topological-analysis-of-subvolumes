"""Extract neuron properties for each of the subtargets generated
by `define-subtargets`
"""
from argparse import ArgumentParser

from .io import read_config as read_cfg
from .io.write_results import read as read_results, write, default_hdf
from .io import logging

STEP = "extract-neurons"
LOG = get_logger(STEP)


def main(args):
    """Launch extraction of  neurons."""
    LOG.warning("Extract neurons for subtargets.")
    cfg = read_cfg.read(args.config)
    paths = cfg["paths"]

    if "circuit" not in paths:
        raise RuntimeError("No circuits defined in config!")
    if "defined_columns" not in paths:
        raise RuntimeError("No defined columns in config!")
    if "neurons" not in paths:
        raise RuntimeError("No neurons in config!")

    if not args.test:
        circuits = cfg["paths"]["circuit"]

    path_targets = cfg["paths"]["define-subtargets"]

    LOG.info("READ targets from path %s", path_targets)
    if args.test:
        LOG.info("TEST pipeline plumbing.")
    else:
        targets = read_results(path_targets, for_step="define-subtargets")
        LOG.info("DONE read number of targets read: %s", targets.shape[0])

    cfg = cfg["parameters"].get(STEP, {})
    params = cfg.get("properties", [])

    LOG.info("Cell properties to extract: %s", params)
    if args.test:
        LOG.info("TEST pipeline plumbing.")
    else:
        extracted = run_extractions(circuits, targets, params)
        LOG.info("DONE, extracting %s", params)

    output = paths.get(STEP, default_hdf(STEP))
    LOG.info("WRITE neuron properties to archive %s\n\t under group %s",
             output[0], output[1])
    if args.test:
        LOG.info("TEST pipeline plumbing.")
    else:
        write(extracted, to_path=output, format="table")
        LOG.info("DONE neuron properties to archive.")

    LOG.warning("DONE extract neurons for %s subtargets", targets.shape[0])

    return output


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract neuron properties.")

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
