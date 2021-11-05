"""Define flatmap column subtargets for a circuit.
"""
import os
from argparse import ArgumentParser
import logging

from .io import read_config
from .io.write_results import write, default_hdf
from .io.logging get_logger

STEP = "define-subtargets"
LOG = get_logger(STEP)


def main(args):
    """Interpret `args` and launch."""
    from flatmap_utility.hexgrid import SubtargetConfig, define_subtargets

    LOG.info("Get subtargets for %s", args)

    LOG.info("Load the config %s", args.config)
    config = SubtargetConfig(args.config, reader=read_config)

    LOG.info("Compute sub-targets with:")
    LOG.info("\tinput circuits %s: ", config.input_circuit)
    LOG.info("\tinput flatmaps %s: ", config.input_flatmap.keys())
    LOG.info("\tdesired mean number of cells in a column: %s", config.mean_target_size)
    LOG.info("\toutput in format %s goes to %s", args.format, config.output)

    if args.sample:
        LOG.info("Sample a fraction %s of the cells", args.sample)
        sample = float(args.sample)
    else:
        sample = None

    subtargets = define_subtargets(config, sample_frac=sample,
                                   format=(args.format if args.format
                                           else config.fmt_dataframe)

    output = config.output
    if args.output:
        try:
            _, hdf_key = output
        except TypeError:
            output = args.output
        else:
            output = (args.output, hdf_key)

    LOG.info("Write result to %s", output)
    output = write(subtargets, to_path=(output or default_hdf("subtargets")))
    LOG.info("DONE writing results to %s", output)
    return output


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate flatmap columnar sub-targets.")

    parser.add_argument("config",
                        help="Path to the configuration to generate sub-targets")

    parser.add_argument("-s", "--sample",
                        help="A float to sample with", default=None)

    parser.add_argument("-f", "--format",
                        help="Format for annotating the columns.", default=None)

    parser.add_argument("-o", "--output",
                        help="Path to the directory to output in.", default=None)

    args = parser.parse_args()

    LOG.warning(str(args))
    main(args)
