"""Define flatmap column subtargets for a circuit.
"""
import os
import argparse
import logging

import read_config
from write_results import write, default_hdf
from hexgrid import SubtargetConfig, define_subtargets


LOG = logging.getLogger("Generate flatmap subtargets")
LOG.setLevel(os.environ.get("LOGLEVEL", "INFO"))


def main(args):
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
                                   format=args.format if args.format else config.fmt_dataframe)

    LOG.info("Write result to %s", config.output)
    output = write(subtargets, to_path=(config.output or default_hdf("subtargets")))
    LOG.info("DONE writing results to %s", output)
    return output


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s",
                        level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S")
    parser = argparse.ArgumentParser(description="Generate flatmap columnar sub-targets.")

    parser.add_argument("config", help="Path to the configuration to generate sub-targets")

    parser.add_argument("-s", "--sample", help="A float to sample with", default=None)

    parser.add_argument("-f", "--format", help="Format for annotating the columns.", default=None)

    args = parser.parse_args()

    LOG.warning(str(args))
    main(args)
