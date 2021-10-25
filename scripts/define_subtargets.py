import os
from collections.abc import Mapping
from pathlib import Path
import argparse
from lazy import lazy
import logging

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sbn


from voxcell.voxel_data import VoxelData
from bluepy import Cell, Circuit
from bluepy.exceptions import BluePyError

import read_config
from hexgrid import SubtargetConfig, define_subtargets


LOG = logging.getLogger("Generate flatmap subtargets")
LOG.setLevel(os.environ.get("LOGLEVEL", "INFO"))

def write_results(extracted, path_output):
    """Expecting the path to output be that to a `*.h5`."""
    extracted.to_hdf(path_output, key="datarame", mode="w")
    return path_output

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

    subtargets = define_subtargets(config, sample_frac=sample, format=args.format)

    LOG.info("Write result to %s", config.output)
    write_results(subtargets, config.output)
    return config.output


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s",
                        level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S")
    parser = argparse.ArgumentParser(description="Generate flatmap columnar sub-targets.")

    parser.add_argument("config", help="Path to the configuration to generate sub-targets")

    parser.add_argument("-s", "--sample", help="A float to sample with", default=None)

    parser.add_argument("-f", "--format", help="Format for annotating the columns.", default="wide")

    args = parser.parse_args()

    LOG.warning(str(args))
    main(args)
