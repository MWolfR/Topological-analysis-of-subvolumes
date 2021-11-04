"""Randomize circuit connectivity matrices."""

import os
from collections import namedtuple
from importlib import import_module
import argparse
import logging
from multiprocessing import Process, Pool, Manager

import numpy as np
import pandas as pd
from scipy import sparse

import read_config
from write_results import (read_toc_plus_payload, write_toc_plus_payload,
                           default_hdf, read as read_result)


LOG = logging.getLogger("Generate randomized subtarget connectivities.")
LOG.setLevel(os.environ.get("LOGLEVEL", "INFO"))


def erdos_renyi(adjacency, node_properties=None, direction=None, in_format=None):
    """Shuffle and digest an adjacency matrix into a sample from a
    population of Erdos-Renyi graphs.
    TODO: What about direction?
    """
    N, M = adjacency.shape
    assert M == N, (N, M)

    direction = direction or "in"
    assert direction in ("in", "out"), direction

    in_format = in_format or "CSR"
    assert in_format in ("CSC", "CSR"), in_format

    if direction == "in":
        adjacency = adjacency.tocsc()
    else:
        adjacency = adjacency.tocsr()

    degrees = np.diff(adjacency.indptr)
    randomized_indices = np.hstack([np.random.choice(range(N), k, replace=False)
                                   for k in degrees])
    csc = sparse.csc_matrix((adjacency.data, randomized_indices, adjacency.indptr),
                            (N, N))
    return csc if in_format=="CSC" else csc.tocsr()


class Algorithm:
    """..."""
    available = {"ERIN": lambda a, n: erdos_renyi(a, n, direction="in"),
                 "EROUT": lambda a, n: erdos_renyi(a, n, direction="out")}

    def __init__(self, name, source, args=None, kwargs=None):
        """..."""
        self._name = name
        self._shuffle = self.load_method(source)
        self._args = args or tuple()
        self._kwargs = kwargs or {}

    @property
    def name(self):
        """..."""
        return self._name

    def load_method(self, source):
        """..."""
        try:
            return self.available[source]
        except KeyError:
            pass

        self._module = import_module(source)

        return self._module.shuffle

    def shuffle(self, adjacency, node_properties=None, log_info=None):
        """..."""
        try:
            matrix = adjacency.matrix
        except AttributeError:
            pass

        if node_properties is not None:
            assert node_properties.shape[0] == matrix.shape[0]

        LOG.info("%sShuffle a matrix of shape %s ",
                 "" if not log_info else log_info + ":\n\t",
                 matrix.shape)
        result = self._shuffle(matrix, node_properties, *self._args, **self._kwargs)

        LOG.info("%sDONE shuffling a matrix of shape %s ",
                 "" if not log_info else log_info + ":\n\t",
                 matrix.shape)

        return result


def randomize_table_of_contents(toc, neurons, algorithms, batch_size=None):
    """..."""
    N = toc.shape[0]

    LOG.info("Randomize %s subtargets using  %s.", N, [a.name for a in algorithms])

    if not batch_size:
        batch_size = int(N / 72) + 1

    batched = (toc.to_frame()
               .assign(batch=np.array(np.floor(np.arange(N) / batch_size), dtype=int)))

    n_algos = len(algorithms)
    n_batches = batched.batch.max() + 1

    def get(batch, label=None, bowl=None):
        """..."""
        LOG.info("RANDOMIZE batch %s / %s with %s targets and columns %s",
                 label, n_batches, batch.shape[0], batch.columns)

        def get_neurons(row):
            """..."""
            index = dict(zip(batch.index.names, row.name))
            return (neurons.loc[index["circuit"], index["subtarget"]]
                    .reset_index(drop=True))

        def shuffle(algorithm, at_index):
            """..."""

            def shuffle_row(r):
                """..."""
                log_info = (f"Batch {label} Algorithm {algorithm.name} "
                            f"({at_index}/ {n_algos}) "
                            f"matrix {r.idx} / {batch.shape[0]}")

                return algorithm.shuffle(r.matrix, get_neurons(r), log_info)

            return batch.assign(idx=range(batch.shape[0])).apply(shuffle_row, axis=1)

        randomized = pd.concat([shuffle(a, i) for i, a in enumerate(algorithms)],
                               axis=0, keys=[a.name for a in algorithms],
                               names=["algorithm"])

        LOG.info("DONE batch %s / %s with %s targets, columns %s: randomized to shape %s",
                 label, n_batches, batch.shape[0], batch.columns, randomized.shape)

        bowl[label] = randomized
        return randomized

    manager = Manager()
    bowl = manager.dict()
    processes = []

    for i, batch in batched.groupby("batch"):

        p = Process(target=get, args=(batch,),
                    kwargs={"label": "chunk-{}".format(i), "bowl": bowl})
        p.start()
        processes.append(p)

    LOG.info("LAUNCHED")

    for p in processes:
        p.join()

    result = pd.concat([randomized for _, randomized in bowl.items()], axis=0)

    LOG.info("DONE randomize %s subtargets using  %s.", N, [a.name for a in algorithms])

    return result


def main(args):
    """..."""
    config = read_config.read(args.config)

    paths = config["paths"]
    if "circuit" not in paths:
        raise RuntimeError("No circuits defined in config!")
    if "defined_columns" not in paths:
        raise RuntimeError("No defined columns in config!")
    if "neurons" not in paths:
        raise RuntimeError("No neurons in config!")
    if "connection_matrices" not in paths:
        raise RuntimeError("No connection matrices in config!")
    if "randomized_matrices" not in paths:
        raise RuntimeError("No randomized matrices in config!")

    hdf_path, hdf_key = paths["neurons"]

    LOG.warning("Load extracted neuron properties from %s", hdf_path)
    neurons = (read_result((hdf_path, hdf_key), "neurons")
               .droplevel(["flat_x", "flat_y"])
               .reset_index()
               .set_index(["circuit", "subtarget"]))

    LOG.warning("DONE loading extracted neuron properties: %s", neurons.shape)

    hdf_path, hdf_key = paths["connection_matrices"]

    LOG.warning("LOAD table of contents for connectivity matrices.")
    toc = read_toc_plus_payload((hdf_path, hdf_key), "analysis")
    LOG.warning("DONE %s subtargets toc", toc.shape[0])

    toc = toc.rename("matrix") #TODO: Make connectivity extraction rename before returing TOC

    if args.sample:
        sample = np.float(args.sample)

        toc = toc.sample(frac=sample) if sample < 1 else toc.sample(n=int(sample)).rename("matrix")

    parameters = config["parameters"].get("randomize_connectivity")
    algorithms = {Algorithm(description["name"], description["source"],
                            description["args"], description["kwargs"])
                  for label, description in parameters["algorithms"].items()}

    hdf_path, hdf_key = paths.get("randomized_matrices", default_hdf("randomized_matrices"))
    if args.output:
        hdf_path = args.output

    randomized = randomize_table_of_contents(toc, neurons, algorithms, args.batch_size)

    LOG.info("Write %s randomized matrices.", randomized.shape[0])
    write_toc_plus_payload(randomized,
                           to_path=(hdf_path, hdf_key),
                           format="table")
    LOG.info("DONE writing %s randomized matrices.", randomized.shape[0])


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s",
                        level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S")
    parser = argparse.ArgumentParser(description="Randomize connectivity of columnar sub-targets.")

    parser.add_argument("config",
                        help="Path to the configuration to generate sub-targets")

    parser.add_argument("-s", "--sample",
                        help="A float to sample with, to be used for testing",
                        default=None)

    parser.add_argument("-b", "--batch-size",
                        help="Number of targets to process in a single thread.",
                        default=None)

    parser.add_argument("-o", "--output",
                        help="Path to the directory to output in, to be used for testing",
                        default=None)

    args = parser.parse_args()

    LOG.warning(str(args))
    main(args)
