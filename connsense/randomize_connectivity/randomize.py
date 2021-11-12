"""Randomize connectivity based on neuron properties.
"""
from multiprocessing import Process, Manager

import numpy as np
import pandas as pd

from ..io.write_results import (read as read_results)
from ..io import logging

STEP = "randomize-connectivity"

LOG = logging.get_logger(STEP)


def get_neuron_properties(hdf_path, hdf_group):
    """
    TODO: move this over to a common place.
    """
    return (read_results((hdf_path, hdf_group), STEP)
            .droplevel(["flat_x", "flat_y"])
            .reset_index()
            .set_index(["circuit", "subtarget"]))


def randomize_table_of_contents(toc, using_neuron_properties,
                                applying_algorithms,
                                with_batches_of_size=None,
                                ncores=72):
    """..."""
    N = toc.shape[0]

    LOG.info("Randomize %s subtargets using  %s.",
             N, [a.name for a in applying_algorithms])

    neurons = using_neuron_properties
    algorithms = applying_algorithms
    batch_size = with_batches_of_size or int(N / (ncores-1)) + 1

    batched = (toc.to_frame()
               .assign(batch=np.array(np.floor(np.arange(N) / batch_size),
                                      dtype=int)))

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

            return (batch.assign(idx=range(batch.shape[0])).apply(shuffle_row, axis=1)
                    .rename("matrix"))

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
