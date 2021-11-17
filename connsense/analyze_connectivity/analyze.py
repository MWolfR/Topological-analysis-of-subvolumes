"""Analyze connectivity of subtargets
"""
from multiprocessing import Process, Manager

import numpy as np
import pandas as pd

from analysis import  Analysis
from ..io.write_results import (read as read_results)
from ..io import logging

STEP = "analyze-connectivity"

LOG = logging.get_logger(STEP)


def get_neuron_properties(hdf_path, hdf_group):
    """..."""
    return (read_results((hdf_path, hdf_group), STEP)
            .droplevel(["flat_x", "flat_y"])
            .reset_index()
            .set_index(["circuit", "subtarget"]))


def analyze_table_of_contents(toc, using_neuron_properties,
                              applying_analyses,
                              with_batches_of_size=None,
                              ncores=72):

    """..."""
    LOG.info("Analyze connectivity: %s", toc.shape[0])

    N = toc.shape[0]

    neurons = using_neuron_properties
    analyses = applying_analyses
    LOG.info("Analyze %s subtargets using  %s.", N, [a.name for a in analyses])

    batch_size = with_batches_of_size or int(N / (ncores-1)) + 1

    batched = (toc.to_frame().
               assign(batch=np.array(np.floor(np.arange(N) / batch_size),
                                     dtype=int)))

    n_analyses = len(analyses)
    n_batches = batched.batch.max() + 1

    def get(batch, label=None, bowl=None):
        """..."""
        LOG.info("ANALYZE batch %s / %s with %s targets and columns %s",
                 label, n_batches, batch.shape[0], batch.columns)

        def get_neurons(row):
            """..."""
            index = dict(zip(batch.index.names, row.name))
            return (neurons.loc[index["circuit"], index["subtarget"]]
                    .reset_index(drop=True))

        def analyze(analysis, at_index):
            """..."""

            def analyze_row(r):
                """..."""
                log_info = (f"Batch {label} Analysis {analysis.name} "
                            f"({at_index}/ {n_analyses}) "
                            f"matrix {r.idx} / {batch.shape[0]}")

                return analysis.apply(r.matrix, get_neurons(r), log_info)

            return (batch.assign(idx=range(batch.shape[0])).apply(analyze_row, axis=1)
                    .rename(analysis.quantity))

        analyzed = pd.concat([analyze(a, i) for i, a in enumerate(analyses)],
                             axis=0, keys=[a.name for a in analyses],
                             names=["analysis"])

        LOG.info("DONE batch %s / %s with %s targets, columns %s: randomized to shape %s",
                 label, n_batches, batch.shape[0], batch.columns, analyzed.shape)

        bowl[label] = analyzed
        return analyzed

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

    result = pd.concat([analyzed for _, analyzed in bowl.items()], axis=0)

    LOG.info("DONE analyzing %s subtargets using  %s.", N, [a.name for a in analyses])

    return result
