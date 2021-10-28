import numpy
import importlib
import os
import logging

from scipy.stats import wasserstein_distance

from write_results import read as read_results, write, default_hdf

read_cfg = importlib.import_module("read_config")


LOG = logging.getLogger("Evaluate flatmap subtargets")
LOG.setLevel(os.environ.get("LOGLEVEL", "INFO"))


def compare_group_distributions(neuron_series, group_index="subtarget", normalize_distance=True):
    all_counts = neuron_series.value_counts(normalize=True)
    divide_by = 1.0
    if normalize_distance:
        divide_by = len(all_counts)

    def grp_ws_distance(grp_series):
        grp_counts = grp_series.value_counts(normalize=True)
        if all_counts.index.is_categorical():
            assert numpy.all(all_counts.index.categories == grp_counts.index.categories)
            return wasserstein_distance(all_counts.index.codes, grp_counts.index.codes,
                                        all_counts.values, grp_counts.values)
        return wasserstein_distance(all_counts.index, grp_counts.index, all_counts.values, grp_counts.values)
    return neuron_series.groupby(group_index).apply(grp_ws_distance) / divide_by


def evaluate_target_composition(neuron_df):
    do_not_evaluate = ["x", "y", "z"]
    do_not_group = ["gid"]

    do_group = list(neuron_df.index.names)
    for x in do_not_group:
        if x in do_group:
            do_group.remove(x)

    nrns = neuron_df[[col for col in neuron_df.columns if col not in do_not_evaluate]]
    result = nrns.apply(lambda series: compare_group_distributions(series, group_index=do_group), axis=0)
    return result


def perform_evaluations(neuron_df, list_of_metrics):
    # TODO: Use list_of_metrics to decide which analyses to run
    return evaluate_target_composition(neuron_df)


def main(fn_cfg):
    cfg = read_cfg.read(fn_cfg)
    paths = cfg["paths"]

    if "tgt_evaluations" not in paths:
        raise RuntimeError("No defined target quality dataset in config!")
    if "neurons" not in paths:
        raise RuntimeError("No neurons in config!")

    path_neurons = paths["neurons"]

    LOG.warning("Read neuron info from path %s", path_neurons)
    neurons = read_results(path_neurons, for_step="evaluate_targets")
    LOG.warning("Number of neurons read: %s", neurons.shape[0])

    cfg = cfg["parameters"].get("test_subtargets", {})
    metrics = cfg.get("metrics", [])
    extracted = perform_evaluations(neurons, metrics)
    write(extracted, to_path=paths.get("tgt_evaluations", default_hdf("tgt_evaluations")), format="table")
