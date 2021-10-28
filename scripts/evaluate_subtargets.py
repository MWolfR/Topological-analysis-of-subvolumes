import numpy
import importlib
import os
import logging
import pandas

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


def get_grouping_names(neuron_df):
    do_not_group = ["gid"]
    do_group = list(neuron_df.index.names)
    for x in do_not_group:
        if x in do_group:
            do_group.remove(x)
    return do_group


def evaluate_target_composition(neuron_df):
    do_not_evaluate = ["x", "y", "z"]
    do_group = get_grouping_names(neuron_df)

    nrns = neuron_df[[col for col in neuron_df.columns if col not in do_not_evaluate]]
    result = nrns.apply(lambda series: compare_group_distributions(series, group_index=do_group), axis=0)
    result.columns = [col + "_composition" for col in result.columns]
    return result


def evaluate_neuron_counts(neuron_df):
    do_group = get_grouping_names(neuron_df)

    nrn_counts = neuron_df.groupby(do_group).apply(len)
    nrn_rel = (nrn_counts - nrn_counts.median()) / (nrn_counts.median() + nrn_counts)
    nrn_rel.name = "neuron_counts"
    return nrn_rel


def evaluate_orthoganality(neuron_df, circuit_dict):
    import bluepy
    from scipy.spatial.transform import Rotation

    do_group = get_grouping_names(neuron_df)

    circuit_dict = dict([(circ_name, bluepy.Circuit(circ_path)) for circ_name, circ_path in circuit_dict.items()])
    orientation_dict = dict([(circ_name, circ.atlas.load_data("orientation"))
                             for circ_name, circ in circuit_dict.items()])

    nxyz = neuron_df[["x", "y", "z"]]
    A = nxyz.groupby(do_group).apply(numpy.array)
    tgt_centers = A.apply(lambda x: x.mean(axis=0))
    B = A.combine(tgt_centers, func=lambda coords, c: coords - c)

    orientation_atlas = A.index.to_frame()["circuit"].apply(lambda c: orientation_dict[c])
    tgt_orient = tgt_centers.combine(orientation_atlas, func=lambda c, orient: orient.lookup(c))
    tgt_rot = tgt_orient.apply(lambda o_vec: Rotation.from_quat(numpy.hstack([o_vec[1:], o_vec[0:1]])).inv())
    C = B.combine(tgt_rot, func=lambda coords, rot: rot.apply(coords))

    def skewed(c):
        return numpy.abs(numpy.corrcoef(c[:, 0], c[:, 1])[0, 1]) \
               + numpy.abs(numpy.corrcoef(c[:, 2], c[:, 1])[0, 1])
    result = C.apply(skewed)
    result.name = "Non-orthogonality"
    return result


def perform_evaluations(neuron_df, list_of_metrics, circuit_dict):
    all_res = []
    for metric in list_of_metrics:
        if metric == "neuron_counts":
            all_res.append(evaluate_neuron_counts(neuron_df))
        elif metric == "target_composition":
            all_res.append(evaluate_target_composition(neuron_df))
        elif metric == "orthogonality":
            all_res.append(evaluate_orthoganality(neuron_df, circuit_dict))
        else:
            raise ValueError("Unknown metric: {0}".format(metric))
    return pandas.concat(all_res, axis=1)


def main(fn_cfg):
    cfg = read_cfg.read(fn_cfg)
    paths = cfg["paths"]

    if "tgt_evaluations" not in paths:
        raise RuntimeError("No defined target quality dataset in config!")
    if "neurons" not in paths:
        raise RuntimeError("No neurons in config!")
    if "circuit" not in paths:
        raise RuntimeError("No circuits defined in config!")

    circuits = paths["circuit"]
    path_neurons = paths["neurons"]

    LOG.warning("Read neuron info from path %s", path_neurons)
    neurons = read_results(path_neurons, for_step="evaluate_targets")
    LOG.warning("Number of neurons read: %s", neurons.shape[0])

    cfg = cfg["parameters"].get("test_subtargets", {})
    metrics = cfg.get("metrics", [])
    extracted = perform_evaluations(neurons, metrics, circuits)
    write(extracted, to_path=paths.get("tgt_evaluations", default_hdf("tgt_evaluations")), format="table")
