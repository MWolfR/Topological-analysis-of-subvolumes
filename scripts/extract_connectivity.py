import importlib
import pandas
import bluepy
import h5py
import numpy
import logging
import os

from tqdm import tqdm
from scipy import sparse

from write_results import read as read_results, write_toc_plus_payload as write, default_hdf

read_cfg = importlib.import_module("read_config")


LOG = logging.getLogger("Extract connection matrices")
LOG.setLevel(os.environ.get("LOGLEVEL", "INFO"))


def connection_matrix_for_gids(sonata_fn, gids):
    idx = numpy.array(gids) - 1  # From gids to sonata "node" indices (base 0 instead of base 1)
    h5 = h5py.File(sonata_fn, "r")['edges/default']  # TODO: Instead of hard coding "default" that could be a config parameter
    N = len(gids)

    indices = []
    indptr = [0]
    for id_post in tqdm(idx):
        ids_pre = []
        ranges = h5['indices']['target_to_source']['node_id_to_ranges'][id_post, :]
        for block in h5['indices']['target_to_source']['range_to_edge_id'][ranges[0]:ranges[1], :]:
            ids_pre.append(h5['source_node_id'][block[0]:block[1]])
        row_ids = numpy.nonzero(numpy.in1d(idx, numpy.hstack(ids_pre)))[0]
        indices.extend(row_ids)
        indptr.append(len(indices))
    mat = sparse.csc_matrix((numpy.ones(len(indices), dtype=bool), indices, indptr), shape=(N, N))
    return mat


def find_connectome_files(circuit_dict):
    def lookup_sonata_files(circ_name, conn_lst):
        circ = circuit_dict[circ_name]
        return [circ.config["connectome"] if conn == "local"
                else circ.config["projections"][conn]
                for conn in conn_lst]

    return lookup_sonata_files


def run_extraction(circuits, subtargets, list_of_connectomes):
    if len(list_of_connectomes) == 0:
        LOG.warning("No connectomes defined. This step will do nothing!")
    circuits = dict([(k, bluepy.Circuit(v)) for k, v in circuits.items()])

    connectome_names = [" + ".join(lst) for lst in list_of_connectomes]
    connectome_series = pandas.Series(list_of_connectomes, index=connectome_names)
    connectome_series.index.name = "connectome"

    # Cross product with list of connectomes
    connectomes = subtargets.index.to_frame().apply(lambda _: connectome_series, axis=1).stack()
    # Connectome to sonata filenames
    sonata_files = connectomes.index.to_frame()["circuit"].combine(connectomes, func=find_connectome_files(circuits))
    # Because pandas is utterly stupid
    subtargets = subtargets[sonata_files.index.droplevel("connectome")]
    subtargets.index = sonata_files.index

    # Extract connection matrices
    def extract(lst_connectome_files, gids):
        mat = sparse.csc_matrix((len(gids), len(gids)), dtype=bool)
        for conn_file in lst_connectome_files:
            mat = mat + connection_matrix_for_gids(conn_file, gids)
        return mat

    con_mats = sonata_files.combine(subtargets, func=extract)
    return con_mats


def main(fn_cfg):
    cfg = read_cfg.read(fn_cfg)
    paths = cfg["paths"]
    if "circuit" not in paths:
        raise RuntimeError("No circuits defined in config!")
    if "defined_columns" not in paths:
        raise RuntimeError("No defined columns in config!")
    if "connection_matrices" not in paths:
        raise RuntimeError("No connection matrices in config!")

    cfg = cfg["parameters"].get("extract_connectivity", {})
    circuits = paths["circuit"]
    path_targets = paths["defined_columns"]

    LOG.warning("Read targets from path %s", path_targets)
    targets = read_results(path_targets, for_step="subtargets")
    LOG.warning("Number of targets read: %s", targets.shape[0])

    extracted = run_extraction(circuits, targets, cfg.get("connectomes", []))
    write(extracted, to_path=paths.get("connection_matrices", default_hdf("connection_matrices")), format="table")


if __name__ == "__main__":
    import sys
    print(main(sys.argv[1]))