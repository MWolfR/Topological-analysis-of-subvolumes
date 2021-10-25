import importlib
import pandas
import bluepy
from os import path


read_cfg = importlib.import_module("read_config")


def run_extractions(circuits, subtargets, cfg):
    params = cfg.get("properties", [])
    if len(params) == 0:
        print("Warning: No properties to extract given. This step will do nothing!")
    if not path.isfile(subtargets):
        raise RuntimeError("defined subtargets at {0} not existing. Run subtarget definition step first!".format(subtargets))

    circuits = dict([(k, bluepy.Circuit(v)) for k, v in circuits.items()])
    subtargets = pandas.read_hdf(subtargets, key="dataframe")

    circuit_frame = subtargets.index.to_frame().apply(lambda x: circuits[x["circuit"]], axis=1)

    def func(circ, gids):
        props = circ.cells.get(gids, properties=params)
        props.columns.name = "property"
        props.index.name = "gid"
        return props

    A = circuit_frame.combine(subtargets, func)
    out = pandas.concat(A.values, keys=A.index.values, names=A.index.names)
    return out


def write_results(extracted, fn_out):
    extracted.to_hdf(fn_out)


def main(fn_cfg):
    cfg = read_cfg.read(fn_cfg)
    paths = cfg["paths"]
    if "circuit" not in paths:
        raise RuntimeError("No circuits defined in config!")
    if "defined_columns" not in paths:
        raise RuntimeError("No defined columns in config!")
    if "neurons" not in paths:
        raise RuntimeError("No neurons in config!")

    circuits = cfg["paths"]["circuit"]
    targets = cfg["paths"]["defined_columns"]
    cfg = cfg["parameters"].get("extract_neurons", {})
    extracted = run_extractions(circuits, targets, cfg)
    write_results(extracted, paths["neurons"])


if __name__ == "__main__":
    import sys
    print(main(sys.argv[1]))
