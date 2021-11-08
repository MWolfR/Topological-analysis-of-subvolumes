"""Bin cell flatmap positions in a hexagonal grid.
"""

import pandas as pd
import numpy as np

from bluepy import Cell

from .tessellate import TriTille
import flatmap_utility as flattened

XYZ = [Cell.X, Cell.Y, Cell.Z]

from .io import logging
LOG = logging.get_logger("Flatmap Utility")


def get_cell_ids(circuit, target=None, sample=None):
    """..."""
    gids = pd.Series(circuit.cells.ids(target), name="gid")

    if isinstance(sample, int):
        return gids.sample(n=sample)

    if isinstance(sample, float):
        return gids.sample(frac=sample)

    assert not sample, sample

    return gids


def get_cell_positions(circuit, target=None, sample=None):
    """..."""
    positions = circuit.cells.get(target, properties=XYZ)
    positions.index.name = "gid"
    if isinstance(sample, int):
        return positions.sample(n=sample)

    if isinstance(sample, float):
        return positions.sample(frac=sample)

    assert not sample, sample

    return positions


def cached(circuit, method):
    """..."""
    try:
        value = getattr(circuit, method.__qualname__)
    except AttributeError:
        value = method(circuit)
        setattr(circuit, method.__qualname__, value)
    return value


def flatmap_positions(circuit):
    """..."""
    flatmap = circuit.atlas.load_data("flatmap")
    orientations = circuit.atlas.load_data("orientation")
    return (flattened
            .supersampled_neuron_locations(circuit, flatmap, orientations)
            .rename(columns={"flat x": "x", "flat y": "y"}))


def get_flatmap(circuit, target=None, sample=None, subpixel=True, dropna=True):
    """..."""
    LOG.info("GET flatmap for target %s sample %s%s",
             target, sample, ", with subsample resolution." if subpixel else ".")

    if not subpixel:
        flatmap = circuit.atlas.load_data("flatmap")
        positions = get_cell_positions(circuit, target, sample)

        fpos =  pd.DataFrame(flatmap.lookup(positions.values),
                             columns=["x", "y"], index=positions.index)

        LOG.info("DONE getting flatmap")
        return fpos[np.logical_and(fpos.x >= 0, fpos.y >= 0)]

    flat_xy = cached(circuit, flatmap_positions)
    if target is not None or sample is not None:
        gids = get_cell_ids(circuit, target, sample)
        in_target = flat_xy.reindex(gids)
    else:
        in_target = flat_xy

    assert in_target.index.name == "gid", in_target.index.name
    LOG.info("DONE getting flatmap")
    return in_target.dropna() if dropna else in_target


def flatmap_hexbin(circuit, radius=10, gridsize=120, sample=None):
    """Bin circuit cell's flatmap coordinates in hexagonal grid.
    """
    positions = get_cell_positions(circuit, sample=sample)
    flatmap = get_flatmap(circuit, positions)

    tritiling = TriTille(radius)

    bins = tritiling.bin_hexagonally(positions, use_columns_row_indexing=True)
    return bins


def name_subtarget(hexbin):
    """Name the subtargets using their column and row index."""
    return f"R{hexbin.row};C{hexbin.col}"


def generate_subtargets(of_radius, in_circuit, with_flatmap=None,
                        base_target=None,
                        origin=None, angle=None):
    """"Generate subtargets of a base target in a circuit's flatspace.
    Each subtarget will flatmap to a tile in a grid of hexagons of given radius.

    TODO: origin and angle: to add more freedom in localizing a hexagon to study.
    """
    if origin or angle:
        raise NotImplementedError("TODO")


    circuit = in_circuit
    flatmap = with_flatmap or circuit.flatmap
    tritille = TriTille(of_radius)
    hexmap = tritille.bin_hexagonally(flatmap, use_columns_row_indexing=False)
    grid = tritille.locate_grid(hexmap)
    annotation = tritille.annotate(grid, using_column_row=True)
    gids_by_gridpoint = hexmap.reset_index().set_index(["i", "j"])
    annotated = grid.assign(subtarget=annotation.loc[grid.index])
    return gids_by_gridpoint.join(annotated).reset_index().set_index("subtarget")


def generate_subtargets_0(circuit, flatmap=None, radius=None, size=None,
                        target=None, sample=None,
                        naming_scheme=None):
    """...
    Input
    ----------
    circuit : BluePyCircuit
    flatmap : flatmap NRRD, omit to use the one on the circuit's atlas
    radius : of the balls that would pack the resulting hexgrid
    size : approximate number of neurons per subtarget.
    target : a BluePy target cell type to define subtargets in, omit to use all cells.
    sample : an int > 1 or float < 1 to use a random sample of the entire target.
    naming_scheme : A call-back or a dict to use to name the subtargets from their location

    Output
    ----------
    pd.Series
    Indexed by : flat_x, flat_y : the x, y-coordinates of the center of the subtarget in the flatmap.
    ~            subtarget_name : a pretty name for the subtarget.
    Values : lists of gids
    """
    assert radius or size, "Need one to define subtargets."
    assert not (radius and size), "Cannot yet use both to define subtargets"

    def for_radius(r):
        """..."""
        return generate_subtargets(circuit, flatmap, r, target=target,
                                   sample=sample, naming_scheme=naming_scheme)

    if size:
        radius, stats = binsearch_radius(circuit, subtarget_size=size,
                                         get_subtargets_for_radius=for_radius,
                                         lower_bound=1., upper_bound=6000.,
                                         tolerance=None,
                                         sample_frac=sample)
        return (radius,
                generate_subtargets(circuit, flatmap, radius=radius, size=None,
                                    target=target, sample=sample,
                                    naming_scheme=naming_scheme))

    radius = radius or 20

    if size:
        raise NotImplementedError("TODO")

    if naming_scheme:
        raise NotImplementedError("TODO")


def get_statistics(circuit, radius, sample_frac=None):
    """..."""
    subtargets = generate_subtargets(circuit, radius=radius, sample=sample_frac)
    subtarget_sizes = (subtargets.groupby("subtarget").agg("size")
                       / (1. if not sample_frac else sample_frac))
    return subtarget_sizes.agg(["count", "min", "median", "mad", "mean", "std", "max"])


def tolerate(observed_value, target_value, tolerance):
    """..."""
    LOG.debug("tolerate ? %s, %s, %s", observed_value, target_value, tolerance)

    if not tolerance:
        return  tolerate(observed_value, target_value, np.sqrt(observed_value))

    assert tolerance > 0
    delta = np.abs(target_value - observed_value)
    result = np.abs(delta) < tolerance

    LOG.debug("\t: %s", result)
    return result


def binsearch_radius(circuit, subtarget_size=30000, get_subtargets_for_radius=None,
                     lower_bound=None, upper_bound=None, tolerance=None,
                     sample_frac=None,
                     n_iter=0):
    """..."""
    lower_bound = lower_bound or 1.
    upper_bound = upper_bound or 6000

    mean_radius = (lower_bound + upper_bound) / 2.

    if not get_subtargets_for_radius:
        _, subtargets = generate_subtargets(circuit, radius=mean_radius,
                                            sample=sample_frac)
    else:
        _, subtargets = get_subtargets_for_radius(mean_radius)

    subtarget_sizes = (subtargets.groupby("subtarget").agg("size")
                       / (1. if not sample_frac else sample_frac))
    stats_mn = subtarget_sizes.agg(["count", "min", "median", "mad", "mean", "std", "max"])

    LOG.info("Find optimal mean radius iteration %s: \n\t mean-radius %s: %s, %s",
             n_iter, mean_radius, stats_mn["mean"], stats_mn["std"])

    if tolerate(stats_mn["mean"], subtarget_size, tolerance):
        return (mean_radius, stats_mn)

    if stats_mn["mean"] < subtarget_size:
        return binsearch_radius(circuit,
                                lower_bound=mean_radius, upper_bound=upper_bound,
                                tolerance=tolerance,
                                sample_frac=sample_frac)

    assert stats_mn["mean"] > subtarget_size

    return binsearch_radius(circuit,
                            lower_bound = lower_bound,
                            upper_bound = mean_radius,
                            tolerance = tolerance,
                            sample_frac = sample_frac,
                            n_iter = n_iter + 1)


def define_subtargets(config, sample_frac=None, format=None):
    """
    config: A `SubtargetConfig` defined above, or path to a JSON file from which it can be loaded.
    """
    format = format or "wide"
    assert format in ("wide", "long")

    def get(label, circuit, flatmap):
        """..."""
        LOG.info("GENERATE subtargets for circuit %s", label)
        _, _subtargets = generate_subtargets(circuit, flatmap,
                                             radius=config.target_radius,
                                             size=config.mean_target_size,
                                             target=config.target,
                                             sample=sample_frac)
        LOG.info("DONE %s subtargets for circuit %s", _subtargets.shape[0], label)
        subtargets = _subtargets.assign(circuit=label) if label else _subtargets
        return subtargets.rename(columns={"x": "flat_x", "y": "flat_y"})

    try:
        _=config.input_circuit.keys
    except AttributeError:
        subtargets =  get(None, config.input_circuit, config.input_flatmap)
        columns = ["gid", "flat_x", "flat_y"]
        index_vars = ["subtarget", "flat_x", "flat_y"]
    else:
        arguments = list(config.argue())
        subtargets = pd.concat([get(label=l, circuit=c, flatmap=f)
                                for l, c, f in arguments])
        columns = ["circuit", "gid", "flat_x", "flat_y"]
        index_vars = ["circuit", "subtarget", "flat_x", "flat_y"]

    if format == "long":
        return subtargets

    def enlist(group):
        row = pd.Series({"flat_x": np.mean(group["flat_x"]), "flat_y": np.mean(group["flat_y"])})
        row["gid"] = list(group["gid"].values)
        return row

    return (subtargets[columns]
            .groupby(index_vars).apply(enlist)["gid"]
            .rename("gids"))
