"""A flatmap subtarget: A target in a circuit is a subset of cells that may be
of interest for whatever reason.  The targets can be defined by layer, mtype,
etype, or any other property of cells. While targets may be easily defined by
cell properties, these properties do not include any information about the
columnar strucuture of a brain tissue.

Flatmap subtargets allow us to subset cell targets over a flat plane.
Each such subtarget is a hexagon chosen to be of a desried size.
"""

import numpy as np
import pandas as pd

from bluepy import Cell, Circuit

from .tessellate import TriTille
from .flatmap_utility import supersampled_neuron_locations as flatten

XYZ = [Cell.X, Cell.Y, Cell.Z]


class _CircuitInterface:
    """Give a circuit interface to a dataframe of cell properties."""
    class _CellInterface:
        """Given a circuit cell like interface to a dataframe of cell properties."""
        def __init__(self, positions):
            """..."""
            self._positions = positions

        def get(self, group=None, properties=None):
            """..."""
            assert group in (None, "Mosaic")

            assert properties == XYZ

            return self._positions

    def __init__(self, positions):
        """..."""
        self._cells = self._CellInterface(positions)

    @property
    def cells(self):
        """..."""
        return self._cells


def get_circuit(in_data):
    """..."""
    if isinstance(in_data, Circuit):
        return in_data

    if isinstance(in_data, pd.DataFrame):
        return _CircuitInterface(in_data[XYZ])

    raise TypeError(f"Unhandled data type {type(in_data)}")


def get_positions(in_data):
    """..."""
    try:
        cells = in_data.cells
    except AttributeError:
        return in_data[XYZ]
    return cells.get(properties=XYZ)


def cache_evaluation_of(method, on_circuit, as_attribute=None):
    """..."""
    name = as_attribute or method.__qualname__
    try:
        value = getattr(on_circuit, name)
    except AttributeError:
        value = method(on_circuit)
        setattr(on_circuit, name, value)

    return value


def get_voxel_flatmap(in_data, or_these):
    """If not provided, may be a flatmap of voxels is defined on the circuit.
    """
    if or_these is not None:
        return or_these

    try:
        atlas = in_data.atlas
    except AttributeError:
        raise TypeError(f"Cannot get an atlas from data of type {type(in_data)}")
    return atlas.load_data("flatmap")


def get_voxel_orientations(in_data, or_these):
    """..."""
    if or_these:
        return or_these

    try:
        atlas = in_data.atlas
    except AttributeError:
        raise TypeError(f"Cannot get an atlas from data of type {type(in_data)}")

    return atlas.load_data("orientation")


def fmap_positions(in_data, over_flatmap_voxels=None, with_orientations=None,
                   to_subpixel_resolution=True, dropna=True):
    """Flatmap 3D positions in a data using flatmap voxel data."""
    if not to_subpixel_resolution:
        positions = get_positions(in_data)
        fpos = pd.DataFrame(over_flatmap_voxels.lookup(positions.values),
                            columns=["x", "y"], index=positions.index)

        return fpos[np.logical_and(fpos.x >= 0, fpos.y >= 0)]

    def flattening(circuit):
        """..."""
        C = circuit
        F = get_voxel_flatmap(in_data, over_flatmap_voxels)
        O = get_voxel_orientations(in_data, with_orientations)
        asxy = {"flat x": "x", "flat y": "y"}
        return flatten(C, F, O).rename(columns=asxy)

    flat_xy = cache_evaluation_of(flattening,
                                  on_circuit=in_data, as_attribute="fmap")
    return flat_xy.dropna() if dropna else flat_xy


def generate(circuit, flatmap_voxels, side, origin=None, angle=None):
    """Generate flatmap subtargets using a grid of hexagons of given side
    for given circuit, and flatmap voxel-data.
    """
    flatmap = (circuit.atlas.load_data("flatmap") if flatmap_voxels is None
               else flatmap_voxels)
    orientations = circuit.atlas.load_data("orientation")

    tritille = TriTille(side, origin, angle)

    flat_positions = fmap_positions(circuit, flatmap, orientations)
    return (tritille.distribute(flat_positions).reset_index()
            .merge(flat_positions.reset_index(), on="gid")
            .set_index("subtarget"))
