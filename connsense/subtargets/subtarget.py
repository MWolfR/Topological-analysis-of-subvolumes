"""Targets defined using the circuit's flatmap.
"""
from collections.abc import Mapping
from pathlib import Path
from lazy import lazy

import numpy as np
import pandas as pd

from bluepy import Circuit
from bluepy.exceptions import BluePyError
from voxcell.voxel_data import VoxelData

from ..io import logging
LOG = logging.get_logger("define-subtargets")


class Subtargets:
    """Define and load subtargets in a circuit's flatmap."""

    @staticmethod
    def read_json(path, reader):
        """..."""
        try:
            path = Path(path)
        except TypeError:
            return path

        config = reader.read(path)
        return config

    def __init__(self, config, label=None, reader=None):
        """
        config : Mapping or path to a JSON file that contains one.
        label  : Label for the subtargets (sub)-section in the config.
        """

        config = self.read_json(config, reader)
        assert isinstance(config, Mapping)

        self._config = config
        self._config_label = label or "define_subtargets"

    @staticmethod
    def load_circuit(with_maybe_config):
        """..."""
        try:
            circuit = Circuit(with_maybe_config)
        except BluePyError:
            circuit = with_maybe_config
        return circuit

    @lazy
    def input_circuit(self):
        """..."""
        paths = self._config["paths"]

        input_circuit = paths["circuit"]
        try:
            configs = input_circuit.items
        except AttributeError:
            config = input_circuit
            return {"_": self.load_circuit(config)}
        return {label: self.load_circuit(config) for label, config in configs()}

    @lazy
    def input_atlas(self):
        """..."""
        try:
            circuits = self.input_circuit.items
        except AttributeError:
            return self.input_circuit.atlas
        return {label: circuit.atlas for label, circuit in circuits()}

    @lazy
    def default_flatmap(self):
        """..."""
        try:
            atlases = self.input_atlas.items
        except AttributeError:
            return self.input_atlas.load_data("flatmap")
        return {circuit: atlas.load_data("flatmap") for circuit, atlas in atlases()}

    @lazy
    def input_flatmap(self):
        """..."""
        paths = self._config["paths"]

        try:
            flatmap = paths["flatmap"]
        except KeyError:
            return self.default_flatmap

        try:
            flatmap_nrrd = Path(flatmap)
        except TypeError:
            pass
        else:
            flatmap = VoxelData.load_nrrd(flatmap_nrrd)
            return {c: flatmap for c in self.circuits.keys()}

        assert isinstance(flatmap, Mapping)

        def resolve_between(circuit, flatmap_nrrd):
            """..."""
            if flatmap_nrrd:
                return VoxelData.load_nrrd(flatmap_nrrd)
            circuit = self.input_circuit[label]
            return circuit.atlas.load_data("flatmap")

        return {c: self.resolve_between(circuit, flatmap.get(c, None))
                for c, circuit in self.input_circuit.items()}

    @lazy
    def mean_target_size(self):
        """..."""
        try:
            value = self.parameters["mean_target_size"]
        except KeyError:
            return None

        assert self.mean_target_radius is None,\
            "Cannot set both radius and mean target size, only one."

        return value

    @lazy
    def target_radius(self):
        """For example, if using hexagons,
        length of the side of the hexagon to tile with.
        """
        try:
            value = self.parameters["radius"]
        except KeyError:
            return None

        assert self.mean_target_size is None,\
            "Cannot set both radius and mean target size, only one."

        return value

    @lazy
    def parameters(self):
        """..."""
        return self._config.get("parameters", {}).get(self._label, {})

    @lazy
    def tolerance(self):
        """Relative tolerance, a non-zero positive number less than 1 that determines
        the origin, rotation, and radius of the triangular tiling to use for binning.
        """
        return self.parameters.get("tolerance", None)

    @lazy
    def target(self):
        """..."""
        return self.parameters.get("base_target", None)

    def argue(self):
        """..."""
        for label, circuit in self.input_circuit.items():
            yield (label, circuit, self.input_flatmap[label])

    @lazy
    def output(self):
        """..."""
        return self._config["paths"]["defined_columns"]

    @lazy
    def fmt_dataframe(self):
        """Specify whether
        wide : gids be in lists per row, or
        long : one gid per row
        """
        fmt = self.parameters.get("format", "wide")
        assert fmt in ("wide", "long")
        return fmt



    def generate(self, label, sample=None):
        """..."""
        from flatmap_utility.hexgrid import get_flatmap
        from flatmap_utility.tessellate import TriTille

        circuit = self.input_circuit[label]
        flatmap = self.input_flatmap[label]
        target = self.target
        radius = self.radius

        LOG.info("GET flatmap positions")
        flatmap = get_flatmap(circuit, target, sample)
        LOG.info("DONE %s flatmap positions", flatmap.shape[0])

        tritille = TriTille(radius)

        LOG.info("BIN into hexmap indices")
        hexmap = tritille.bin_hexagonally(flatmap, use_columns_row_indexing=False)
        LOG.info("DONE %s hexmap indices", hexmap.shape[0])

        grid = tritille.locate_grid(hexmap)
        LOG.info("ANNOTATE columns")
        annotation = tritille.annotate(grid, using_column_row=True)
        gids_by_gridpoint = hexmap.reset_index().set_index(["i", "j"])
        annotated = grid.assign(subtarget=annotation.loc[grid.index])
        LOG.info("DONE %s annotations", annotation.shape[0])

        return gids_by_gridpoint.join(annotated).reset_index().set_index("subtarget")


    def define(self, sample=None, format=None):
        """Define subtargets for all input (circuit, flatmap).
        """
        format = format or "wide"
        assert format in ("wide", "long")

        subtargets = pd.concat([self.generate(c) for c in self.input_circuit[c]])
        if format == "long":
            return Subtargets

        variables = ["circuit", "gid", "flat_x", "flat_y"]
        index_vars = ["circuit", "subtarget", "flat_x", "flat_y"]

        def enlist(group):
            """..."""
            return pd.Series({"flat_x": np.mean(group["flat_x"]),
                              "flat_y": np.mean(group["flat_y"]),
                              "gids": group["gid"].to_list()})

        return subtargets[variables].groupby(index_vars).apply("enlist").gids
