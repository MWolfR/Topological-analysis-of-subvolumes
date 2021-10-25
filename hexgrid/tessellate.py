"""Tilings of a 2D plane.
"""

import os
from collections.abc import Mapping
from pathlib import Path
import json
from lazy import lazy
import logging

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sbna

from voxcell.voxel_data import VoxelData
from bluepy import Cell, Circuit
from bluepy.exceptions import BluePyError


LOG = logging.getLogger("Tessellate-2D with triangles.")
LOG.setLevel(os.environ.get("LOGLEVEL", "INFO"))


def P(x, y):
    """..."""
    return np.array([x, y])


def Array2D(x, y):
    """..."""
    return np.array([x, y])


def Frame2D(**kwargs):
    """..."""
    return pd.DataFrame(kwargs)


def to_cartesian_dataframe(positions):
    """..."""
    return Frame2D(x=positions.rho * np.cos(positions.phi),
                   y=positions.rho * np.sin(positions.phi))


def to_cartesian_position(rho, phi):
    """..."""
    return np.array([rho * np.cos(phi), rho * np.sin(phi)])


def convert_cartesian(arg0, arg1=None):
    """..."""
    if isinstance(arg0, pd.DataFrame):
        assert arg1 is None
        return to_cartesian_dataframe(arg9)

    try:
        rho, phi = arg0
    except TypeError:
        assert arg1 is not None
        return convert_cartesian((arg0, arg1))

    assert arg1 is None

    return to_cartesian_position(rho, phi)


def to_polar_dataframe(positions):
    """..."""
    return Frame2D(rho=np.linalg.norm(positions, axis=1),
                   phi=np.arctan2(positions.y, positions.x))


def to_polar_position(x, y):
    """..."""
    return np.array([np.sqrt(x**2 + y**2),  np.arctan2(y, x)])


def convert_polar(arg0, arg1=None):
    """..."""
    if isinstance(arg0, pd.DataFrame):
        assert arg1 is None
        to_polar_dataframe(arg0)

    try:
        x, y = arg0
    except TypeError:
        assert arg1 is not None
        return convert_polar((arg0, arg1))

    assert arg1 is None

    return to_polar_position(x, y)


class Line:
    """Help to draw lines..."""
    def __init__(self, through, at_angle, graphic=None):
        """..."""
        self._origin = through
        self._angle = - (2 * np.pi - at_angle) if at_angle >= np.pi else at_angle
        self._slope = np.tan(at_angle)
        self._graphic = graphic

    def y(self, x):
        """..."""
        x0, y0 = self._origin
        return y0 + self._slope * (x - x0)

    def x(self, y):
        """..."""
        x0, y0 = self._origin
        return x0 + (y - y0) / self._slope

    @property
    def angle(self):
        """..."""
        return self._angle

    def plot(self, x0=None, x1=None, y0=None, y1=None,
             fmt=None, **kwargs):
        """..."""
        if self._graphic is None:
            raise TypeError("This Line was defined without a graphic.")

        def resolve(x, y):
            """..."""
            assert not (x is None and y is None),\
                "At least one of x or y should have been non-null."
            if x is not None:
                assert y is None
                y = self.y(x)
            elif y is not None:
                assert x is None
                x = self.x(y)

            return (x, y)

        x0, y0 = resolve(x0, y0)
        x1, y1 = resolve(x1, y1)

        print("plot line ", x0, y0, "to", x1, y1)
        _, axes = self._graphic
        axes.plot([x0, x1], [y0, y1], fmt, **kwargs)

        return self._graphic

    @classmethod
    def connecting(self, point, to_point, in_graphic=None):
        """..."""
        origin  = (point + to_point) / 2.
        dx, dy = to_point - point
        angle = np.arctan2(dy, dx)
        return Line(origin, angle, in_graphic)


def plot_segment(graphic, p0, p1, fmt=None, **kwargs):
    """..."""
    _, axes = graphic
    axes.plot([p0[0], p1[0]], [p0[1], p1[1]], fmt, **kwargs)
    return graphic


class TriTille:
    """A traingular tesselation.
    """
    def __init__(self, side, origin=None, angle=None):
        """
        side : length of the triangle side
        origin : (x, y) location of the origin of the tiling
        angle : angle of the tiling's x-axis w.r.t to the X-axis.
        """
        self._side = side
        self._origin = np.array([0., 0.]) if origin is None else origin
        self._angle  = 0. if angle is None else angle

        self._ratio = np.array([np.sqrt(3.), 1.])

    def translate(self, positions):
        """Translate positions with respect to the origin."""
        return positions - self._origin

    def untranslate(self, positions):
        """..."""
        return positions + self._origin

    def rotate(self, positions):
        """Rotate positions to align with this TriTille's x-axis.
        `positions` are expected to be relative to this origin.
        """
        polar = convert_polar(positions)
        if isinstance(positions, pd.DataFrame):
            return convert_cartesian(polar.assign(phi=(polar.phi - self._angle)))

        rho, phi = polar
        if np.isclose(rho, 0.):
            return np.array([0., 0.])
        return convert_cartesian(np.array([rho, phi - self._angle]))

    def unrotate(self, positions):
        """..."""
        polar = convert_polar(positions)

        if isinstance(positions, pd.DataFrame):
            return convert_cartesian(polar.assign(phi=(polar.phi + self._angle)))

        rho, phi = polar
        if np.isclose(rho, 0.):
            return np.array([0., 0.])
        return convert_cartesian(np.array([rho, phi + self._angle]))

    def relative(self, positions):
        """Position values relative to the origin and x-axis of the tiling.

        positions: (x, y) values
        """
        return self.rotate(self.translate(positions))

    def transform(self, xys):
        """Transform (x, y) positions to (u, v) axii.
        """
        relpos = self.relative(xys)

        u = relpos.x - relpos.y
        v = relpos.x + relpos.y
        return Frame2D(u=u, v=v) / self._ratio

    def reverse_transform(self, uvs):
        """Transform (u, v) positions to (x, y).
        """
        relpos = self._ratio * Frame2D(x=(uvs.u + uvs.v) / 2., y=-(uvs.u - uvs.v) / 2.)

        return self.untranslate(self.unrotate(relpos))


    def display(self, gridsize, origin=None, graphic=None,
                return_methods=False, hexgrid=False, **kwargs):
        """Display this triangular lattice.
        """
        gridsize = gridsize or (120, 120)
        try:
            width, height = gridsize
        except TypeError:
            width = height = gridsize

        window = np.array([width, height])

        ori = origin if origin is not None else self._origin

        if graphic is None:
            figure = plt.figure(figsize=(12, 12))
            axes = figure.add_subplot(111, aspect=1.)
        else:
            figure, axes = graphic
        graphic = (figure, axes)

        def draw_boundary(color="black", linestyle="-", linewidth=4, label="grid",
                          joinstyle="bevel", padding=None):
            """..."""


            x0, y0 = ori
            boundary = Rectangle(xy=ori, height=height, width=width,
                                 fill=False, color=color,
                                 linestyle=linestyle, linewidth=linewidth,
                                 joinstyle=joinstyle,
                                 label=label)
            axes.add_patch(boundary)

            try:
                xpad, ypad = padding or self._side
            except TypeError:
                xpad = ypad = padding or self._side

            axes.set_xlim([x0 - xpad, x0 + width + xpad])
            axes.set_ylim([y0 - ypad, y0 + height + ypad])
            return graphic

        def draw_line(through, at_angle, fmt):
            """but only the portion that can be seen through the grid window.
            """
            line = Line(through, at_angle)
            x0, y0 = ori; x1, y1 = ori + window

            line_x0 = x0; line_y0 = line.y(x0)
            if line_y0 < y0:
                if line._slope <= 0:
                    return None
                line_x0 = line.x(y0)
                if line_x0 >= x1:
                    return None
                line_y0 = y0

            if line_y0 > y1:
                if line._slope >= 0.:
                    return None
                line_x0 = line.x(y1)
                if line_x0 >= x1:
                    return None
                line_y0 = y1

            line_x1 = x1; line_y1 = line.y(x1)
            if line_y1 < y0:
                if line._slope >=0:
                    return None
                line_x1 = line.x(y0)
                if line_x1 < x0:
                    return None
                line_y1 = y0

            if line_y1 >= y1:
                if line._slope <= 0:
                    return None
                line_x1 = line.x(y1)
                if line_x1 < x0:
                    return None
                line_y1 = y1

            #print("through", through, "at angle ", at_angle)
            #print("\tsegment", (line_x0, line_y0), (line_x1, line_y1))

            axes.plot([line_x0, line_x1], [line_y0, line_y1], fmt)
            return graphic

        def draw_relxaxis(j, fmt=None):
            """x-axis in this TriTille's reference frame."""
            if hexgrid:
                return None

            in_tritille_coords = P(0, j * self._side)
            in_grid_coords = self.untranslate(self.unrotate(in_tritille_coords))
            return draw_line(in_grid_coords, self._angle, fmt="k--")

        def draw_relyaxis(i, fmt=None):
            """y-axis in this TriTille's reference frame."""
            in_tritille_coords = P(i * self._side, 0.) * self._ratio / 2.
            in_grid_coords = self.untranslate(self.unrotate(in_tritille_coords))
            return draw_line(in_grid_coords, self._angle + np.pi/2, fmt or "k-" if hexgrid else "k--")

        def draw_vaxis(i, fmt="k-"):
            """..."""
            in_tritille_coords = P(0, i * self._side)
            in_grid_coords = self.untranslate(self.unrotate(in_tritille_coords))
            return draw_line(through=in_grid_coords, at_angle=(self._angle + np.pi / 6.), fmt=fmt)

        def draw_uaxis(j, fmt="k-"):
            """..."""
            in_tritille_coords = P(j * self._side, 0.) * self._ratio
            in_grid_coords = self.untranslate(self.unrotate(in_tritille_coords))
            return draw_line(through=in_grid_coords, at_angle=(self._angle - np.pi / 6.), fmt=fmt)

        if return_methods:
            return {"draw_boundary": draw_boundary, "draw_line": draw_line,
                    "draw_relxaxis": draw_relxaxis, "draw_relyaxis": draw_relyaxis,
                    "draw_uaxis": draw_uaxis, "draw_vaxis": draw_vaxis}

        draw_boundary(**kwargs.get("draw_boundary", {}))

        hmax = height + self._origin[1] + 1
        imax = int(hmax / (self._ratio[0] * self._side / 2.))
        vi_outside_grid = [i for i in range(-imax, imax)
                          if not draw_vaxis(i, **kwargs.get("draw_vaxis", {}))]
        LOG.warning("Lines along the v-axis that didn't fit the window: %s / %s",
                    len(vi_outside_grid), (2 * imax))
        LOG.warning("\t: %s", vi_outside_grid)

        ryi_outside_grid = [i for i in range(-imax, imax)
                            if not draw_relyaxis(i, **kwargs.get("draw_relyaxis", {}))]
        LOG.warning("Lines along the TriTille's y-axis that didn't fit the window: %s / %s",
                    len(ryi_outside_grid), (2 * imax))
        LOG.warning("\t: %s", ryi_outside_grid)

        wmax = width + self._origin[0] + 1
        jmax = int(wmax / self._side)
        j_outside_grid = [j for j in range(-jmax, jmax)
                          if not draw_uaxis(j, **kwargs.get("draw_uaxis", {}))]

        LOG.warning("Lines along the u-axis that didn't fit the window: %s / %s",
                    len(j_outside_grid), 2 * jmax)
        LOG.warning("\t: %s", j_outside_grid)

        rxj_outside_grid = [j for j in range(-jmax, jmax)
                            if not draw_relxaxis(j, **kwargs.get("draw_relxaxis", {}))]
        LOG.warning("Lines along the TriTille's x-axis that didn't fit the window: %s / %s",
                    len(rxj_outside_grid), (2 * jmax))
        LOG.warning("\t: %s", rxj_outside_grid)
        return graphic
