"""Randomize a connection matrix by node properties.
"""

import numpy as np
import pandas as pd
import scipy as sp


def _resolve(matrix):
    """..."""
    try:
        return matrix.matrix
    except AttributeError:
        pass
    return matrix


def get_connections(matrix):
    """..."""
    if isinstance(matrix, pd.DataFrame):
        return matrix

    coomat = _resolve(matrix).tocoo(copy=False)
    return (pd.DataFrame({"pre": coomat.row, "post": coomat.col, "value": coomat.data})
            .set_index(["pre", "post"]))


def _check_available(properties, among):
    """..."""
    for p in properties:
        assert p in among, f"{p} is missing among available: {among}"
    return True


def append_columns_for(properties, to_nodes, selecting=None):
    """"..."""
    selecting = selecting or to_nodex.columns.values
    with_props = properties[selecting].loc[to_nodes.values].assign(node=to_nodes.values)
    columns = pd.MultiIndex.from_tuples([(to_nodes.names, p) for p in with_props.columns])
    return pd.DataFrame(with_props.values, columns=columns)


def shuffle(matrix, node_properties, direction=None,
            pre_family=None, post_family=None,
            return_as=None):
    """..."""
    assert not direction or direction in ("in", "out")

    _check_available(pre_family, among=node.columns)
    _check_available(post_family, among=node.columns)

    cnxns = get_connections(matrix)

    pre_with_properites = append_columns_for(node_properies, to_nodes=cnxns.pre,
                                             selecting=pre_family)

    post_with_properites = append_columns_for(node_properies, to_nodes=cnxns.post,
                                              selecting=post_family)

    with_properties = pd.concat([pre_with_properties, post_with_properties], axis=1)

    shuffled = with_properties.groupby(pretups + postups).apply(shuffle).droplevel)None)

    if return_as == pd.DataFrame:
        return shuffled

    raise NotImplementedError(f"return as {return_as}")
