"""Rewire the connections of selected nodes."""

import numpy as np
from scipy import sparse


def shuffle(adjacency, node_properties=None, invariant_degree=None, in_format=None,
            **kwargs):
    """Shuffle the connections in an adjacency matrix.

    adjacency : A scipy.sparse.csr_matrix / csc_matrix / coo_matrix...
    node_properties : A pandas.DataFrame < node-property > value>
    invariant_degree : "IN" , "OUT", or None
    """

    N, M = adjacency.shape
    assert N == M, f"Expecting a square matrix not {(N, M)}"

    inv = invariant_degree.upper() if invariant_degree else "IN" #TODO: implement undirectional also
    assert inv in ("IN", "OUT")

    fmt = in_format.upper() if in_format else "CSR"
    assert fmt in ("CSC", "CSR")

    if inv == "IN":
        adjacency = adjacency.tocsc()
    else:
        adjacency = adjacency.tocsr()

    degrees = np.diff(adjacency.indptr)
    randomized_indices = np.hstack([np.random.choice(range(N), k, replace=False)
                                    for k in degrees])
    csc = sparse.csc_matrix((adjacency.data, randomized_indices, adjacency.indptr),
                            (N, N))
    return csc if fmt=="CSC" else csc.tocsr()
