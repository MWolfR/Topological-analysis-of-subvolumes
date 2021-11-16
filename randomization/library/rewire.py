"""Rewire the connections of selected nodes."""

import numpy as np
from scipy import sparse


def shuffle(adjacency, node_properties=None, direction=None, in_format=None,
            **kwargs):
    """Shuffle the connections in an adjacency matrix.

    adjacency : A scipy.sparse.csr_matrix / csc_matrix / coo_matrix...
    node_properties : A pandas.DataFrame < node-property > value>
    direction : "IN" , "OUT", or None
    ~           if IN`then shuffle each node's in coming sources
    ~           if `OUT` then shuffle each node's out going targets
    ~           if `None` then shuffle both the source and target nodes of each edge.
    """

    N, M = adjacency.shape
    assert N == M, f"Expecting a square matrix not {(N, M)}"

    assert direction in ("IN", "OUT", None)

    if not direction:
        raise NotImplementedError("Only direction=IN or direction=OUT... TODO")

    invariant = "SOURCE" if direction == "OUT" else "TARGET"

    fmt = in_format.upper() if in_format else "CSR"
    assert fmt in ("CSC", "CSR")

    if invariant == "SOURCE":
        adjacency = adjacency.tocsc()
    else:
        adjacency = adjacency.tocsr()

    degrees = np.diff(adjacency.indptr)
    randomized_indices = np.hstack([np.random.choice(range(N), k, replace=False)
                                    for k in degrees])
    csc = sparse.csc_matrix((adjacency.data, randomized_indices, adjacency.indptr),
                            (N, N))
    return csc if fmt=="CSC" else csc.tocsr()
