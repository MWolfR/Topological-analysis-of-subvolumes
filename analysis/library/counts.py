"""Several kinds of counts in a circuit."""
import numpy as np
import pandas as pd

def get_edges(adjacency):
    edge_nodes = np.where(pd.DataFrame.sparse.from_spmatrix(adjacency))
    return pd.DataFrame({"source": edge_nodes[0], "target": edge_nodes[1]})

def count_edges_by_synapse_class(adjacency, node_properties,
                                 stats=["min", "max", "mean", "std", "median", "mad"],
                                 **kwargs):
    """..."""
    edges = get_edges(adjacency)

    source_sc = node_properties.synapse_class.iloc[edges.source.values]
    target_sc = node_properties.synapse_class.iloc[edges.target.values]

    edges_sc = pd.DataFrame({"source": source_sc, "target": target_sc})
    return edges_sc.value_counts()
