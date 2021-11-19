"""Utils to change data formats."""

import numpy as np
import pandas as pd

def widen_by_index(level, dataframe):
    """Widen a dataframe by an index level."""
    groups = dataframe.groupby(level)
    return pd.concat([g.droplevel(level) for _, g in groups], axis=1,
                     keys=[i for i, _ in groups], names=[level])


def stretch_by_column(name, dataframe):
    """..."""
    try:
        data = dataframe["name"] #expecring MultIndexed Column
    except KeyError:
        data = dataframe #dataframe columns contain the same quantity

    def stretch(idx, row):
        return pd.concat([row], axis=0, keys=[idx], names=data.index.names)
    return (pd.concat([stretch(idx, row) for idx, row in data.iterrows()])
            .rename(name))
