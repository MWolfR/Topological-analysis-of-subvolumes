"""Utils to change data formats."""

import numpy as np
import pandas as pd

def widen_by_index(level, dataframe):
    """Widen a dataframe by an index level."""
    groups = dataframe.groupby(level)
    return pd.concat([g.droplevel(level) for _, g in groups], axis=1,
                     keys=[i for i, _ in groups], names=[level])
