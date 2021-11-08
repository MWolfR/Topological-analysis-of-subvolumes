"""
An algorithm to shuffle with.
"""


class Algorithm:
    """..."""
    def __init__(self, name, source, args=None, kwargs-None):
        """Define an algorithm with its name, source code, and the args and kwargs
        needed to call it's `.shuffle` method
        """
        self._name = name
        self._shuffle = self.load_method(source)
        self._args = args or tuple()
        self._kwargs = kwargs or {}

    @property
    def name(self):
        """..."""
        return self._name

    def shuffle(self, adjacency, node_properties=None, log_info=None):
        """..."""
        try:
            matrix = adjacency.matrix
        except AttributeError:
            pass

        if node_properties is not None:
            assert node_properties.shape[0] == matrix.shape[0]

        LOG.info("%sShuffle a matrix of shape %s ",
                 "" if not log_info else log_info + ":\n\t",
                 matrix.shape)
        result = self._shuffle(matrix, node_properties,
                               *self._args, **self._kwargs)
        LOG.info("%sDONE shuffling a matrix of shape %s ",
                 "" if not log_info else log_info + ":\n\t",
                 matrix.shape)

        return result
