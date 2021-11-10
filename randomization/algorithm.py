"""
An algorithm to shuffle with.
"""
from importlib import import_module

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

    def load_method(self, source):
        """..."""
        try:
            module = import_module(source)
        except ModuleNotFoundError:
            pass
        else:
            self._module = module
            return self._module.shuffle

        try:
            return source.shuffle
        except AttributeError:
            pass

        raise TypeError(f"Cannot infer a method to shuffle in {source}")

    def shuffle(self, adjacency, node_properties=None, log_info=None):
        """..."""
        try:
            matrix = adjacency.matrix
        except AttributeError:
            pass

        if node_properties is not None:
            assert node_properties.shape[0] == matrix.shape[0]

        return self._shuffle(matrix, node_properties,
                             *self._args, **self._kwargs)

    @staticmethod
    def from_config(description):
        """Define an algorithm using a description provided in a
        topology analysis config.
        """
        return Algorithm(description["name"], description["source"],
                         description["args"], description["kwargs"])
