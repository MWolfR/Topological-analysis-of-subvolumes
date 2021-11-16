"""
An algorithm to shuffle with.
"""
from typing import Protocol, abstractmethod
from pathlib import Path
import importlib


class DoesNotShuffle(TypeError):
    pass


def import_module(from_path):
    """TODO: remove from here. should not be used where Algorithm is defined.
    """
    path = Path(from_path)

    assert path.exists

    assert path.suffix == ".py", f"Not a python file {path}!"

    spec = importlib.util.spec_from_file_location(path.stem, path)

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    if not hasattr(module, "shuffle"):
        raise DoesNotShuffle(module)

    return module


class Algorithm(Protocol):
    """Describe what may quack as a `randomization.Algorithm
    `"""
    @property
    @abstractmethod
    def name(self):
        """A name for the algorithm."""
        raise NotImplementedError

    @abstractmethod
    def apply(self, adjacency, node_properties=None, logger=None):
        """Apply the algorithm, i.e,
        Shuffle an adjacency matrix, using the given node properties
        """
        raise NotImplementedError


class ShuffleFromSource(Algorithm):
    """..."""
    def __init__(self, name, source, method=None, args=None, kwargs=None):
        """Define an algorithm with its name, source code,
        and the args and kwargs needed to call it's `.shuffle` method
        """
        self._name = name
        self._shuffle = self.load(source, method)
        self._args = args or tuple()
        self._kwargs = kwargs or {}

    @property
    def name(self):
        """..."""
        return self._name

    def load(self, source, method):
        """..."""
        method = method or "shuffle"
        try:
           run = getattr(source, method)
        except AttributeError:
            pass
        else:
            self._module = source
            return run

        if callable(source):
            #TODO: inspect source
            return source

        self._module = import_module(from_path=source)
        try:
            return getattr(self._module, method)
        except AttributeError:
            pass


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
        print("description from config: ", description)
        name = description["name"]
        source = description["source"]
        method = description.get("method", None)
        args = description.get("args", [])
        kwargs = description.get("kwargs", {})

        return Algorithm(name, source, method, args, kwargs)
