"""
An analysis to run ...
"""
from pathlib import Path
import importlib

class DoesNotAnalyze(TypeError):
    pass


def import_module(from_path):
    """..."""
    path = Path(from_path)

    assert path.exists

    assert path.suffix == ".py", f"Not a python file {path}!"

    spec = importlib.util.spec_from_file_location(path.stem, path)

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    if not hasattr(module, "analyze"):
        raise DoesNotShuffle(module.__qualname__)

    return module


class Analysis:
    """..."""
    def __init__(self, name, source, args=None, kwargs=None):
        """Define an algorithm with its name, source code, and the args and kwargs
        needed to call it's `.shuffle` method
        """
        self._name = name
        self._analyze = self.load_method(source)
        self._args = args or tuple()
        self._kwargs = kwargs or {}

    @property
    def name(self):
        """..."""
        return self._name

    def load_method(self, source):
        """..."""
        try:
           analyze = source.analyze
        except AttributeError:
            pass
        else:
            self._module = source
            return self._module.analyze

        if callable(source):
            #TODO: inspect source
            return source

        self._module = import_module(from_path=source)
        return self._module.analyze

    def analyze(self, adjacency, node_properties=None, log_info=None):
        """
        adjacency : A scipy.sparse matrix
        node_properties : A pandas.DataFrame<node-property -> value>
        """
        """..."""
        def wake_up_lazy(matrix):
            try:
                matrix = matrix.matrix
            except AttributeError:
                pass
            return matrix

        matrix = wake_up_lazy(adjacency)
        N, M =  matrix.shape

        assert N == M, (N, M)

        assert node_properties is None or node_properties.shape[0] == N, (node_properties, N)

        return self._analyze(matrix, node_properties, *self.args, **self.kwargs)


    @staticmethod
    def from_config(description):
        """Define an algorithm using a description provided in a
        topology analysis config.
        """
        return Analysis(description["name"], description["source"],
                        description["args"], description["kwargs"])
