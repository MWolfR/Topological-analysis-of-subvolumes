"""General utilities."""

import importlib
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from randomization import Algorithm


def import_module_with_name(n):
    """Must be in the environment."""
    assert isinstance(n, str)
    return importlib.import_module(n)


def load_module_from_path(p):
    """Load a module from a path.
    """
    path = Path(p)

    if not path.exists:
        raise FileNotFoundError(p.as_posix())

    if  path.suffix != ".py":
        raise ValueError(f"Not a python file {path}!")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module) #

    return module


def get_module(from_object, with_function=None):
    """Get module from an object.
    Read the code to see what object can be resolved to a module.
    If `with_method`, look for the method in the module.
    """
    def iterate(functions):
        """..."""
        if isinstance(functions, str):
            return [functions]
        try:
            items = iter(functions)
        except TypeError:
            items = [functions]

        return items

    def check(module, has_function=None):
        """..."""
        if not has_functions:
            return module

        def get_method(function):
            """..."""
            try:
                method = getattr(module, function)
            except AttributeError:
                raise TypeError(f" {module} is missing required method {function}.")
            return method

        if isinstance(has_function, str):
            methods = get_method(has_function)

        methods = {f: get_method(f) for f in iterate(has_function)}

        return (module, methods)

    try:
        module = import_module_with_name(str(from_object))
    except ModuleNotFoundError:
        module = load_module_from_path(p=from_object)
        if not module:
            raise ModuleNotFoundError(f"that was specified by {from_object}")

    return check(module)


def collect_plugins_of_type(T, in_config):
    """..."""
    return {T(name, description) for name, description in items()}


class AlgorithmFromSource(Algorithm):
    """..."""
    @staticmethod
    def read_source(description):
        """"..."""
        return description["source"]

    @staticmethod
    def read_functions(description):
        """..."""
        return description["functions"]

    @staticmethod
    def load_args(functions):
        """..."""
        return [description.get("args", []) for description in functions]

    @staticmethod
    def load_kwargs(functions):
        """..."""
        return [description.get("kwargs", {}) for description in functions]

    def load_application(self, source, functions):
        """..."""
        self._functions = [f["name"] for f in functions]
        unique = set(self._functions)

        module, methods = get_module(from_object=source, with_function=unique)
        self._module = module
        self._methods = methods

        self._args = self.load_args(functions)
        self._kwargs = self.load_kwargs(functions)
        return

    def __init__(self, name, description):
        """..."""
        self._name = name
        source = self.read_source(description)
        functions = self.read_functions(description)
        self.load_application(source, functions)

    @property
    def name(self):
        """..."""
        return self._name

    def apply(self, adjacency, node_properties):
        """..."""
        def apply_method_indexed(i):
            """..."""
            function = self._functions[i]
            method = self._methods[function]
            args = self._args[i]
            kwargs = self._kwargs[i]
            label = self._label(function, args, kwargs)
            return (label, method(adjacency, node_properties, *args, **kwargs))

        N = len(self._functions)
        labels, matrices = zip(*[apply_method_indexed(i) for i in range(N)])

        return pd.Series(matrices, name="matrix", index=pd.Index(labels, names="algorithm"))


def get_algorithms(in_config):
    """..."""
    algorithms = in_config["algorithms"]
    return collect_plugins_of_type(AlgorithmFromSource, in_config=algorithms)
