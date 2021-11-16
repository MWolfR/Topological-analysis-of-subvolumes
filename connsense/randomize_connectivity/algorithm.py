"""Load a `randomization.Algorihtm from source code.`"""
from randomization import Algorithm
from ..plugins import import_module


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

def collect_plugins_of_type(T, in_config):
    """..."""
    return {T(name, description) for name, description in items()}


class SingleMethodAlgorithmFromSource(Algorithm):
    """Algorithms defined as such in the config:

    algorithms : {'erin': {'source': '/gpfs/bbp.cscs.ch/project/proj83/home/sood/analyses/manuscript/topological-analysis-subvolumes/topological-analysis-of-subvolumes/randomization/library/rewire.py',
                            'kwargs': {'invariant_degree': 'IN'},
                            'name': 'connections-rewired-controlling-in-degree'},
                 'erout': {'source': '/gpfs/bbp.cscs.ch/project/proj83/home/sood/analyses/manuscript/topological-analysis-subvolumes/topological-analysis-of-subvolumes/randomization/library/rewire.py',
                            'kwargs': {'invariant_degree': 'OUT'},
                            'name': 'connections-rewired-controlling-out-degree'},
                 'erdos_renyi': {'source': '/gpfs/bbp.cscs.ch/project/proj83/home/sood/analyses/manuscript/topological-analysis-subvolumes/topologists_connectome_analysis/randomization/ER_shuffle.py',
                                 'method': 'ER_shuffle',
                                 'kwargs': {},
                                 'name': 'erdos-renyi'}}
    """
    @staticmethod
    def read_method(description):
        """..."""
        return description.get("method", "shuffle")

    @staticmethod
    def read_source(description):
        """..."""
        return description["source"]

    @staticmethod
    def read_args(description):
        """..."""
        return description.get("args", [])

    @staticmethod
    def read_kwargs(description):
        """..."""
        return description.get("kwargs", {})

    def __init__(self, name, description):
        """..."""
        self._name = name
        self._source = self.read_source(description)
        self._args = self.read_args(description)
        self._kwargs = self.read_kwargs(description)
        self._method = self.read_method(description)
        self._shuffle = self.load(description)

    @property
    def name(self):
        """..."""
        return self._name

    def load(self, description):
        """..."""
        source = self.read_source(description)
        method = self.read_method(description)

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

        module, method = import_module(from_path=source, with_method=method)
        self._module = module
        return method


    def apply(self, adjacency, node_properties=None, log_info=None):
        """..."""
        try:
            matrix = adjacency.matrix
        except AttributeError:
            pass

        if node_properties is not None:
            assert node_properties.shape[0] == matrix.shape[0]

        return self._shuffle(matrix, node_properties,
                             *self._args, **self._kwargs)
