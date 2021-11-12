"""Interface to the HDFStore where the pipeline stores its data."""
from lazy import lazy

from ..io.write_results import (read_subtargets,
                                read_node_properties,
                                read_toc_plus_payload)


class HDFStore:
    """Handle the pipeline's data.
    """
    def __init__(self, root, groups):
        """..."""
        self._root = root
        self._groups = groups

    def get_path(self, step):
        """..."""
        return (self._root, self._groups[step])

    @lazy
    def columns(self):
        """lists of gids for each subtarget column in the database."""
        try:
            return read_subtargets(self.get_path("define-subtargets"))
        except (KeyError, FileNotFoundError):
            return None

    @lazy
    def nodes(self):
        """Subtarget nodes that have been saved to the HDf store."""
        try:
            return read_node_properties(self.get_path("extract-neurons"))
        except (KeyError, FileNotFoundError):
            return None

    def _read_matrix_toc(self, step):
        """Only for the steps that store connectivity matrices."""
        return read_toc_plus_payload(self.get_path(step), step)

    @lazy
    def adjacency(self):
        """Original connectivity of subtargets that have been saved to the HDF store."""
        try:
            return self._read_matrix_toc("extract-connectivity")
        except (KeyError, FileNotFoundError):
            return None

    @lazy
    def randomizations(self):
        """Read randomizations."""
        try:
            return self._read_matrix_toc("randomize-connectivity")
        except (KeyError, FileNotFoundError):
            return None

    @lazy
    def analyses(self):
        """A TOC for analyses results available in the HDF store."""
        raise NotImplementedError

    @lazy
    def circuits(self):
        """Available circuits for which subtargets have been computed."""
        return self.columns.index.get_level_values("circuit").unique().to_list()

    def get_subtargets(self, circuit):
        """All subtargets defined for a circuit."""
        if self.columns is None:
            return None

        columns = self.columns.xs(circuit, level="circuit")
        return columns.index.get_level_values("subtarget").unique().to_list()

    def get_nodes(self, circuit, subtarget):
        """..."""
        if self.nodes is None:
            return None

        level = ["circuit", "subtarget"]
        query = [circuit, subtarget]
        return self.nodes.xs(query, level=level)

    def get_adjacency(self, circuit, subtarget, connectome):
        """..."""
        if self.adjacency is None:
            return None

        if connectome:
            level = ["circuit", "connectome", "subtarget"]
            query = [circuit, connectome, subtarget]
        else:
            level = ["circuit",  "subtarget"]
            query = [circuit, subtarget]

        adj = self.adjacency.xs(query, level=level)
        if adj.shape[0] == 1:
            return adj.iloc[0].matrix
        return adj

    def get_randomizations(self, circuit, subtarget, connectome, algorithms):
        """..."""
        if self.randomizations is None:
            return None

        if connectome:
            level = ["circuit", "connectome", "subtarget"]
            query = [circuit, connectome, subtarget]
        else:
            level = ["circuit",  "subtarget"]
            query = [circuit, subtarget]

        randomizations = self.randomizations.xs(query, level=level)

        if not algorithms:
            return randomizations

        return randmomizations.loc[algorithms]

    def get_data(self, circuit, subtarget, connectome=None, randomizations=None):
        """Get available data for a subtarget."""
        args = (circuit, subtarget)
        return OrderedDict([("nodes", self.get_nodes(*args)),
                            ("adjacency", self.get_adjacency(*args, connectome or "local")),
                            ("randomizations", self.get_randomizations(*args, connectome, randomizations))])
