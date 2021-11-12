"""The topological analysis pipeline, starting with a configuration.
"""
from importlib import import_module
from abc import ABC, abstractmethod, abstractclassmethod, abstractstaticmethod
from collections.abc import Mapping
from collections import OrderedDict, namedtuple
from pathlib import Path
from lazy import lazy

from ..io import read_config
from ..io.write_results import (read_toc_plus_payload, read_node_properties,
                                read_subtargets)
from ..io import logging

LOG = logging.get_logger("pipeline.")

from .step import Step
from .store import HDFStore

PipelineState = namedtuple("PipelineState", ["complete", "running", "queue"],
                           defaults=[None, None, None])


class TopologicalAnalysis:
    """..."""
    from connsense import define_subtargets
    from connsense import extract_neurons
    from connsense import extract_connectivity
    from connsense import randomize_connectivity
    from connsense import analyze_connectivity

    __steps__ = OrderedDict([("define-subtargets"      , Step(define_subtargets)),
                             ("extract-neurons"        , Step(extract_neurons)),
                             ("extract-connectivity"   , Step(extract_connectivity)),
                             ("randomize-connectivity" , Step(randomize_connectivity)),
                             ("analyze-connectivity"   , Step(analyze_connectivity))])

    @classmethod
    def sequence_of_steps(cls):
        """..."""
        return cls.__steps__.items()

    @classmethod
    def subset(complete, configured_steps):
        """configured : list of steps."""

        if configured_steps is None:
            return complete.sequence_of_steps()

        return OrderedDict([(step, to_take) for step, to_take in complete.sequence_of_steps
                            if step in configured_steps])

    @classmethod
    def read(cls, config, raw=False):
        """..."""
        try:
            path = Path(config)
        except TypeError:
            assert isinstance(config, Mapping)
            return config
        return read_config.read(path, raw=raw)

    @classmethod
    def read_steps(cls, config):
        """config : Mapping<key: value>."""
        try:
            configured = config["steps"]
        except KeyError:
            configured = list(cls.__steps__.keys())
        return configured

    def __init__(self, config, mode="inspect", dispatcher=None):
        """Read the pipeline steps to run from the config.
        """
        assert mode in ("inspect", "run"), mode

        self._config = self.read(config)

        config_raw = self.read(config, raw=True)
        steps = config_raw["paths"]["steps"]
        self._data = HDFStore(steps["root"], steps["groups"])

        self._mode = mode

        self.configured_steps =  self.read_steps(self._config)
        self.state = PipelineState(complete=OrderedDict(),
                                   running=None,
                                   queue=self.configured_steps)

    @property
    def data(self):
        """..."""
        return self._data

    def dispatch(self, step, *args, **kwargs):
        """..."""
        result = self.__steps__[step].run(self._config, *args, **kwargs)
        return result

    def get_h5group(self, step):
        """..."""
        return self._data_groups.get(step)

    def run(self, steps=None, *args, **kwargs):
        """Run the pipeline.
        """
        if self._mode == "inspect":
            raise RuntimeError("Cannot run a read-only pipeline."
                               " You can use read-only mode to inspect the data"
                               " that has already been computed.")

        LOG.warning("Dispatch from %s queue: %s",
                    len(self.state.queue), self.state.queue)
        if steps:
            self.state = PipelineState(complete=self.state.complete,
                                       running=self.state.running,
                                       queue=steps)
        while self.state.queue:
            step = self.state.queue.pop(0)

            LOG.warning("Dispatch pipeline step %s", step)

            self.running = step
            result = self.dispatch(step, *args, **kwargs)
            self.running = None
            self.state.complete[step] = result

            LOG.warning("DONE pipeline step %s: %s", step, result)

        LOG.warning("DONE running %s steps: ", len(self.state.complete))
        for i, (step, result) in enumerate(self.state.complete.items()):
            LOG.warning("\t(%s). %s: %s", i, step, result)

        return self.state
