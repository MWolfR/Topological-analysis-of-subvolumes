"""The topological analysis pipeline, starting with a configuration.
"""
from importlib import import_module
from abc import ABC, abstractmethod, abstractclassmethod, abstractstaticmethod
from collections.abc import Mapping
from collections import OrderedDict, namedtuple
from pathlib import Path

from .io import logging

LOG = logging.get_logger("pipeline.")


class Runnable(ABC):
    """A runnable must have a run method."""
    @abstractmethod
    def run(self, config, *args, **kwargs):
        """Give me a config, and I will go."""
        raise NotImplementedError


class Step(Runnable):
    """An individual step in the pipeline,
    should implement a run method.
    """
    def __init__(self, obj):
        """A pipeline step."""
        if callable(obj):
            runner = obj
        else:
            try:
                from_path = Path(obj)
            except TypeError:
                module = obj
            else:
                module = import_module(from_path)

            try:
                run = module.run
            except AttributeError:
                raise TypeError("To be define a step, a module needs to be runnable."
                               f"Found {module} with no `run` method.")
            else:
                #TODO: inspect run
                pass


            runner = module

        self._runner = runner

    def run(self, config, *args, **kwargs):
        """..."""
        self._runner.run(config, *args, **kwargs)

    def check_state(self, pipeline):
        """Check where a pipeline is along the sequence of steps that define it."""
        return True


PipelineState = namedtuple("PipelineState", ["complete", "running", "queue"],
                           defaults=[None, None, None])


class Disbatch:
    """Create sbatch scripts and dispatch them."""
    @classmethod
    def sbatch(cls, key, value):
        """..."""
        return f"#SBATCH --{key}={value}"

    @classmethod
    def allocate_sbatch(cls, slurmargs):
        """..."""
        return '\n'.join([cls.sbatch(*keyval)
                         for keyval in (("node", slurmargs.get("node", 1)),
                                        ("time", slurmargs.get("time", "24:00:00")),
                                        ("exclusive", None),
                                        ("constraint", slurmargs.get("constraint", "cpu")),
                                        ("mem", slurmargs.get("mem", 0)),
                                        ("partition", slurmargs.get("partition", "prod")),
                                        ("account", slurmargs.get("account", "proj83")))])


    def __init__(self, **slurmargs):
        """Initialize with slurm arguments.
        """
        preamble = "#!/bin/bash"
        self._slurm = '\n'.join([preamble, self.allocate_sbatch(slurmargs)])

    def stamp_label(self, l, sbatch_io):
        """Prepare a stamp for a label l."""
        assert sbactch_io in ("output", "error")

        destination = "out" if sbatch_io == "output" else "err"
        return self.sbatch(sbatch_io, f"topological-analysis-{l}.{destination}")

    def slurm(self, job):
        """..."""
        return '\n'.join([self._slurm,
                          self.stamp_label(job, "error"), self.stamp_label(job, "output")])

    def dispatch(self, label, runnable, config, *args, working_dir=None, venv=None,
                 **kwargs):
        """..."""
        try:
            working_dir = Path(working_dir)
        except TypeError:
            working_dir = Path("topological-anlaysis")

        if not working_dir.is_absolute():
            working_dir = Path.cwd() / working_dir

        working_dir.mkdir(exist_ok=True, parents=False)

        #add_symlink(working_dir,  )

        slurm = self.slurm(job=label)

        try:
            path_venv = Path(venv)
        except TypeError:
            pass
        else:
            slurm += f"\nsource {path_venv}"

        code = """
        if __name__=="main":

        """

        command = f"sbatch {label}.sbatch {config}"



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
    def read(cls, config):
        """..."""
        from .io import read_config
        try:
            path = Path(config)
        except TypeError:
            assert isinstance(config, Mapping)
            return config
        return read_config.read(path)

    @classmethod
    def read_steps(cls, config):
        """config : Mapping<key: value>."""
        try:
            configured = config["steps"]
        except KeyError:
            configured = list(cls.__steps__.keys())
        return configured

    def __init__(self, config, dispatcher=None):
        """Read the pipeline steps to run from the config.
        """
        self._config = self.read(config)
        self.configured_steps =  self.read_steps(self._config)
        self.state = PipelineState(complete=OrderedDict(),
                                   running=None,
                                   queue=self.configured_steps)
        self._dispatcher = dispatcher or Disbatch()

    def dispatch(self, step):
        """..."""
        self.running = step
        result = self._dispatcher.dispatch(step, self.__steps__[step],)
        return result

    def run(self, steps=None, *args, **kwargs):
        """Run the pipeline.
        """
        LOG.warning("Dispatch from %s queue: %s", len(self.state.queue), self.state.queue)
        if steps:
            self.state = PipelineState(complete=self.state.complete,
                                       running=self.state.running,
                                       queue=steps)
        while self.state.queue:
            step = self.state.queue.pop(0)
            LOG.warning("Dispatch pipeline step %s", step)

            result = self.__steps__[step].run(self._config, *args, **kwargs)
            LOG.warning("DONE pipeline step %s: %s", step, result)
            self.state.complete[step] = result

        LOG.warning("DONE running %s steps: %s", len(self.state.queue), self.state.queue)

        return self.state
