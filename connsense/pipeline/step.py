"""A single step in a pipeline."""
from types import ModuleType
from importlib import import_module
from pathlib import Path

from ..plugins import load_module_from_path as load_module
from .runnable import Runnable


class Step(Runnable):
    """An individual step in the pipeline,
    should implement a run method.
    """
    def __init__(self, obj):
        """A pipeline step."""
        if callable(obj):
            runner = obj
        elif isinstance(obj, ModuleType):
            runner = obj
        else:
            try:
                with_name = str(obj)
                module = import_module(with_name)
            except ModuleNotFoundError:
                try:
                    from_path = Path(obj)
                    module = load_module(from_path)
                    assert module
                except TypeError:
                    runner = None
                else:
                    runner = module
            else:
                runner = module

            if not runner:
                raise ValueError(f"Couldn't find a runner in obj={obj}")

            try:
                run = module.run
            except AttributeError:
                raise TypeError("To define a step, a module needs to be runnable."
                               f"Found {module} with no `run` method.")
            else:
                #TODO: inspect run
                pass

            runner = module

        self._runner = runner

    def run(self, config, *args, **kwargs):
        """..."""
        return self._runner.run(config, *args, **kwargs)

    def check_state(self, pipeline):
        """TODO: Check where a pipeline is along the sequence of steps that define it."""
        return True
