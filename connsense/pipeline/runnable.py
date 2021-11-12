"""Define what can be run."""

from abc import ABC, abstractmethod

class Runnable(ABC):
    """Specify what a subclass should implement to be able to run."""
    @abstractmethod
    def run(self, config, *args, **kwargs):
        """Run with a config"""
        raise NotImplementedError("TODO: Describe the required implementation.")
