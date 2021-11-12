"""General utilities."""

import importlib

def load_module(from_path):
    """Load a module from a path."""
    path = Path(from_path)

    assert path.exists

    assert path.suffix == ".py", f"Not a python file {path}!"

    spec = importlib.util.spec_from_file_location(path.stem, path)

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    if not hasattr(module, "shuffle"):
        raise DoesNotShuffle(module)

    return module
