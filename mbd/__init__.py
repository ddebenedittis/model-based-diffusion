import importlib

__version__ = "0.1.0"
__all__ = ["utils", "envs", "planners", "ais"]


def __getattr__(name):
    """Import submodules on first access (PEP 562).

    `ais` needs only JAX, but `utils`/`envs`/`planners` pull in brax. Importing all
    four eagerly made `import mbd.ais` require the whole brax stack; downstream
    consumers that only use the AIS core (e.g. mrmbd) no longer pay for it.
    """
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
