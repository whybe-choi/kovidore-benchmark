__version__ = "0.1.0"

from .main import main
from .evaluate import run_benchmark, ALL_TASKS

# Import evaluate module to register custom tasks with MTEB
from . import evaluate


# Register custom tasks with MTEB
def _register_kovidore_tasks():
    """Register KoVidore custom tasks with MTEB's TASKS_REGISTRY"""
    try:
        import importlib
        import inspect
        from . import evaluate
        from mteb.abstasks import AbsTask

        # `mteb.get_tasks` is the function; the registry lives in the module of the same name.
        registry = importlib.import_module("mteb.get_tasks")._TASKS_REGISTRY

        # Find all classes in evaluate module that inherit from AbsTask
        custom_tasks = []
        for name, obj in inspect.getmembers(evaluate, inspect.isclass):
            # Check if it's a subclass of AbsTask and has "KoVidore" in the name
            if issubclass(obj, AbsTask) and name.startswith("KoVidore") and hasattr(obj, "metadata"):
                custom_tasks.append(obj)

        # Register each discovered task, keeping MTEB's own definition where one exists (KoVidore v2)
        for task_cls in custom_tasks:
            registry.setdefault(task_cls.metadata.name, task_cls)

    except (ImportError, AttributeError):
        # mteb not available (or its registry moved), skip registration
        pass


# Register tasks when module is imported
_register_kovidore_tasks()

__all__ = ["main", "run_benchmark", "ALL_TASKS"]
