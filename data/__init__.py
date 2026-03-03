"""
Utilities for selecting dataset-specific dataloaders without touching train.py.
"""

from importlib import import_module
from typing import Any, Callable, Sequence

_DATASET_REGISTRY = {
    "brats": "data.brats",
    "bmad": "data.bmad",
    "resc": "data.resc",
}

AVAILABLE_DATASETS: Sequence[str] = tuple(_DATASET_REGISTRY.keys())


def _load_dataset_module(dataset_name: str):
    """Import and return the python module that backs ``dataset_name``."""
    try:
        module_path = _DATASET_REGISTRY[dataset_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: {', '.join(AVAILABLE_DATASETS)}"
        ) from exc

    return import_module(module_path)


def _dispatch(dataset_name: str, fn_name: str, **kwargs):
    module = _load_dataset_module(dataset_name)
    if not hasattr(module, fn_name):
        raise AttributeError(
            f"Dataset '{dataset_name}' does not define '{fn_name}'."
        )
    fn: Callable[..., Any] = getattr(module, fn_name)
    return fn(**kwargs)


def get_train_loader(dataset_name: str, **kwargs):
    """
    Build the training loader for ``dataset_name``.

    ``kwargs`` are forwarded to ``data.<dataset>.get_train_loader`` so each
    dataset can keep its own signature (e.g. different default subfolders).
    """
    return _dispatch(dataset_name, "get_train_loader", **kwargs)


def get_anomaly_loader(dataset_name: str, **kwargs):
    """Build the anomaly/validation loader for ``dataset_name``."""
    return _dispatch(dataset_name, "get_anomaly_loader", **kwargs)
