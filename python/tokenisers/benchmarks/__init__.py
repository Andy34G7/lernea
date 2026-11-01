"""
Benchmarks package for Tokka-Bench, Intrinsic suite, and Colab small-LM training.

Submodules load on first use, so the tokka and intrinsic suites do not import torch.
"""

import importlib

__all__ = ["tokka", "intrinsic", "colab", "throughput", "flores", "common", "report"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module 'tokenisers.benchmarks' has no attribute {name!r}")
    module = importlib.import_module(f"tokenisers.benchmarks.{name}")
    globals()[name] = module
    return module
