from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from nydsge.core import NotPortedError, Observable, Parameter, PseudoObservable, Setting
from nydsge.runtime import RuntimeConfig

try:
    __version__ = version("nydsge")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "NotPortedError",
    "Observable",
    "Parameter",
    "PseudoObservable",
    "RuntimeConfig",
    "Setting",
]
