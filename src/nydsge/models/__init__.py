from __future__ import annotations

from nydsge.models.m1002 import Model1002
from nydsge.models.registry import (
    ModelRegistryEntry,
    available_models,
    create_model,
    get_model_entry,
    normalize_model_name,
    register_model,
)

register_model(
    "m1002",
    Model1002,
    description="New York Fed DSGE Model1002 representative-agent model.",
    default_subspec="ss10",
    aliases=("Model1002", "model1002"),
)

__all__ = [
    "Model1002",
    "ModelRegistryEntry",
    "available_models",
    "create_model",
    "get_model_entry",
    "normalize_model_name",
    "register_model",
]
