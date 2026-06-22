from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from nydsge.core import DSGEModel
from nydsge.runtime import RuntimeConfig

ModelFactory = Callable[..., DSGEModel]


@dataclass(frozen=True)
class ModelRegistryEntry:
    name: str
    factory: ModelFactory = field(repr=False, compare=False)
    description: str = ""
    default_subspec: str | None = None
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "default_subspec": self.default_subspec,
            "aliases": list(self.aliases),
        }


_REGISTRY: dict[str, ModelRegistryEntry] = {}
_ALIASES: dict[str, str] = {}


def normalize_model_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        msg = "Model name cannot be empty."
        raise ValueError(msg)
    return normalized


def register_model(
    name: str,
    factory: ModelFactory,
    *,
    description: str = "",
    default_subspec: str | None = None,
    aliases: tuple[str, ...] = (),
) -> ModelRegistryEntry:
    primary = normalize_model_name(name)
    normalized_names = [normalize_model_name(candidate) for candidate in (name, *aliases)]
    for normalized in normalized_names:
        if normalized in _ALIASES:
            msg = f"Model name or alias is already registered: {normalized}"
            raise ValueError(msg)

    entry = ModelRegistryEntry(
        name=name,
        factory=factory,
        description=description,
        default_subspec=default_subspec,
        aliases=aliases,
    )
    _REGISTRY[primary] = entry
    for normalized in normalized_names:
        _ALIASES[normalized] = primary
    return entry


def available_models() -> tuple[ModelRegistryEntry, ...]:
    return tuple(sorted(_REGISTRY.values(), key=lambda entry: entry.name))


def get_model_entry(name: str) -> ModelRegistryEntry:
    normalized = normalize_model_name(name)
    if normalized not in _ALIASES:
        available = ", ".join(entry.name for entry in available_models()) or "none"
        msg = f"Unknown model {name!r}; available models: {available}"
        raise KeyError(msg)
    return _REGISTRY[_ALIASES[normalized]]


def create_model(
    name: str,
    *,
    subspec: str | None = None,
    settings: Mapping[str, Any] | None = None,
    runtime: RuntimeConfig | None = None,
    testing: bool = False,
) -> DSGEModel:
    entry = get_model_entry(name)
    effective_subspec = entry.default_subspec if subspec is None else subspec
    kwargs: dict[str, Any] = {}
    if effective_subspec is not None:
        kwargs["subspec"] = effective_subspec
    if settings is not None:
        kwargs["settings"] = dict(settings)
    if runtime is not None:
        kwargs["runtime"] = runtime
    if testing:
        kwargs["testing"] = testing
    return entry.factory(**kwargs)
