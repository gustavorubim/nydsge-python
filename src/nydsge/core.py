from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from math import pow
from typing import Any

from nydsge.runtime import RuntimeConfig


class NotPortedError(NotImplementedError):
    """Raised when a public API exists but the Julia kernel is not translated yet."""


@dataclass(frozen=True)
class Setting:
    name: str
    value: Any
    description: str = ""
    print_flag: bool = True


@dataclass(frozen=True)
class Parameter:
    name: str
    value: float
    fixed: bool = False
    value_bounds: tuple[float, float] | None = None
    transform: str = "identity"
    scaling: str = "identity"
    prior: Any | None = None
    description: str = ""
    tex_label: str = ""
    category: str = ""
    regime: str = "baseline"

    @property
    def scaled_value(self) -> float:
        if self.scaling == "identity":
            return self.value
        if self.scaling == "discount_rate":
            return 1.0 / (1.0 + self.value / 100.0)
        if self.scaling == "gross_rate":
            return 1.0 + self.value / 100.0
        if self.scaling == "fomega":
            return 1.0 - pow(1.0 - self.value, 0.25)
        if self.scaling == "quarterly_spread":
            return pow(1.0 + self.value / 100.0, 0.25)
        if self.scaling == "percent":
            return self.value / 100.0
        msg = f"Unsupported parameter scaling: {self.scaling}"
        raise ValueError(msg)


@dataclass(frozen=True)
class Observable:
    name: str
    source_names: tuple[str, ...]
    description: str
    long_description: str
    reverse_transform: str = "identity"
    forward_transform: str = "identity"


@dataclass(frozen=True)
class PseudoObservable:
    name: str
    description: str
    reverse_transform: str = "identity"


@dataclass
class ModelIndexMaps:
    endogenous_states: OrderedDict[str, int] = field(default_factory=OrderedDict)
    exogenous_shocks: OrderedDict[str, int] = field(default_factory=OrderedDict)
    expected_shocks: OrderedDict[str, int] = field(default_factory=OrderedDict)
    equilibrium_conditions: OrderedDict[str, int] = field(default_factory=OrderedDict)
    endogenous_states_augmented: OrderedDict[str, int] = field(default_factory=OrderedDict)
    observables: OrderedDict[str, int] = field(default_factory=OrderedDict)
    pseudo_observables: OrderedDict[str, int] = field(default_factory=OrderedDict)


class DSGEModel:
    spec: str
    subspec: str

    def __init__(
        self,
        *,
        spec: str,
        subspec: str,
        runtime: RuntimeConfig | None = None,
        settings: dict[str, Any] | None = None,
        testing: bool = False,
    ) -> None:
        self.spec = spec
        self.subspec = subspec
        self.runtime = runtime or RuntimeConfig()
        self.testing = testing
        self.settings: OrderedDict[str, Setting] = OrderedDict()
        self.test_settings: OrderedDict[str, Setting] = OrderedDict()
        self.parameters: OrderedDict[str, Parameter] = OrderedDict()
        self.steady_state: OrderedDict[str, float] = OrderedDict()
        self.observable_mappings: OrderedDict[str, Observable] = OrderedDict()
        self.pseudo_observable_mappings: OrderedDict[str, PseudoObservable] = OrderedDict()
        self.indexes = ModelIndexMaps()
        for name, value in (settings or {}).items():
            self.set_setting(name, value)

    @property
    def observables(self) -> OrderedDict[str, int]:
        return self.indexes.observables

    @property
    def pseudo_observables(self) -> OrderedDict[str, int]:
        return self.indexes.pseudo_observables

    def description(self) -> str:
        return f"{self.spec}, {self.subspec}"

    def set_setting(self, name: str, value: Any, description: str = "") -> None:
        self.settings[name] = Setting(name=name, value=value, description=description)

    def get_setting(self, name: str, default: Any | None = None) -> Any:
        if self.testing and name in self.test_settings:
            return self.test_settings[name].value
        if name in self.settings:
            return self.settings[name].value
        return default

    def add_parameter(self, parameter: Parameter) -> None:
        self.parameters[parameter.name] = parameter

    def numeric_value(self, name: str) -> float:
        if name in self.parameters:
            return self.parameters[name].scaled_value
        if name in self.steady_state:
            return self.steady_state[name]
        msg = f"Unknown model parameter or steady state: {name}"
        raise KeyError(msg)

    def set_steady_state(self, name: str, value: float) -> None:
        self.steady_state[name] = float(value)

    def add_observable(self, observable: Observable) -> None:
        self.observable_mappings[observable.name] = observable

    def add_pseudo_observable(self, pseudo_observable: PseudoObservable) -> None:
        self.pseudo_observable_mappings[pseudo_observable.name] = pseudo_observable

    def build_one_based_index(self, names: list[str]) -> OrderedDict[str, int]:
        if len(names) != len(set(names)):
            seen: set[str] = set()
            duplicates: list[str] = []
            for name in names:
                if name in seen and name not in duplicates:
                    duplicates.append(name)
                seen.add(name)
            msg = f"Duplicate model index names: {', '.join(duplicates)}"
            raise ValueError(msg)
        return OrderedDict((name, idx) for idx, name in enumerate(names, start=1))

    def __getitem__(self, name: str) -> Parameter | float:
        if name in self.parameters:
            return self.parameters[name]
        if name in self.steady_state:
            return self.steady_state[name]
        msg = f"Unknown model parameter or steady state: {name}"
        raise KeyError(msg)
