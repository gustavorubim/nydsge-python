from __future__ import annotations

import pytest

from nydsge.models import (
    Model1002,
    available_models,
    create_model,
    get_model_entry,
    register_model,
)
from nydsge.runtime import RuntimeConfig


def test_model_registry_exposes_model1002() -> None:
    entries = available_models()

    assert [entry.name for entry in entries] == ["m1002"]
    entry = get_model_entry("Model1002")
    assert entry.name == "m1002"
    assert entry.default_subspec == "ss10"
    assert entry.aliases == ("Model1002", "model1002")
    assert entry.to_dict() == {
        "name": "m1002",
        "description": "New York Fed DSGE Model1002 representative-agent model.",
        "default_subspec": "ss10",
        "aliases": ["Model1002", "model1002"],
    }


def test_create_model_uses_default_subspec_and_constructor_inputs() -> None:
    runtime = RuntimeConfig(backend="numpy", device="cpu", dtype="float64")

    model = create_model(
        "model1002",
        runtime=runtime,
        settings={"n_mon_anticipated_shocks": 0},
    )

    assert isinstance(model, Model1002)
    assert model.spec == "m1002"
    assert model.subspec == "ss10"
    assert model.runtime is runtime
    assert model.get_setting("n_mon_anticipated_shocks") == 0


def test_create_model_accepts_explicit_subspec() -> None:
    model = create_model("m1002", subspec="ss59")

    assert isinstance(model, Model1002)
    assert model.subspec == "ss59"


def test_model_registry_rejects_duplicate_names_and_aliases() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_model("m1002", Model1002)

    with pytest.raises(ValueError, match="already registered"):
        register_model("temporary-model", Model1002, aliases=("Model1002",))


def test_unknown_model_lists_available_models() -> None:
    with pytest.raises(KeyError, match="m1002"):
        get_model_entry("missing")
