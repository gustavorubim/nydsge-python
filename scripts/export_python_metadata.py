"""Regenerate the Python metadata contract consumed by the Julia oracle exporter.

The Julia exporter (``tools/oracle_julia/export_model1002.jl``) reads
``tools/oracle_julia/python_metadata_ss10.json`` so it can emit observable and
pseudo-observable transform/source metadata that matches the Python candidate
exactly. That file is the single source of truth shared across the language
boundary; this script rebuilds it from the live ``Model1002`` so it cannot
silently drift from what ``nydsge`` actually exports.

``tests/test_metadata_contract.py`` imports ``build_metadata_contract`` from this
module and asserts the committed JSON still matches the live model. Run this
script whenever the Python observable/pseudo-observable transform or source
metadata changes:

    uv run python scripts/export_python_metadata.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nydsge.models import Model1002

CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "oracle_julia" / "python_metadata_ss10.json"
)


def build_metadata_contract(
    *,
    subspec: str = "ss10",
    data_vintage: str = "181115",
    forecast_start: str = "2018-Q4",
) -> dict[str, Any]:
    """Build the transform/source metadata contract from the live model."""
    model = Model1002(
        subspec=subspec,
        settings={"data_vintage": data_vintage, "date_forecast_start": forecast_start},
    )
    observables = {
        name: {
            "sources": "|".join(model.observable_mappings[name].source_names),
            "forward": model.observable_mappings[name].forward_transform,
            "reverse": model.observable_mappings[name].reverse_transform,
        }
        for name in model.observables
    }
    pseudo_observables = {
        name: {
            "forward": model.pseudo_observable_mappings[name].forward_transform,
            "reverse": model.pseudo_observable_mappings[name].reverse_transform,
        }
        for name in model.pseudo_observables
    }
    return {"observables": observables, "pseudo_observables": pseudo_observables}


def render_contract(contract: dict[str, Any]) -> str:
    """Render the contract to the canonical JSON form written on disk."""
    return json.dumps(contract, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    contract = build_metadata_contract()
    CONTRACT_PATH.write_text(render_contract(contract), encoding="utf-8")
    print(CONTRACT_PATH)


if __name__ == "__main__":
    main()
