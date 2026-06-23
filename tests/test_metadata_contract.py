"""Guard the Julia/Python metadata contract against silent drift.

``tools/oracle_julia/python_metadata_ss10.json`` lets the Julia oracle emit
transform/source metadata that matches the Python candidate. Nothing else keeps
it in sync with the live model, so these tests fail if it goes stale. Regenerate
with ``uv run python scripts/export_python_metadata.py``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "export_python_metadata.py"
_CONTRACT_PATH = _REPO_ROOT / "tools" / "oracle_julia" / "python_metadata_ss10.json"


def _load_generator():
    spec = importlib.util.spec_from_file_location("export_python_metadata", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metadata_contract_matches_live_model() -> None:
    generator = _load_generator()
    live = generator.build_metadata_contract()
    committed = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert committed == live, (
        "tools/oracle_julia/python_metadata_ss10.json is out of sync with the live "
        "Model1002 metadata. Run `uv run python scripts/export_python_metadata.py`."
    )


def test_metadata_contract_file_is_canonical() -> None:
    generator = _load_generator()
    expected = generator.render_contract(generator.build_metadata_contract())
    actual = _CONTRACT_PATH.read_text(encoding="utf-8")
    assert actual == expected, (
        "tools/oracle_julia/python_metadata_ss10.json is not the canonical render. "
        "Run `uv run python scripts/export_python_metadata.py`."
    )
