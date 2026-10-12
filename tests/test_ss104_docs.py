"""Keep estimated-forecast claims consistent with the saved M7 search status."""

from __future__ import annotations

import json
from pathlib import Path


def test_unconverged_m7_forecast_has_experimental_readiness_labels():
    root = Path(__file__).resolve().parents[1]
    checkpoint = json.loads(
        (root / "data/estimates/m1002_ss104_mode.json").read_text(encoding="utf-8")
    )
    assert checkpoint["status"] == "time_limit"
    assert checkpoint["diagnostics"]["success"] is False
    assert checkpoint["mcmc"]["runs"] == 0
    for name in ("README.md", "docs/ss104_port.md", "docs/ss104_forecast_comparison.md"):
        document = (root / name).read_text(encoding="utf-8")
        assert "EXPERIMENTAL" in document, name
        assert "Port correctness: MET" in document, name
        assert "Estimated-forecast readiness: NOT MET" in document, name
        assert "unconverged" in document and "data gaps" in document, name
