from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nydsge.data import transform_data
from nydsge.models import Model1002
from nydsge.public_sources import (
    DSGEJL_PRE1991_ASACX10,
    FERNALD_TFP_URL,
    GSW_YIELD_CURVE_URL,
    PUBLIC_SOURCE_COLUMNS,
    SPF_INFLATION_URL,
    build_public_sources,
    load_public_sources_csv,
    merge_public_sources,
    parse_fernald_tfp,
    parse_gsw_sveny10,
    parse_spf_cpi10,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/public/nonfred_sources_20261006.csv"
FRED_CACHE = ROOT / "data/public/fred_levels_20261006.csv"

openpyxl = pytest.importorskip("openpyxl")


def _xlsx(frame: pd.DataFrame, *, sheet: str = "Sheet1", header: bool = True) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name=sheet, index=False, header=header)
    return buffer.getvalue()


def _spf_bytes() -> bytes:
    return _xlsx(
        pd.DataFrame(
            {
                "YEAR": [1991, 1991, 1992],
                "QUARTER": [3, 4, 1],
                "INFPGDP1YR": [3.0, 3.1, 3.2],
                "INFCPI1YR": [3.5, 3.4, 3.3],
                "INFCPI10YR": [np.nan, 4.0, 3.7],
            }
        )
    )


def _fernald_bytes() -> bytes:
    rows = [
        ["Note: All variables are percent change at an annual rate", None, None, None],
        ["date", "dY", "alpha", "dtfp"],
        ["1959:Q1", None, 0.33, None],
        ["1959:Q2", 5.0, 0.33, 4.6],
        ["1959:Q3", 1.0, 0.34, -1.2],
        ["", None, None, None],
        ["mean", 2.0, 0.33, 1.1],
    ]
    return _xlsx(pd.DataFrame(rows), sheet="quarterly", header=False)


def _gsw_bytes() -> bytes:
    text = (
        '"Note: not an official statistical release"\n\n'
        "Series,Compounding Convention,Mnemonic(s)\n"
        "Zero-coupon yield,Continuously Compounded,SVENYXX\n\n"
        "Date,BETA0,SVENY09,SVENY10\n"
        "1971-08-16,1,6.0,NA\n"
        "1971-08-17,1,6.0,6.0\n"
        "1971-09-30,1,6.0,6.4\n"
        "1971-10-01,1,6.0,7.0\n"
    )
    return text.encode()


def test_parse_spf_cpi10_splices_dsgejl_pre1991_history() -> None:
    out = parse_spf_cpi10(_spf_bytes())
    assert out["date"].iloc[0] == DSGEJL_PRE1991_ASACX10[0][0] == "1979-Q4"
    assert out.set_index("date").loc["1991-Q4", "ASACX10"] == pytest.approx(4.0)
    assert out.set_index("date").loc["1992-Q1", "ASACX10"] == pytest.approx(3.7)
    # pre-1991Q4 values come only from the DSGE.jl reference (semiannual, 24 values)
    early = out.loc[out["date"].str[:4].astype(int) < 1991]
    assert len(early) == len(DSGEJL_PRE1991_ASACX10) - 1  # 1991-Q1 is the 24th
    assert not out["date"].duplicated().any()
    without = parse_spf_cpi10(_spf_bytes(), include_pre1991=False)
    assert list(without["date"]) == ["1991-Q4", "1992-Q1"]


def test_parse_fernald_tfp_reads_dtfp_and_alpha() -> None:
    out = parse_fernald_tfp(_fernald_bytes())
    assert list(out["date"]) == ["1959-Q2", "1959-Q3"]
    np.testing.assert_allclose(out["TFPKQ"], [4.6, -1.2])
    np.testing.assert_allclose(out["TFPJQ"], [0.33, 0.34])


def test_parse_gsw_sveny10_quarterly_mean_of_daily() -> None:
    out = parse_gsw_sveny10(_gsw_bytes()).set_index("date")
    assert out.loc["1971-Q3", "FYCCZA"] == pytest.approx(6.2)
    assert out.loc["1971-Q4", "FYCCZA"] == pytest.approx(7.0)


def test_build_public_sources_uses_fetcher_and_merges() -> None:
    payloads = {
        SPF_INFLATION_URL: _spf_bytes(),
        FERNALD_TFP_URL: _fernald_bytes(),
        GSW_YIELD_CURVE_URL: _gsw_bytes(),
    }
    seen: list[str] = []

    def fetch(url: str) -> bytes:
        seen.append(url)
        return payloads[url]

    out = build_public_sources(fetch)
    assert set(seen) == set(payloads)
    assert list(out.columns) == ["date", *PUBLIC_SOURCE_COLUMNS]
    assert out["date"].iloc[0] == "1959-Q2"


def test_merge_public_sources_only_fills_gaps() -> None:
    levels = pd.DataFrame({"date": ["2000-Q1", "2000-Q2"], "FYCCZA": [np.nan, 9.9]})
    public = pd.DataFrame(
        {
            "date": ["2000-Q1", "2000-Q2"],
            "ASACX10": [2.5, 2.5],
            "FYCCZA": [6.5, 6.0],
            "TFPKQ": [1.0, 2.0],
            "TFPJQ": [0.3, 0.3],
        }
    )
    out = merge_public_sources(levels, public)
    np.testing.assert_allclose(out["FYCCZA"], [6.5, 9.9])
    np.testing.assert_allclose(out["ASACX10"], [2.5, 2.5])


def test_committed_cache_spans_and_matches_dsgejl_reference_inputs() -> None:
    cache = load_public_sources_csv(CACHE).set_index("date")
    first = {c: cache[c].first_valid_index() for c in PUBLIC_SOURCE_COLUMNS}
    assert first == {
        "ASACX10": "1979-Q4",
        "FYCCZA": "1971-Q3",
        "TFPKQ": "1959-Q1",
        "TFPJQ": "1959-Q1",
    }
    # Values from DSGE.jl test/reference/input_data/raw/dlx_160812.csv (Haver ASACX10/FYCCZA).
    reference = {
        "1971-Q3": (np.nan, 6.2371),
        "1985-Q4": (4.55, 9.9393),
        "1991-Q4": (4.00, 7.6687),
        "2000-Q1": (2.50, 6.5776),
        "2008-Q4": (2.50, 3.8926),
    }
    for quarter, (asacx10, fyccza) in reference.items():
        if np.isfinite(asacx10):
            assert cache.loc[quarter, "ASACX10"] == pytest.approx(asacx10, abs=5e-3)
        assert cache.loc[quarter, "FYCCZA"] == pytest.approx(fyccza, abs=1e-3)


def test_transforms_wire_public_sources_into_measurement_rows() -> None:
    model = Model1002(
        "ss10",
        settings={
            "date_presample_start": "1959-Q3",
            "date_forecast_start": "2026-Q3",
            "n_mon_anticipated_shocks": 0,
        },
    )
    levels = pd.read_csv(FRED_CACHE)
    levels = levels.loc[(levels["date"] >= "1959-Q2") & (levels["date"] <= "2026-Q2")]
    levels = merge_public_sources(levels.reset_index(drop=True), load_public_sources_csv(CACHE))
    obs = transform_data(model, levels).set_index("date")
    cache = load_public_sources_csv(CACHE).set_index("date")
    q = "2026-Q2"
    assert obs.loc[q, "obs_longinflation"] == pytest.approx((cache.loc[q, "ASACX10"] - 0.5) / 4)
    assert obs.loc[q, "obs_longrate"] == pytest.approx(cache.loc[q, "FYCCZA"] / 4)
    window = cache.loc["1959-Q2":"2026-Q2", "TFPKQ"]
    expected_tfp = (cache.loc[q, "TFPKQ"] - window.mean()) / (4 * (1 - cache.loc[q, "TFPJQ"]))
    assert obs.loc[q, "obs_tfp"] == pytest.approx(expected_tfp)
    assert "obs_nominalrate1" not in obs.columns
    assert np.isnan(obs.loc["1979-Q3", "obs_longinflation"])
    assert np.isfinite(obs.loc["1979-Q4", "obs_longinflation"])
