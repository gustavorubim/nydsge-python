"""Free public replacements for the non-FRED Model1002 observables.

DSGE.jl's ``Model1002`` reads four observable blocks from Haver Analytics or
internal Board data.  Three have exact, free public equivalents:

=================  ==========================  ==========================================
DSGE.jl mnemonic   Public source               Column / construction
=================  ==========================  ==========================================
``ASACX10``        Philadelphia Fed SPF        ``INFCPI10YR`` (median 10y CPI expectation)
``FYCCZA``         Fed Board GSW yield curve   ``SVENY10`` (10y zero-coupon, cont. comp.),
                                               quarterly mean of daily observations
``TFPKQ/TFPJQ``    SF Fed (Fernald) TFP file   ``dtfp`` / ``alpha`` (quarterly sheet)
=================  ==========================  ==========================================

The DSGE.jl documentation (``docs/src/frbny_data.md``) names exactly these
spreadsheet series.  The fourth block, OIS-implied expected policy rates
(``ant1..ant6``), comes from internal Federal Reserve Board data and has no
free equivalent; see ``docs/c1_c2_fixes.md``.

Pre-1991Q4 ``ASACX10``: Haver's series starts in 1979Q4 with semiannual
Blue Chip/Livingston values that the Philadelphia Fed spreadsheet omits.  The
values below are copied verbatim from DSGE.jl's public reference input
``test/reference/input_data/raw/dlx_160812.csv`` (BSD-3-Clause); from 1991Q4
onward that file matches the Philadelphia Fed series to rounding (max abs
difference 0.005 pp).
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

SPF_INFLATION_URL = (
    "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
    "survey-of-professional-forecasters/historical-data/inflation.xlsx"
)
FERNALD_TFP_URL = "https://www.frbsf.org/wp-content/uploads/quarterly_tfp.xlsx"
GSW_YIELD_CURVE_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
PUBLIC_SOURCE_COLUMNS = ("ASACX10", "FYCCZA", "TFPKQ", "TFPJQ")

# DSGE.jl test/reference/input_data/raw/dlx_160812.csv, ASACX10 before 1991-Q4.
DSGEJL_PRE1991_ASACX10: tuple[tuple[str, float], ...] = (
    ("1979-Q4", 6.90),
    ("1980-Q2", 7.85),
    ("1980-Q4", 8.25),
    ("1981-Q2", 7.80),
    ("1981-Q4", 7.05),
    ("1982-Q2", 5.70),
    ("1982-Q4", 5.65),
    ("1983-Q1", 5.30),
    ("1983-Q4", 5.20),
    ("1984-Q1", 5.60),
    ("1984-Q4", 5.20),
    ("1985-Q1", 4.35),
    ("1985-Q4", 4.55),
    ("1986-Q1", 4.00),
    ("1986-Q4", 4.00),
    ("1987-Q1", 4.20),
    ("1987-Q4", 4.50),
    ("1988-Q1", 4.30),
    ("1988-Q4", 4.35),
    ("1989-Q1", 4.25),
    ("1989-Q4", 4.20),
    ("1990-Q1", 3.95),
    ("1990-Q4", 4.15),
    ("1991-Q1", 4.05),
)

Fetcher = Callable[[str], bytes]


def default_fetcher(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (nydsge public-sources loader)"})
    with urlopen(request, timeout=60.0) as response:
        return response.read()


def parse_spf_cpi10(raw: bytes, *, include_pre1991: bool = True) -> pd.DataFrame:
    """Return quarterly ``ASACX10`` (percent, annual) from the SPF ``inflation.xlsx`` file."""

    frame = pd.read_excel(io.BytesIO(raw))
    required = {"YEAR", "QUARTER", "INFCPI10YR"}
    if not required.issubset(frame.columns):
        msg = f"SPF inflation file is missing columns {sorted(required - set(frame.columns))}."
        raise ValueError(msg)
    frame = frame.dropna(subset=["YEAR", "QUARTER"])
    out = pd.DataFrame(
        {
            "date": [
                f"{int(year)}-Q{int(quarter)}"
                for year, quarter in zip(frame["YEAR"], frame["QUARTER"], strict=True)
            ],
            "ASACX10": pd.to_numeric(frame["INFCPI10YR"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["ASACX10"])
    if include_pre1991:
        first = out["date"].map(_quarter_index).min() if not out.empty else None
        early = pd.DataFrame(DSGEJL_PRE1991_ASACX10, columns=["date", "ASACX10"])
        if first is not None:
            early = early.loc[early["date"].map(_quarter_index) < first]
        out = pd.concat([early, out], ignore_index=True)
    return _sorted_unique(out)


def parse_fernald_tfp(raw: bytes) -> pd.DataFrame:
    """Return quarterly ``TFPKQ`` (dtfp, % annualized) and ``TFPJQ`` (alpha) from Fernald."""

    sheet = pd.read_excel(io.BytesIO(raw), sheet_name="quarterly", header=None)
    header_rows = np.flatnonzero(sheet.iloc[:, 0].astype(str).str.strip().eq("date"))
    if header_rows.size == 0:
        raise ValueError("Fernald TFP quarterly sheet has no 'date' header row.")
    header = int(header_rows[0])
    body = sheet.iloc[header + 1 :].copy()
    body.columns = [str(value).strip() for value in sheet.iloc[header]]
    for column in ("date", "dtfp", "alpha"):
        if column not in body.columns:
            raise ValueError(f"Fernald TFP quarterly sheet is missing column {column!r}.")
    labels = body["date"].astype(str).str.strip()
    body = body.loc[labels.str.fullmatch(r"\d{4}:Q[1-4]")]
    out = pd.DataFrame(
        {
            "date": body["date"].astype(str).str.replace(":", "-", regex=False).to_numpy(),
            "TFPKQ": pd.to_numeric(body["dtfp"], errors="coerce").to_numpy(),
            "TFPJQ": pd.to_numeric(body["alpha"], errors="coerce").to_numpy(),
        }
    )
    out = out.dropna(subset=["TFPKQ", "TFPJQ"])
    return _sorted_unique(out)


def parse_gsw_sveny10(raw: bytes) -> pd.DataFrame:
    """Return ``FYCCZA``: quarterly mean of the daily GSW 10-year zero-coupon yield."""

    text = raw.decode("utf-8", errors="replace").splitlines()
    try:
        start = next(i for i, line in enumerate(text) if line.startswith("Date,"))
    except StopIteration as err:
        raise ValueError("GSW yield-curve file has no 'Date,' header row.") from err
    frame = pd.read_csv(io.StringIO("\n".join(text[start:])), usecols=["Date", "SVENY10"])
    frame["SVENY10"] = pd.to_numeric(frame["SVENY10"], errors="coerce")
    frame = frame.dropna(subset=["SVENY10"])
    dates = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.loc[dates.notna()]
    labels = [f"{d.year}-Q{(d.month - 1) // 3 + 1}" for d in dates[dates.notna()]]
    frame = frame.assign(date=labels)
    out = frame.groupby("date", sort=False)["SVENY10"].mean().reset_index()
    return _sorted_unique(out.rename(columns={"SVENY10": "FYCCZA"}))


def build_public_sources(fetcher: Fetcher | None = None) -> pd.DataFrame:
    """Download the three public sources and merge them on quarter labels."""

    fetch = default_fetcher if fetcher is None else fetcher
    spf = parse_spf_cpi10(fetch(SPF_INFLATION_URL))
    tfp = parse_fernald_tfp(fetch(FERNALD_TFP_URL))
    gsw = parse_gsw_sveny10(fetch(GSW_YIELD_CURVE_URL))
    merged = spf.merge(gsw, on="date", how="outer").merge(tfp, on="date", how="outer")
    return _sorted_unique(merged[["date", *PUBLIC_SOURCE_COLUMNS]])


def load_public_sources_csv(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [c for c in ("date", *PUBLIC_SOURCE_COLUMNS) if c not in frame.columns]
    if missing:
        raise ValueError(f"Public-sources CSV {path} is missing columns {missing}.")
    return _sorted_unique(frame[["date", *PUBLIC_SOURCE_COLUMNS]])


def merge_public_sources(levels: pd.DataFrame, public: pd.DataFrame) -> pd.DataFrame:
    """Left-join public non-FRED sources onto a quarterly FRED levels frame.

    Existing non-missing values in ``levels`` are kept; public values only fill gaps.
    """

    if "date" not in levels.columns:
        raise ValueError("Levels frame must include a 'date' column.")
    out = levels.copy()
    indexed = public.set_index("date")
    for column in PUBLIC_SOURCE_COLUMNS:
        if column not in indexed.columns:
            continue
        values = out["date"].astype(str).map(indexed[column])
        if column in out.columns:
            out[column] = out[column].where(out[column].notna(), values)
        else:
            out[column] = values.astype(float)
    return out


def _quarter_index(label: str) -> int:
    year, quarter = str(label).split("-Q", 1)
    return int(year) * 4 + int(quarter[0])


def _sorted_unique(frame: pd.DataFrame) -> pd.DataFrame:
    if frame["date"].duplicated().any():
        dupes = sorted(set(frame.loc[frame["date"].duplicated(), "date"]))
        raise ValueError(f"Duplicate quarters in public source: {dupes[:5]}")
    order = frame["date"].map(_quarter_index).argsort(kind="stable")
    return frame.iloc[np.asarray(order)].reset_index(drop=True)
