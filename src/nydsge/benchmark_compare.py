from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkResultRecord:
    machine: str
    backend: str
    device: str
    kernel: str
    horizon: int
    repeats: int
    dtype: str
    available: bool
    skipped: bool
    reason: str
    elapsed_seconds: float | None


@dataclass(frozen=True)
class BenchmarkCrossMachineReport:
    machine: str
    path: str
    schema_version: int
    command: dict[str, Any]
    platform: dict[str, Any]
    runtime_statuses: list[dict[str, Any]]
    results: list[BenchmarkResultRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "machine": self.machine,
            "path": self.path,
            "schema_version": self.schema_version,
            "command": self.command,
            "platform": self.platform,
            "runtime_statuses": self.runtime_statuses,
            "results": [record.__dict__ for record in self.results],
        }


@dataclass(frozen=True)
class BenchmarkCrossMachineRow:
    kernel: str
    backend: str
    device: str
    horizon: int
    repeats: int
    dtype: str
    machine: str
    elapsed_seconds: float | None
    skipped: bool
    available: bool
    reason: str


@dataclass(frozen=True)
class BenchmarkCrossMachineSummary:
    signature: dict[str, Any]
    rows: list[BenchmarkCrossMachineRow]
    created_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_utc": self.created_utc,
            "signature": self.signature,
            "rows": [
                {
                    "kernel": row.kernel,
                    "backend": row.backend,
                    "device": row.device,
                    "horizon": row.horizon,
                    "repeats": row.repeats,
                    "dtype": row.dtype,
                    "machine": row.machine,
                    "elapsed_seconds": row.elapsed_seconds,
                    "skipped": row.skipped,
                    "available": row.available,
                    "reason": row.reason,
                }
                for row in self.rows
            ],
        }


def benchmark_report_command_signature(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "kernel": _coerce_command_str(payload, "kernel"),
        "horizon": _coerce_command_int(payload, "horizon"),
        "periods": _coerce_command_int(payload, "periods"),
        "batches": _coerce_command_int(payload, "batches"),
        "draws": _coerce_command_int(payload, "draws"),
        "repeats": _coerce_command_int(payload, "repeats"),
        "dtype": _coerce_command_str(payload, "dtype"),
        "include_pseudo": _coerce_command_bool(payload, "include_pseudo"),
    }


def load_benchmark_report(path: Path | str) -> BenchmarkCrossMachineReport:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))

    if not isinstance(payload, dict):
        msg = "Benchmark report must be a JSON object."
        raise ValueError(msg)

    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, int):
        msg = "Benchmark report is missing schema_version."
        raise ValueError(msg)

    command = payload.get("command")
    if not isinstance(command, dict):
        msg = "Benchmark report is missing command block."
        raise ValueError(msg)

    platform = payload.get("platform")
    if not isinstance(platform, dict):
        msg = "Benchmark report is missing platform metadata."
        raise ValueError(msg)

    runtime_statuses = payload.get("runtime_statuses")
    if not isinstance(runtime_statuses, list):
        msg = "Benchmark report is missing runtime_statuses."
        raise ValueError(msg)

    result_payloads = payload.get("results")
    if not isinstance(result_payloads, list):
        msg = "Benchmark report is missing result rows."
        raise ValueError(msg)

    results = [
        _result_record_from_payload(machine=infer_machine_from_report_path(source), payload=result)
        for result in result_payloads
    ]

    return BenchmarkCrossMachineReport(
        machine=infer_machine_from_report_path(source),
        path=str(source),
        schema_version=schema_version,
        command={k: v for k, v in command.items()},
        platform={k: v for k, v in platform.items()},
        runtime_statuses=[item for item in runtime_statuses],
        results=results,
    )


def infer_machine_from_report_path(path: Path | str) -> str:
    source = Path(path)
    stem = source.stem
    if "_" in stem:
        first = stem.split("_", 1)[0]
        if first:
            return first
    return stem


def _result_record_from_payload(
    machine: str,
    payload: dict[str, Any],
) -> BenchmarkResultRecord:
    if not isinstance(payload, dict):
        msg = "Benchmark result entry must be an object."
        raise ValueError(msg)

    backend = payload.get("backend")
    if not isinstance(backend, str) or not backend:
        msg = "Benchmark result is missing backend."
        raise ValueError(msg)

    device = payload.get("device")
    if not isinstance(device, str) or not device:
        msg = "Benchmark result is missing device."
        raise ValueError(msg)

    return BenchmarkResultRecord(
        machine=machine,
        backend=backend,
        device=device,
        kernel=_coerce_non_empty_str(payload, "kernel"),
        horizon=_coerce_payload_int(payload, "horizon"),
        repeats=_coerce_payload_int(payload, "repeats"),
        dtype=_coerce_optional_str(payload, "dtype"),
        available=bool(payload.get("available", False)),
        skipped=bool(payload.get("skipped", False)),
        reason="" if payload.get("reason") is None else str(payload.get("reason")),
        elapsed_seconds=_coerce_optional_float(payload.get("elapsed_seconds")),
    )


def build_summary(
    reports: list[BenchmarkCrossMachineReport],
    *,
    kernel: str | None = None,
    enforce_signature: bool = True,
) -> BenchmarkCrossMachineSummary:
    if not reports:
        msg = "At least one report is required."
        raise ValueError(msg)

    signatures = [benchmark_report_command_signature(report.command) for report in reports]
    canonical = signatures[0]
    if enforce_signature:
        for report, signature in zip(reports, signatures, strict=True):
            if signature != canonical:
                msg = f"{report.machine} command signature differs from {reports[0].machine}."
                raise ValueError(msg)

    rows: list[BenchmarkCrossMachineRow] = []
    for report in reports:
        for result in report.results:
            if kernel is not None and result.kernel != kernel:
                continue
            rows.append(
                BenchmarkCrossMachineRow(
                    kernel=result.kernel,
                    backend=result.backend,
                    device=result.device,
                    horizon=result.horizon,
                    repeats=result.repeats,
                    dtype=result.dtype,
                    machine=report.machine,
                    elapsed_seconds=result.elapsed_seconds,
                    skipped=result.skipped,
                    available=result.available,
                    reason=result.reason,
                )
            )

    rows = sorted(
        rows,
        key=lambda item: (
            item.kernel,
            item.backend,
            item.device,
            item.horizon,
            item.repeats,
            item.dtype,
            item.machine,
        ),
    )

    return BenchmarkCrossMachineSummary(
        signature=canonical,
        rows=rows,
        created_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )


def speedup_reference_from_reference(
    rows: list[BenchmarkCrossMachineRow],
    *,
    baseline: tuple[str, str] | None = None,
) -> dict[tuple[str, str, str, int, int, str], list[tuple[str, float, float]]]:
    grouped: dict[
        tuple[str, str, str, int, int, str],
        list[BenchmarkCrossMachineRow],
    ] = {}
    for row in rows:
        key = (row.kernel, row.backend, row.device, row.horizon, row.repeats, row.dtype)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)

    table: dict[
        tuple[str, str, str, int, int, str],
        list[tuple[str, float, float]],
    ] = {}
    for key, entries in grouped.items():
        candidates = sorted(
            [
                entry
                for entry in entries
                if entry.elapsed_seconds is not None and not entry.skipped and entry.available
            ],
            key=lambda entry: entry.elapsed_seconds or float("inf"),
        )
        if not candidates:
            continue

        baseline_row = candidates[0]
        if baseline is not None:
            baseline_row = next(
                (entry for entry in candidates if (entry.machine, entry.backend) == baseline),
                baseline_row,
            )

        baseline_elapsed = baseline_row.elapsed_seconds
        if baseline_elapsed is None or baseline_elapsed == 0.0:
            continue

        table[key] = [
            (
                entry.machine,
                float(entry.elapsed_seconds),
                float(baseline_elapsed / float(entry.elapsed_seconds)),
            )
            for entry in candidates
            if entry.elapsed_seconds is not None
        ]

    return table


def _coerce_command_str(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if value is None:
        msg = f"Benchmark command is missing {field_name}."
        raise ValueError(msg)
    return str(value)


def _coerce_command_int(payload: dict[str, Any], field_name: str) -> int:
    if field_name not in payload:
        msg = f"Benchmark command is missing {field_name}."
        raise ValueError(msg)
    return _coerce_int(payload[field_name], field_name)


def _coerce_command_bool(payload: dict[str, Any], field_name: str) -> bool:
    return bool(payload.get(field_name, False))


def _coerce_non_empty_str(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        msg = f"Benchmark result entry is missing {field_name}."
        raise ValueError(msg)
    return value


def _coerce_optional_str(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if value is None:
        msg = f"Benchmark result entry is missing {field_name}."
        raise ValueError(msg)
    return str(value)


def _coerce_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        msg = f"Benchmark field {field_name} must be an integer, got {value!r}."
        raise ValueError(msg)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        msg = f"Benchmark field {field_name} must be an integer, got {value!r}."
        raise ValueError(msg)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as err:
            msg = f"Benchmark field {field_name} must be an integer, got {value!r}."
            raise ValueError(msg) from err
    msg = f"Benchmark field {field_name} must be an integer, got {value!r}."
    raise ValueError(msg)


def _coerce_payload_int(payload: dict[str, Any], field_name: str) -> int:
    if field_name not in payload:
        msg = f"Benchmark result entry is missing {field_name}."
        raise ValueError(msg)
    return _coerce_int(payload[field_name], field_name)


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"Benchmark result elapsed_seconds must be numeric, got {value!r}."
        raise ValueError(msg)
    return float(value)
