from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Annotated, Any, cast, get_args

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from nydsge import __version__
from nydsge.bench import (
    ParityKernel,
    benchmark_forecast_targets,
    benchmark_kalman_targets,
    compare_backend_parity_targets,
)
from nydsge.core import NotPortedError
from nydsge.data import (
    build_data_csv,
    build_data_csv_from_sources,
    date_labels_for_sample,
    df_to_matrix,
    download_current_fred_source_csv,
    download_fred_api_source_csv,
    load_data,
    quarter_labels_from_start,
)
from nydsge.estimate import (
    EstimateResult,
    MetropolisHastingsResult,
    estimation_mode_from_result,
    load_estimation_mode,
    load_sampler_result,
    save_estimation_mode,
    save_sampler_result,
)
from nydsge.estimate import (
    estimate as estimate_model,
)
from nydsge.financial_frictions import (
    d2g_domega2_fn,
    d2g_domega_dsigma_fn,
    d2gamma_domega2_fn,
    d2gamma_domega_dsigma_fn,
    dg_domega_fn,
    dg_dsigma_fn,
    dgamma_domega_fn,
    dgamma_dsigma_fn,
    g_fn,
    gamma_fn,
    mu_fn,
    nk_fn,
    omega_fn,
    zeta_bomega_fn,
    zeta_spb_fn,
    zeta_zomega_fn,
)
from nydsge.forecast import (
    ForecastOutput,
    MeansBands,
    build_zlb_conditional_observations,
    compute_meansbands,
    forecast_one,
    reverse_transform_forecast,
    reverse_transform_meansbands,
)
from nydsge.kalman import KalmanResult, kalman_log_likelihood, model_process_covariances
from nydsge.models import Model1002
from nydsge.purity import audit_runtime_purity
from nydsge.runtime import (
    BackendName,
    DeviceName,
    DTypeName,
    RuntimeConfig,
    UnsupportedRuntimeError,
    runtime_report,
)
from nydsge.solve import CanonicalSolveMethod, compute_system, solve_canonical
from nydsge.vv import (
    check_fixture_coverage,
    compare_fixture_dirs,
    load_canonical_fixture,
    required_fixture_arrays,
    resolve_tolerance_profile,
    save_canonical_fixture,
    save_fixture_manifest,
    save_forecast_fixture,
    save_kalman_fixture,
    save_meansbands_fixture,
    save_parameter_fixture,
    save_posterior_fixture,
    save_steady_state_fixture,
    save_system_fixture,
    save_transition_fixture,
)

app = typer.Typer(help="Native Python tools for the NY Fed DSGE model port.")
data_app = typer.Typer(help="Data loading and build commands.")
vv_app = typer.Typer(help="Verification and validation commands.")
console = Console()

FINANCIAL_FRICTIONS_INPUT_NAMES = ("z", "sigma", "spr")
FINANCIAL_FRICTIONS_CASES = (
    ("default", -2.42825276274453, 0.5, (1.0 + 1.7444 / 100.0) ** 0.25),
    ("lower_sigma", -2.42825276274453, 0.45, (1.0 + 1.7444 / 100.0) ** 0.25),
    ("wide_spread", -2.1, 0.6, (1.0 + 2.5 / 100.0) ** 0.25),
)
FINANCIAL_FRICTIONS_FUNCTION_NAMES = (
    "omega",
    "G",
    "Gamma",
    "dG_domega",
    "d2G_domega2",
    "dGamma_domega",
    "d2Gamma_domega2",
    "dG_dsigma",
    "d2G_domega_dsigma",
    "dGamma_dsigma",
    "d2Gamma_domega_dsigma",
    "mu",
    "nk",
    "zeta_bomega",
    "zeta_zomega",
    "zeta_spb",
)


@app.command()
def doctor(
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="Optional backend to resolve: auto, numpy, torch, or jax."),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Optional device to resolve: auto, cpu, cuda, or mps."),
    ] = None,
    dtype: Annotated[
        str | None,
        typer.Option("--dtype", help="Optional dtype to resolve: float64 or float32."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Report native backend/device availability."""
    report = runtime_report()
    check_requested = backend is not None or device is not None or dtype is not None
    requested_payload: dict[str, Any] | None = None
    requested_exit_code = 0
    if check_requested:
        requested_backend = _parse_backend("auto" if backend is None else backend)
        requested_device = _parse_device("auto" if device is None else device)
        requested_dtype = _parse_dtype("float64" if dtype is None else dtype)
        try:
            resolved = RuntimeConfig(
                backend=requested_backend,
                device=requested_device,
                dtype=requested_dtype,
            ).resolve()
            requested_payload = {
                "requested": {
                    "backend": requested_backend,
                    "device": requested_device,
                    "dtype": requested_dtype,
                },
                "available": True,
                "resolved": resolved.__dict__,
                "reason": resolved.reason,
            }
        except UnsupportedRuntimeError as err:
            requested_exit_code = 1
            requested_payload = {
                "requested": {
                    "backend": requested_backend,
                    "device": requested_device,
                    "dtype": requested_dtype,
                },
                "available": False,
                "resolved": None,
                "reason": str(err),
            }
    platform_exit_code = (
        1 if any(status.backend == "platform" and not status.available for status in report) else 0
    )
    exit_code = max(platform_exit_code, requested_exit_code)
    if json_output:
        if check_requested:
            typer.echo(
                json.dumps(
                    {
                        "report": [status.__dict__ for status in report],
                        "requested_runtime": requested_payload,
                    },
                    indent=2,
                )
            )
        else:
            typer.echo(json.dumps([status.__dict__ for status in report], indent=2))
        if exit_code:
            raise typer.Exit(code=exit_code)
        return
    table = Table(title=f"nydsge {__version__} runtime doctor")
    table.add_column("Backend")
    table.add_column("Device")
    table.add_column("Available")
    table.add_column("Reason")
    for status in report:
        table.add_row(
            status.backend,
            status.device,
            "yes" if status.available else "no",
            status.reason,
        )
    console.print(table)
    if requested_payload is not None:
        request_table = Table(title="Requested runtime")
        request_table.add_column("Metric")
        request_table.add_column("Value")
        request_table.add_row("backend", str(requested_payload["requested"]["backend"]))
        request_table.add_row("device", str(requested_payload["requested"]["device"]))
        request_table.add_row("dtype", str(requested_payload["requested"]["dtype"]))
        request_table.add_row("available", "yes" if requested_payload["available"] else "no")
        request_table.add_row("reason", str(requested_payload["reason"]))
        if requested_payload["resolved"] is not None:
            resolved_payload = cast(dict[str, object], requested_payload["resolved"])
            request_table.add_row("resolved_backend", str(resolved_payload["backend"]))
            request_table.add_row("resolved_device", str(resolved_payload["device"]))
        console.print(request_table)
    if exit_code:
        raise typer.Exit(code=exit_code)


@data_app.command("build")
def data_build(
    input_path: Annotated[
        Path,
        typer.Option("--input", help="Raw level CSV path to transform into observables."),
    ],
    output_path: Annotated[
        Path,
        typer.Option("--output", help="Output observable CSV path."),
    ],
    population_forecast_path: Annotated[
        Path | None,
        typer.Option(
            "--population-forecast",
            help="Optional population forecast CSV used to extend HP filtering.",
        ),
    ] = None,
    subspec: Annotated[str, typer.Option(help="Model1002 subspec to build.")] = "ss10",
    hpfilter_population: Annotated[
        bool,
        typer.Option(
            "--hpfilter-population/--no-hpfilter-population",
            help="Use HP-filtered population levels for per-capita observables.",
        ),
    ] = True,
    population_hpfilter_lambda: Annotated[
        float,
        typer.Option(
            "--population-hpfilter-lambda",
            help="HP filter smoothing parameter for quarterly population levels.",
        ),
    ] = 1600.0,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Build model observable data from a local raw level CSV."""
    model = Model1002(
        subspec=subspec,
        settings={
            "hpfilter_population": hpfilter_population,
            "population_hpfilter_lambda": population_hpfilter_lambda,
        },
    )
    try:
        transformed = build_data_csv(
            model,
            input_path=input_path,
            output_path=output_path,
            population_forecast_path=population_forecast_path,
        )
    except (FileNotFoundError, KeyError, ValueError) as err:
        raise _not_ported_exit(str(err)) from err
    payload = {
        "input": str(input_path),
        "output": str(output_path),
        "population_forecast": (
            None if population_forecast_path is None else str(population_forecast_path)
        ),
        "subspec": subspec,
        "hpfilter_population": hpfilter_population,
        "population_hpfilter_lambda": population_hpfilter_lambda,
        "rows": int(transformed.shape[0]),
        "columns": int(transformed.shape[1]),
        "observables": len(model.observables),
        "first_date": (
            None
            if "date" not in transformed or transformed.empty
            else str(transformed["date"].iloc[0])
        ),
        "last_date": (
            None
            if "date" not in transformed or transformed.empty
            else str(transformed["date"].iloc[-1])
        ),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    table = Table(title=f"{model.description()} data build")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in payload.items():
        table.add_row(key, str(value))
    console.print(table)


@data_app.command("fetch-fred")
def data_fetch_fred(
    output_path: Annotated[
        Path,
        typer.Option("--output", help="Output FRED source CSV path."),
    ],
    start_date: Annotated[
        str | None,
        typer.Option("--start-date", help="First required source quarter, e.g. 1959-Q3."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option("--end-date", help="Last required source quarter, e.g. 2018-Q3."),
    ] = None,
    aggregation: Annotated[
        str,
        typer.Option("--aggregation", help="Quarter aggregation for higher-frequency FRED series."),
    ] = "mean",
    subspec: Annotated[str, typer.Option(help="Model1002 subspec to inspect.")] = "ss10",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Fetch current public FRED graph CSV data for model-declared FRED series."""
    model = Model1002(subspec=subspec)
    try:
        levels = download_current_fred_source_csv(
            model,
            output_path=output_path,
            start_date=start_date,
            end_date=end_date,
            aggregation=aggregation,
        )
    except (OSError, ValueError) as err:
        raise _not_ported_exit(str(err)) from err
    payload = {
        "output": str(output_path),
        "subspec": subspec,
        "start_date": start_date,
        "end_date": end_date,
        "aggregation": aggregation,
        "rows": int(levels.shape[0]),
        "columns": int(levels.shape[1]),
        "fred_series": max(0, int(levels.shape[1]) - 1),
        "first_date": (
            None if "date" not in levels or levels.empty else str(levels["date"].iloc[0])
        ),
        "last_date": (
            None if "date" not in levels or levels.empty else str(levels["date"].iloc[-1])
        ),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    table = Table(title=f"{model.description()} FRED source fetch")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in payload.items():
        table.add_row(key, str(value))
    console.print(table)


@data_app.command("fetch-fred-api")
def data_fetch_fred_api(
    output_path: Annotated[
        Path,
        typer.Option("--output", help="Output FRED source CSV path."),
    ],
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="FRED API key; defaults to FRED_API_KEY."),
    ] = None,
    realtime_start: Annotated[
        str | None,
        typer.Option(
            "--realtime-start",
            help="FRED realtime_start date, e.g. 2018-11-15 or 181115.",
        ),
    ] = None,
    realtime_end: Annotated[
        str | None,
        typer.Option(
            "--realtime-end",
            help="FRED realtime_end date, e.g. 2018-11-15 or 181115.",
        ),
    ] = None,
    start_date: Annotated[
        str | None,
        typer.Option("--start-date", help="First required source quarter, e.g. 1959-Q3."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option("--end-date", help="Last required source quarter, e.g. 2018-Q3."),
    ] = None,
    aggregation: Annotated[
        str,
        typer.Option("--aggregation", help="Quarter aggregation for higher-frequency FRED series."),
    ] = "mean",
    subspec: Annotated[str, typer.Option(help="Model1002 subspec to inspect.")] = "ss10",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Fetch FRED observations API data for model-declared FRED series."""
    model = Model1002(subspec=subspec)
    try:
        levels = download_fred_api_source_csv(
            model,
            output_path=output_path,
            api_key=api_key,
            realtime_start=realtime_start,
            realtime_end=realtime_end,
            start_date=start_date,
            end_date=end_date,
            aggregation=aggregation,
        )
    except (OSError, ValueError) as err:
        raise _not_ported_exit(str(err)) from err
    payload = {
        "output": str(output_path),
        "subspec": subspec,
        "realtime_start": realtime_start,
        "realtime_end": realtime_end,
        "start_date": start_date,
        "end_date": end_date,
        "aggregation": aggregation,
        "rows": int(levels.shape[0]),
        "columns": int(levels.shape[1]),
        "fred_series": max(0, int(levels.shape[1]) - 1),
        "first_date": (
            None if "date" not in levels or levels.empty else str(levels["date"].iloc[0])
        ),
        "last_date": (
            None if "date" not in levels or levels.empty else str(levels["date"].iloc[-1])
        ),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    table = Table(title=f"{model.description()} FRED API source fetch")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in payload.items():
        table.add_row(key, str(value))
    console.print(table)


@data_app.command("build-sources")
def data_build_sources(
    source_root: Annotated[
        Path,
        typer.Option("--source-root", help="Directory containing raw source CSV files."),
    ],
    output_path: Annotated[
        Path,
        typer.Option("--output", help="Output observable CSV path."),
    ],
    vintage: Annotated[
        str | None,
        typer.Option("--vintage", help="Data vintage suffix for source files."),
    ] = None,
    start_date: Annotated[
        str | None,
        typer.Option("--start-date", help="First required source quarter, e.g. 1959-Q3."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option("--end-date", help="Last required source quarter, e.g. 2018-Q3."),
    ] = None,
    population_forecast_path: Annotated[
        Path | None,
        typer.Option(
            "--population-forecast",
            help="Optional population forecast CSV used to extend HP filtering.",
        ),
    ] = None,
    subspec: Annotated[str, typer.Option(help="Model1002 subspec to build.")] = "ss10",
    hpfilter_population: Annotated[
        bool,
        typer.Option(
            "--hpfilter-population/--no-hpfilter-population",
            help="Use HP-filtered population levels for per-capita observables.",
        ),
    ] = True,
    population_hpfilter_lambda: Annotated[
        float,
        typer.Option(
            "--population-hpfilter-lambda",
            help="HP filter smoothing parameter for quarterly population levels.",
        ),
    ] = 1600.0,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Build observable data from local raw source files declared by the model."""
    model = Model1002(
        subspec=subspec,
        settings={
            "hpfilter_population": hpfilter_population,
            "population_hpfilter_lambda": population_hpfilter_lambda,
        },
    )
    try:
        transformed = build_data_csv_from_sources(
            model,
            source_root=source_root,
            output_path=output_path,
            vintage=vintage,
            start_date=start_date,
            end_date=end_date,
            population_forecast_path=population_forecast_path,
        )
    except (FileNotFoundError, KeyError, ValueError) as err:
        raise _not_ported_exit(str(err)) from err
    payload = {
        "source_root": str(source_root),
        "output": str(output_path),
        "vintage": str(vintage if vintage is not None else model.get_setting("data_vintage")),
        "start_date": start_date,
        "end_date": end_date,
        "population_forecast": (
            None if population_forecast_path is None else str(population_forecast_path)
        ),
        "subspec": subspec,
        "hpfilter_population": hpfilter_population,
        "population_hpfilter_lambda": population_hpfilter_lambda,
        "rows": int(transformed.shape[0]),
        "columns": int(transformed.shape[1]),
        "observables": len(model.observables),
        "first_date": (
            None
            if "date" not in transformed or transformed.empty
            else str(transformed["date"].iloc[0])
        ),
        "last_date": (
            None
            if "date" not in transformed or transformed.empty
            else str(transformed["date"].iloc[-1])
        ),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    table = Table(title=f"{model.description()} source data build")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in payload.items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def solve(
    subspec: str = "ss10",
    backend: str = "auto",
    device: str = "auto",
) -> None:
    """Construct Model1002 and run the state-space solve once translated."""
    model = Model1002(
        subspec=subspec,
        runtime=RuntimeConfig(backend=_parse_backend(backend), device=_parse_device(device)),
    )
    try:
        system = compute_system(model)
    except NotPortedError as err:
        raise _not_ported_exit(str(err)) from err
    table = Table(title=f"{model.description()} solved system")
    table.add_column("Matrix")
    table.add_column("Shape")
    table.add_row("TTT", str(system.transition.TTT.shape))
    table.add_row("RRR", str(system.transition.RRR.shape))
    table.add_row("CCC", str(system.transition.CCC.shape))
    table.add_row("ZZ", str(system.measurement.ZZ.shape))
    table.add_row("DD", str(system.measurement.DD.shape))
    table.add_row("QQ", str(system.measurement.QQ.shape))
    table.add_row("EE", str(system.measurement.EE.shape))
    if system.pseudo_measurement is not None:
        table.add_row("ZZ_pseudo", str(system.pseudo_measurement.ZZ_pseudo.shape))
        table.add_row("DD_pseudo", str(system.pseudo_measurement.DD_pseudo.shape))
    console.print(table)


@app.command()
def estimate(
    data_path: Annotated[
        Path,
        typer.Option("--data", help="CSV data path used for posterior evaluation."),
    ],
    subspec: str = "ss10",
    backend: Annotated[str, typer.Option(help="Runtime backend.")] = "auto",
    device: Annotated[str, typer.Option(help="Runtime device.")] = "auto",
    optimize: Annotated[
        bool,
        typer.Option("--optimize", help="Optimize the posterior before reporting metrics."),
    ] = False,
    parameters: Annotated[
        str | None,
        typer.Option(
            "--parameters",
            help=(
                "Comma-separated parameter names for optimization; "
                "defaults to prior-backed free parameters."
            ),
        ),
    ] = None,
    maxiter: Annotated[int, typer.Option(help="Maximum optimizer iterations.")] = 100,
    hessian: Annotated[
        bool,
        typer.Option(
            "--hessian", help="Compute a finite-difference Hessian at the optimized mode."
        ),
    ] = False,
    mh_draws: Annotated[
        int,
        typer.Option("--mh-draws", help="Retained Metropolis-Hastings draws."),
    ] = 0,
    mh_burnin: Annotated[
        int,
        typer.Option("--mh-burnin", help="Metropolis-Hastings burn-in draws."),
    ] = 0,
    proposal_scale: Annotated[
        float,
        typer.Option("--proposal-scale", help="Scale applied to proposal covariance."),
    ] = 1.0,
    proposal_covariance_path: Annotated[
        Path | None,
        typer.Option(
            "--proposal-covariance",
            help="CSV, NPY, or NPZ proposal covariance in estimation space.",
        ),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Metropolis-Hastings random seed."),
    ] = None,
    sampler_output: Annotated[
        Path | None,
        typer.Option("--sampler-output", help="Write Metropolis-Hastings draws to .npz."),
    ] = None,
    mode_input: Annotated[
        Path | None,
        typer.Option("--mode-input", help="Load an optimized mode/Hessian .npz archive."),
    ] = None,
    mode_output: Annotated[
        Path | None,
        typer.Option("--mode-output", help="Write optimized mode/Hessian results to .npz."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Evaluate the translated Model1002 likelihood and posterior at current parameters."""
    model = Model1002(
        subspec=subspec,
        runtime=RuntimeConfig(backend=_parse_backend(backend), device=_parse_device(device)),
    )
    try:
        data_frame = load_data(model, path=data_path)
        start_date = _sample_start_date(model, data_frame, in_sample=True)
        data = df_to_matrix(model, data_frame)
        mode_result = None if mode_input is None else load_estimation_mode(mode_input)
        result = estimate_model(
            model,
            data,
            start_date=start_date,
            optimize=optimize,
            parameter_names=_parse_parameter_names(parameters),
            maxiter=maxiter,
            compute_hessian=hessian,
            mh_draws=mh_draws,
            mh_burnin=mh_burnin,
            proposal_scale=proposal_scale,
            proposal_covariance=_load_proposal_covariance(proposal_covariance_path),
            seed=seed,
            mode=mode_result,
        )
        mode_output_path = None
        if mode_output is not None:
            mode_output_path = save_estimation_mode(
                estimation_mode_from_result(result),
                mode_output,
            )
        sampler_output_path = None
        if sampler_output is not None:
            if result.sampler is None:
                msg = "--sampler-output requires --mh-draws > 0."
                raise ValueError(msg)
            sampler_output_path = save_sampler_result(result.sampler, sampler_output)
    except (
        FileNotFoundError,
        KeyError,
        NotPortedError,
        UnsupportedRuntimeError,
        ValueError,
    ) as err:
        raise _not_ported_exit(str(err)) from err

    payload = {
        "subspec": subspec,
        "log_likelihood": result.log_likelihood,
        "log_prior": result.log_prior,
        "log_posterior": result.log_posterior,
        "n_parameters": len(result.parameter_values),
        "filtered_states_shape": list(result.kalman.filtered_states.shape),
        "optimization": (
            None
            if result.optimization is None
            else {
                "parameter_names": list(result.optimization.parameter_names),
                "objective_value": result.optimization.objective_value,
                "success": result.optimization.success,
                "message": result.optimization.message,
                "iterations": result.optimization.iterations,
                "function_evaluations": result.optimization.function_evaluations,
            }
        ),
        "hessian_shape": None if result.hessian is None else list(result.hessian.shape),
        "mode_input": None if mode_input is None else str(mode_input),
        "mode_output": None if mode_output_path is None else str(mode_output_path),
        "sampler_output": None if sampler_output_path is None else str(sampler_output_path),
        "sampler": (
            None
            if result.sampler is None
            else {
                "parameter_names": list(result.sampler.parameter_names),
                "estimation_draws_shape": list(result.sampler.estimation_draws.shape),
                "parameter_draws_shape": list(result.sampler.parameter_draws.shape),
                "log_posterior_shape": list(result.sampler.log_posterior.shape),
                "proposal_covariance_shape": list(result.sampler.proposal_covariance.shape),
                "acceptance_rate": result.sampler.acceptance_rate,
                "seed": result.sampler.seed,
                "burnin": result.sampler.burnin,
            }
        ),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return

    table = Table(title=f"{model.description()} posterior evaluation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("log_likelihood", f"{result.log_likelihood:.6f}")
    table.add_row("log_prior", f"{result.log_prior:.6f}")
    table.add_row("log_posterior", f"{result.log_posterior:.6f}")
    table.add_row("parameters", str(len(result.parameter_values)))
    if result.optimization is not None:
        table.add_row("optimized", str(result.optimization.parameter_names))
    if result.hessian is not None:
        table.add_row("hessian", str(result.hessian.shape))
    if result.sampler is not None:
        table.add_row("mh_draws", str(result.sampler.parameter_draws.shape))
        table.add_row("mh_acceptance", f"{result.sampler.acceptance_rate:.4f}")
    console.print(table)


@app.command()
def forecast(
    subspec: str = "ss10",
    input_type: Annotated[str, typer.Option(help="Forecast input type.")] = "mode",
    cond_type: Annotated[str, typer.Option(help="Conditioning type.")] = "none",
    horizon: Annotated[int, typer.Option(help="Forecast horizon.")] = 40,
    backend: Annotated[str, typer.Option(help="Runtime backend.")] = "auto",
    device: Annotated[str, typer.Option(help="Runtime device.")] = "auto",
    data_path: Annotated[
        Path | None,
        typer.Option("--data", help="CSV data path used for filter/smoother-backed histories."),
    ] = None,
    include_history: Annotated[
        bool,
        typer.Option("--include-history", help="Include history-backed histstates and histobs."),
    ] = False,
    history_method: Annotated[
        str,
        typer.Option("--history-method", help="History method: filtered or smoothed."),
    ] = "filtered",
    include_pseudo: Annotated[
        bool,
        typer.Option("--include-pseudo", help="Include pseudo-observable forecast outputs."),
    ] = False,
    draws: Annotated[
        int,
        typer.Option("--draws", help="Full forecast shock draws; required for input_type='full'."),
    ] = 0,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Full forecast random seed."),
    ] = None,
    sampler_draws: Annotated[
        Path | None,
        typer.Option("--sampler-draws", help="Sampler .npz archive for parameter-draw forecasts."),
    ] = None,
    shock_samples_path: Annotated[
        Path | None,
        typer.Option(
            "--shock-samples",
            help="Shock sample .npy/.npz/.h5 archive for full structural-shock forecasts.",
        ),
    ] = None,
    zlb_rates: Annotated[
        str | None,
        typer.Option(
            "--zlb-rates",
            help=(
                "Comma-separated policy-rate path for ZLB/full conditioning; "
                "annualized percentage units by default."
            ),
        ),
    ] = None,
    zlb_floor: Annotated[
        float,
        typer.Option("--zlb-floor", help="Lower bound applied to --zlb-rates."),
    ] = 0.0,
    zlb_rate_units: Annotated[
        str,
        typer.Option("--zlb-rate-units", help="ZLB rate units: annualized, quarterly, or model."),
    ] = "annualized",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
    transformed: Annotated[
        bool,
        typer.Option("--transformed", help="Report observables in declared reporting units."),
    ] = False,
) -> None:
    """Run the translated Model1002 unconditional forecast smoke path."""
    model = Model1002(
        subspec=subspec,
        runtime=RuntimeConfig(backend=_parse_backend(backend), device=_parse_device(device)),
    )
    try:
        output_vars = ["forecastobs", "forecaststates"]
        if include_history:
            output_vars.extend(["histobs", "histstates"])
        if include_pseudo:
            output_vars.append("forecastpseudo")
            if include_history:
                output_vars.append("histpseudo")
        data = _load_cli_data(model, data_path)
        effective_cond_type = cond_type
        conditional_periods = None
        if zlb_rates is not None:
            if data is not None:
                msg = "--zlb-rates cannot be combined with --data."
                raise ValueError(msg)
            zlb_rate_path = _parse_number_list(zlb_rates)
            data = build_zlb_conditional_observations(
                model,
                zlb_rate_path,
                floor=zlb_floor,
                rate_units=zlb_rate_units,
            )
            effective_cond_type = "full"
            conditional_periods = int(zlb_rate_path.size)
        shock_samples = _load_shock_samples(shock_samples_path)
        output = forecast_one(
            model,
            input_type=input_type,
            cond_type=effective_cond_type,
            output_vars=output_vars,
            horizon=horizon,
            data=data,
            history_method=history_method,
            conditional_periods=conditional_periods,
            draws=draws,
            seed=seed,
            shock_samples=shock_samples,
            sampler=_load_sampler_draws(sampler_draws),
        )
        if transformed:
            output = reverse_transform_forecast(model, output)
    except (
        FileNotFoundError,
        KeyError,
        NotPortedError,
        UnsupportedRuntimeError,
        ValueError,
    ) as err:
        raise _not_ported_exit(str(err)) from err

    payload = {
        "subspec": subspec,
        "input_type": input_type,
        "cond_type": effective_cond_type,
        "horizon": horizon,
        "history_method": history_method,
        "draws": draws,
        "seed": seed,
        "sampler_draws": None if sampler_draws is None else str(sampler_draws),
        "shock_samples": None if shock_samples_path is None else str(shock_samples_path),
        "shock_samples_shape": None if shock_samples is None else list(shock_samples.shape),
        "zlb_rates": None if zlb_rates is None else _parse_number_list(zlb_rates).tolist(),
        "zlb_floor": zlb_floor,
        "zlb_rate_units": zlb_rate_units,
        "transformed": transformed,
        "states_shape": list(output.states.shape),
        "observables_shape": list(output.observables.shape),
        "pseudo_observables_shape": (
            None if output.pseudo_observables is None else list(output.pseudo_observables.shape)
        ),
        "conditional_shocks_shape": (
            None if output.conditional_shocks is None else list(output.conditional_shocks.shape)
        ),
        "conditional_states_shape": (
            None if output.conditional_states is None else list(output.conditional_states.shape)
        ),
        "conditional_observables_shape": (
            None
            if output.conditional_observables is None
            else list(output.conditional_observables.shape)
        ),
        "history_states_shape": (
            None if output.history_states is None else list(output.history_states.shape)
        ),
        "history_observables_shape": (
            None if output.history_observables is None else list(output.history_observables.shape)
        ),
        "history_pseudo_observables_shape": (
            None
            if output.history_pseudo_observables is None
            else list(output.history_pseudo_observables.shape)
        ),
        "state_samples_shape": (
            None if output.state_samples is None else list(output.state_samples.shape)
        ),
        "observable_samples_shape": (
            None if output.observable_samples is None else list(output.observable_samples.shape)
        ),
        "pseudo_observable_samples_shape": (
            None
            if output.pseudo_observable_samples is None
            else list(output.pseudo_observable_samples.shape)
        ),
        "history_state_samples_shape": (
            None
            if output.history_state_samples is None
            else list(output.history_state_samples.shape)
        ),
        "history_observable_samples_shape": (
            None
            if output.history_observable_samples is None
            else list(output.history_observable_samples.shape)
        ),
        "history_pseudo_observable_samples_shape": (
            None
            if output.history_pseudo_observable_samples is None
            else list(output.history_pseudo_observable_samples.shape)
        ),
        "log_likelihood": output.log_likelihood,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    table = Table(title=f"{model.description()} forecast")
    table.add_column("Array")
    table.add_column("Shape")
    table.add_row("states", str(output.states.shape))
    table.add_row("observables", str(output.observables.shape))
    if output.pseudo_observables is not None:
        table.add_row("pseudo_observables", str(output.pseudo_observables.shape))
    if output.conditional_shocks is not None:
        table.add_row("conditional_shocks", str(output.conditional_shocks.shape))
    if output.conditional_states is not None:
        table.add_row("conditional_states", str(output.conditional_states.shape))
    if output.conditional_observables is not None:
        table.add_row("conditional_observables", str(output.conditional_observables.shape))
    if output.history_states is not None:
        table.add_row("history_states", str(output.history_states.shape))
    if output.history_observables is not None:
        table.add_row("history_observables", str(output.history_observables.shape))
    if output.history_pseudo_observables is not None:
        table.add_row(
            "history_pseudo_observables",
            str(output.history_pseudo_observables.shape),
        )
    if output.observable_samples is not None:
        table.add_row("observable_samples", str(output.observable_samples.shape))
    console.print(table)


@app.command()
def meansbands(
    subspec: str = "ss10",
    input_type: Annotated[str, typer.Option(help="Forecast input type.")] = "mode",
    cond_type: Annotated[str, typer.Option(help="Conditioning type.")] = "none",
    horizon: Annotated[int, typer.Option(help="Forecast horizon.")] = 40,
    source: Annotated[
        str, typer.Option(help="Band source: observables or states.")
    ] = "observables",
    draws: Annotated[
        int,
        typer.Option("--draws", help="Full forecast shock draws; required for input_type='full'."),
    ] = 0,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Full forecast random seed."),
    ] = None,
    sampler_draws: Annotated[
        Path | None,
        typer.Option("--sampler-draws", help="Sampler .npz archive for parameter-draw bands."),
    ] = None,
    shock_samples_path: Annotated[
        Path | None,
        typer.Option(
            "--shock-samples",
            help="Shock sample .npy/.npz/.h5 archive for full structural-shock bands.",
        ),
    ] = None,
    lower_quantile: Annotated[
        float,
        typer.Option("--lower-quantile", help="Lower quantile for full forecast bands."),
    ] = 0.05,
    upper_quantile: Annotated[
        float,
        typer.Option("--upper-quantile", help="Upper quantile for full forecast bands."),
    ] = 0.95,
    data_path: Annotated[
        Path | None,
        typer.Option("--data", help="CSV data path used for filter/smoother-backed history bands."),
    ] = None,
    history_method: Annotated[
        str,
        typer.Option("--history-method", help="History method: filtered or smoothed."),
    ] = "filtered",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
    transformed: Annotated[
        bool,
        typer.Option("--transformed", help="Report observable bands in declared reporting units."),
    ] = False,
) -> None:
    """Compute translated Model1002 deterministic means/bands."""
    model = Model1002(subspec=subspec)
    try:
        output_vars = _forecast_output_vars_for_source(source)
        shock_samples = _load_shock_samples(shock_samples_path)
        bands = compute_meansbands(
            model,
            input_type=input_type,
            cond_type=cond_type,
            output_vars=output_vars,
            horizon=horizon,
            source=source,
            data=_load_cli_data(model, data_path),
            history_method=history_method,
            draws=draws,
            seed=seed,
            shock_samples=shock_samples,
            sampler=_load_sampler_draws(sampler_draws),
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
        if transformed:
            if source not in _transformable_band_sources():
                msg = "--transformed is only valid with observable or pseudo-observable sources."
                raise ValueError(msg)
            bands = reverse_transform_meansbands(model, bands, source=source)
    except (
        FileNotFoundError,
        KeyError,
        NotPortedError,
        UnsupportedRuntimeError,
        ValueError,
    ) as err:
        raise _not_ported_exit(str(err)) from err

    payload = {
        "subspec": subspec,
        "input_type": input_type,
        "cond_type": cond_type,
        "horizon": horizon,
        "source": source,
        "history_method": history_method,
        "draws": draws,
        "seed": seed,
        "sampler_draws": None if sampler_draws is None else str(sampler_draws),
        "shock_samples": None if shock_samples_path is None else str(shock_samples_path),
        "shock_samples_shape": None if shock_samples is None else list(shock_samples.shape),
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "transformed": transformed,
        "mean_shape": list(bands.mean.shape),
        "lower_shape": list(bands.lower.shape),
        "upper_shape": list(bands.upper.shape),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    table = Table(title=f"{model.description()} means/bands")
    table.add_column("Array")
    table.add_column("Shape")
    table.add_row("mean", str(bands.mean.shape))
    table.add_row("lower", str(bands.lower.shape))
    table.add_row("upper", str(bands.upper.shape))
    console.print(table)


@app.command()
def bench(
    kernel: Annotated[
        str,
        typer.Option(help="Benchmark kernel: forecast, kalman, or all."),
    ] = "forecast",
    horizon: Annotated[int, typer.Option(help="Forecast horizon to benchmark.")] = 40,
    periods: Annotated[int, typer.Option(help="Kalman data periods to benchmark.")] = 40,
    repeats: Annotated[int, typer.Option(help="Number of timed repeats per target.")] = 3,
    dtype: Annotated[str, typer.Option(help="Array dtype: float64 or float32.")] = "float64",
    include_pseudo: Annotated[
        bool,
        typer.Option("--include-pseudo", help="Include pseudo-observable forecast outputs."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Benchmark native kernels and explicitly report skipped targets."""
    try:
        if kernel not in {"forecast", "kalman", "all"}:
            msg = "Benchmark kernel must be forecast, kalman, or all."
            raise ValueError(msg)
        parsed_dtype = _parse_dtype(dtype)
        results = []
        if kernel in {"forecast", "all"}:
            results.extend(
                benchmark_forecast_targets(
                    horizon=horizon,
                    repeats=repeats,
                    dtype=parsed_dtype,
                    include_pseudo=include_pseudo,
                )
            )
        if kernel in {"kalman", "all"}:
            results.extend(
                benchmark_kalman_targets(
                    periods=periods,
                    repeats=repeats,
                    dtype=parsed_dtype,
                )
            )
    except ValueError as err:
        raise _not_ported_exit(str(err)) from err
    if json_output:
        typer.echo(json.dumps([result.to_dict() for result in results], indent=2))
        return

    table = Table(title="Native benchmark targets")
    table.add_column("Kernel")
    table.add_column("Backend")
    table.add_column("Device")
    table.add_column("Status")
    table.add_column("Elapsed")
    table.add_column("Reason")
    for result in results:
        if result.skipped:
            status = "skipped"
        elif result.available:
            status = "ran"
        else:
            status = "failed"
        elapsed = "" if result.elapsed_seconds is None else f"{result.elapsed_seconds:.6f}s"
        table.add_row(result.kernel, result.backend, result.device, status, elapsed, result.reason)
    console.print(table)


@vv_app.command("compare")
def vv_compare(
    oracle_dir: Annotated[
        Path,
        typer.Option(help="Julia oracle fixture directory."),
    ] = Path("tests/fixtures/oracle"),
    candidate_dir: Annotated[
        Path,
        typer.Option(help="Python candidate fixture directory."),
    ] = Path("tests/fixtures/candidate"),
    tolerance_profile: Annotated[
        str,
        typer.Option(
            "--tolerance-profile",
            help="Named tolerance profile: strict, cpu-oracle, forecast, or accelerator.",
        ),
    ] = "strict",
    atol: Annotated[
        float | None,
        typer.Option(help="Absolute tolerance override for the selected profile."),
    ] = None,
    rtol: Annotated[
        float | None,
        typer.Option(help="Relative tolerance override for the selected profile."),
    ] = None,
    coverage_profile: Annotated[
        str | None,
        typer.Option(
            "--profile",
            help="Only compare arrays required by a fixture coverage profile.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Compare Python outputs against exported Julia oracle fixtures."""
    try:
        profile = resolve_tolerance_profile(tolerance_profile, atol=atol, rtol=rtol)
        array_names = (
            None if coverage_profile is None else required_fixture_arrays(coverage_profile)
        )
        report = compare_fixture_dirs(
            oracle_dir,
            candidate_dir,
            atol=profile.atol,
            rtol=profile.rtol,
            array_names=array_names,
        )
    except ValueError as err:
        raise _not_ported_exit(str(err)) from err
    except (FileNotFoundError, NotADirectoryError) as err:
        console.print(f"[yellow]{err}[/yellow]")
        raise typer.Exit(code=2) from err
    if json_output:
        typer.echo(
            json.dumps(
                {
                    **report.to_dict(),
                    "tolerance_profile": profile.to_dict(),
                    "coverage_profile": None if coverage_profile is None else coverage_profile,
                },
                indent=2,
            )
        )
    else:
        table = Table(title=f"V&V fixture comparison ({profile.name})")
        table.add_column("Array")
        table.add_column("Status")
        table.add_column("Max abs")
        table.add_column("Max rel")
        table.add_column("Worst", overflow="fold")
        table.add_column("Message")
        for item in report.comparisons:
            table.add_row(
                item.name,
                item.status,
                "" if item.max_abs_diff is None else f"{item.max_abs_diff:.3e}",
                "" if item.max_rel_diff is None else f"{item.max_rel_diff:.3e}",
                _format_comparison_location(item.max_abs_index, item.max_abs_label),
                item.message,
            )
        console.print(table)
    if not report.passed:
        raise typer.Exit(code=1)


@vv_app.command("oracle-coverage")
def vv_oracle_coverage(
    oracle_dir: Annotated[
        Path,
        typer.Option(help="Oracle fixture directory to validate."),
    ] = Path("tests/fixtures/oracle"),
    profile: Annotated[
        str,
        typer.Option("--profile", help="Coverage profile: matrix or hard-target."),
    ] = "hard-target",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Check whether an oracle directory has required parity fixture arrays."""
    try:
        report = check_fixture_coverage(oracle_dir, profile=profile)
    except (FileNotFoundError, NotADirectoryError, ValueError) as err:
        console.print(f"[yellow]{err}[/yellow]")
        raise typer.Exit(code=2) from err
    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2))
    else:
        table = Table(title=f"Oracle fixture coverage ({report.profile})")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("oracle_dir", str(report.fixture_dir))
        table.add_row("required", str(len(report.required)))
        table.add_row("available", str(len(report.available)))
        table.add_row("missing", str(len(report.missing)))
        if report.missing:
            table.add_section()
            for name in report.missing:
                table.add_row("missing", name)
        console.print(table)
    if not report.passed:
        raise typer.Exit(code=1)


@vv_app.command("export-financial-frictions")
def vv_export_financial_frictions(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory where Python financial-frictions fixtures are written."),
    ] = Path("tests/fixtures/candidate"),
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Export Python BGG financial-frictions helper formula fixtures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs, values = _financial_frictions_fixture_arrays()
    path = output_dir / "financial_frictions.npz"
    np.savez(path, inputs=inputs, values=values)
    manifest_path = save_fixture_manifest(
        output_dir,
        {
            "fixture_kind": "financial_frictions",
            "shapes": {
                "financial_frictions": {
                    "inputs": list(inputs.shape),
                    "values": list(values.shape),
                }
            },
            "labels": _financial_frictions_labels(),
        },
    )
    payload = {
        "output": str(path),
        "manifest": str(manifest_path),
        "cases": [case[0] for case in FINANCIAL_FRICTIONS_CASES],
        "functions": list(FINANCIAL_FRICTIONS_FUNCTION_NAMES),
        "input_shape": list(inputs.shape),
        "values_shape": list(values.shape),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
    else:
        table = Table(title="Financial-frictions fixture export")
        table.add_column("Output")
        table.add_column("Cases")
        table.add_column("Functions")
        table.add_row(str(path), str(inputs.shape[0]), str(values.shape[1]))
        console.print(table)


@vv_app.command("backend-parity")
def vv_backend_parity(
    kernel: Annotated[
        str,
        typer.Option(help="Parity kernel: forecast, kalman, or all."),
    ] = "all",
    horizon: Annotated[int, typer.Option(help="Forecast horizon to compare.")] = 40,
    periods: Annotated[int, typer.Option(help="Kalman data periods to compare.")] = 40,
    dtype: Annotated[str, typer.Option(help="Target dtype: float64 or float32.")] = "float64",
    include_pseudo: Annotated[
        bool,
        typer.Option("--include-pseudo", help="Include pseudo-observable forecast outputs."),
    ] = False,
    tolerance_profile: Annotated[
        str,
        typer.Option(
            "--tolerance-profile",
            help="Named tolerance profile: strict, cpu-oracle, forecast, or accelerator.",
        ),
    ] = "strict",
    atol: Annotated[
        float | None,
        typer.Option(help="Absolute tolerance override for the selected profile."),
    ] = None,
    rtol: Annotated[
        float | None,
        typer.Option(help="Relative tolerance override for the selected profile."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Compare available native backends against NumPy CPU reference outputs."""
    try:
        profile = resolve_tolerance_profile(tolerance_profile, atol=atol, rtol=rtol)
        results = compare_backend_parity_targets(
            kernel=_parse_parity_kernel(kernel),
            horizon=horizon,
            periods=periods,
            dtype=_parse_dtype(dtype),
            include_pseudo=include_pseudo,
            atol=profile.atol,
            rtol=profile.rtol,
        )
    except ValueError as err:
        raise _not_ported_exit(str(err)) from err
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "tolerance_profile": profile.to_dict(),
                    "results": [result.to_dict() for result in results],
                },
                indent=2,
            )
        )
    else:
        table = Table(title=f"Backend parity against NumPy CPU ({profile.name})")
        table.add_column("Kernel")
        table.add_column("Backend")
        table.add_column("Device")
        table.add_column("Status")
        table.add_column("Max abs")
        table.add_column("Max rel")
        table.add_column("Reason")
        for result in results:
            if result.skipped:
                status = "skipped"
            elif result.passed:
                status = "passed"
            else:
                status = "failed"
            table.add_row(
                result.kernel,
                result.backend,
                result.device,
                status,
                "" if result.max_abs_diff is None else f"{result.max_abs_diff:.3e}",
                "" if result.max_rel_diff is None else f"{result.max_rel_diff:.3e}",
                result.reason,
            )
        console.print(table)
    if any(not result.skipped and not result.passed for result in results):
        raise typer.Exit(code=1)


@vv_app.command("runtime-purity")
def vv_runtime_purity(
    root: Annotated[
        Path,
        typer.Option(help="Runtime package root to audit."),
    ] = Path("src/nydsge"),
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Verify that runtime package paths do not invoke Julia, shells, or WSL."""
    try:
        report = audit_runtime_purity(root)
    except (FileNotFoundError, NotADirectoryError) as err:
        console.print(f"[yellow]{err}[/yellow]")
        raise typer.Exit(code=2) from err
    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2))
    else:
        table = Table(title="Runtime purity audit")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("root", str(report.root))
        table.add_row("checked_files", str(report.checked_files))
        table.add_row("passed", str(report.passed))
        if report.findings:
            table.add_section()
            table.add_row("findings", str(len(report.findings)))
            for finding in report.findings:
                table.add_row(str(finding.path), f"{finding.line}: {finding.pattern}")
        console.print(table)
    if not report.passed:
        raise typer.Exit(code=1)


@vv_app.command("solve-canonical")
def vv_solve_canonical(
    input_dir: Annotated[
        Path,
        typer.Option(help="Directory containing Gamma0, Gamma1, C, Psi, and Pi fixtures."),
    ] = Path("tests/fixtures/oracle"),
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory where Python transition fixtures will be written."),
    ] = Path("tests/fixtures/candidate"),
    method: Annotated[
        str,
        typer.Option(help="Canonical solver method: auto, direct, or gensys."),
    ] = "auto",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Solve canonical-form fixture matrices and write transition candidate outputs."""
    solve_method = _parse_canonical_solve_method(method)
    try:
        canonical = load_canonical_fixture(input_dir)
        result = solve_canonical(canonical, method=solve_method)
        path = save_transition_fixture(result, output_dir)
    except (FileNotFoundError, NotADirectoryError, KeyError, ValueError) as err:
        console.print(f"[yellow]{err}[/yellow]")
        raise typer.Exit(code=2) from err

    payload = {
        "output": str(path),
        "method": result.method,
        "eu": list(result.eu),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
    else:
        table = Table(title="Canonical solve fixture")
        table.add_column("Output")
        table.add_column("Method")
        table.add_column("EU")
        table.add_row(str(path), result.method, str(result.eu))
        console.print(table)


@vv_app.command("export-system")
def vv_export_system(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory where Python system fixtures will be written."),
    ] = Path("tests/fixtures/candidate"),
    subspec: Annotated[str, typer.Option(help="Model1002 subspec to export.")] = "ss10",
    data_vintage: Annotated[
        str,
        typer.Option("--data-vintage", help="Model data vintage setting."),
    ] = "181115",
    forecast_start: Annotated[
        str,
        typer.Option("--forecast-start", help="First forecast quarter, e.g. 2018-Q4."),
    ] = "2018-Q4",
    backend: Annotated[str, typer.Option(help="Runtime backend.")] = "auto",
    device: Annotated[str, typer.Option(help="Runtime device.")] = "auto",
    filename: Annotated[str, typer.Option(help="Output .npz filename.")] = "system.npz",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Export the Python Model1002 system fixture for oracle comparison."""
    model = Model1002(
        subspec=subspec,
        runtime=RuntimeConfig(backend=_parse_backend(backend), device=_parse_device(device)),
        settings=_model1002_settings(
            data_vintage=data_vintage,
            forecast_start=forecast_start,
        ),
    )
    try:
        system = compute_system(model)
        path = save_system_fixture(system, output_dir, filename=filename)
        manifest_path = save_fixture_manifest(
            output_dir,
            _system_manifest(
                model,
                backend=backend,
                device=device,
            ),
        )
    except (NotPortedError, ValueError) as err:
        console.print(f"[yellow]{err}[/yellow]")
        raise typer.Exit(code=2) from err

    payload = {
        "output": str(path),
        "manifest": str(manifest_path),
        "subspec": subspec,
        "data_vintage": data_vintage,
        "forecast_start": forecast_start,
        "matrices": {
            "TTT": list(system.transition.TTT.shape),
            "RRR": list(system.transition.RRR.shape),
            "CCC": list(system.transition.CCC.shape),
            "ZZ": list(system.measurement.ZZ.shape),
            "DD": list(system.measurement.DD.shape),
            "QQ": list(system.measurement.QQ.shape),
            "EE": list(system.measurement.EE.shape),
            **(
                {}
                if system.pseudo_measurement is None
                else {
                    "ZZ_pseudo": list(system.pseudo_measurement.ZZ_pseudo.shape),
                    "DD_pseudo": list(system.pseudo_measurement.DD_pseudo.shape),
                }
            ),
        },
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return

    table = Table(title="Python system fixture export")
    table.add_column("Output")
    table.add_column("Subspec")
    table.add_column("TTT")
    table.add_row(str(path), subspec, str(system.transition.TTT.shape))
    console.print(table)


@vv_app.command("export-parameters")
def vv_export_parameters(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory where Python parameter fixtures will be written."),
    ] = Path("tests/fixtures/candidate"),
    subspec: Annotated[str, typer.Option(help="Model1002 subspec to export.")] = "ss10",
    data_vintage: Annotated[
        str,
        typer.Option("--data-vintage", help="Model data vintage setting."),
    ] = "181115",
    forecast_start: Annotated[
        str,
        typer.Option("--forecast-start", help="First forecast quarter, e.g. 2018-Q4."),
    ] = "2018-Q4",
    filename: Annotated[str, typer.Option(help="Output .npz filename.")] = "parameters.npz",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Export Python Model1002 parameter fixtures for oracle comparison."""
    model = Model1002(
        subspec=subspec,
        settings=_model1002_settings(
            data_vintage=data_vintage,
            forecast_start=forecast_start,
        ),
    )
    try:
        path = save_parameter_fixture(model.parameters, output_dir, filename=filename)
        manifest_path = save_fixture_manifest(
            output_dir,
            _parameter_manifest(model, parameter_path=path),
        )
    except ValueError as err:
        console.print(f"[yellow]{err}[/yellow]")
        raise typer.Exit(code=2) from err

    payload = {
        "output": str(path),
        "manifest": str(manifest_path),
        "subspec": subspec,
        "data_vintage": data_vintage,
        "forecast_start": forecast_start,
        "parameters": len(model.parameters),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return

    table = Table(title="Python parameter fixture export")
    table.add_column("Output")
    table.add_column("Subspec")
    table.add_column("Parameters")
    table.add_row(str(path), subspec, str(len(model.parameters)))
    console.print(table)


@vv_app.command("export-steady-state")
def vv_export_steady_state(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory where Python steady-state fixtures will be written."),
    ] = Path("tests/fixtures/candidate"),
    subspec: Annotated[str, typer.Option(help="Model1002 subspec to export.")] = "ss10",
    data_vintage: Annotated[
        str,
        typer.Option("--data-vintage", help="Model data vintage setting."),
    ] = "181115",
    forecast_start: Annotated[
        str,
        typer.Option("--forecast-start", help="First forecast quarter, e.g. 2018-Q4."),
    ] = "2018-Q4",
    filename: Annotated[str, typer.Option(help="Output .npz filename.")] = "steady_state.npz",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Export Python Model1002 steady-state fixtures for oracle comparison."""
    model = Model1002(
        subspec=subspec,
        settings=_model1002_settings(
            data_vintage=data_vintage,
            forecast_start=forecast_start,
        ),
    )
    try:
        steady_state = model.steadystate()
        path = save_steady_state_fixture(steady_state, output_dir, filename=filename)
        manifest_path = save_fixture_manifest(
            output_dir,
            _steady_state_manifest(model, steady_state_path=path),
        )
    except (NotPortedError, ValueError) as err:
        console.print(f"[yellow]{err}[/yellow]")
        raise typer.Exit(code=2) from err

    payload = {
        "output": str(path),
        "manifest": str(manifest_path),
        "subspec": subspec,
        "data_vintage": data_vintage,
        "forecast_start": forecast_start,
        "steady_state": len(steady_state),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return

    table = Table(title="Python steady-state fixture export")
    table.add_column("Output")
    table.add_column("Subspec")
    table.add_column("Steady-state values")
    table.add_row(str(path), subspec, str(len(steady_state)))
    console.print(table)


@vv_app.command("export-matrices")
def vv_export_matrices(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory where Python matrix fixtures will be written."),
    ] = Path("tests/fixtures/candidate"),
    subspec: Annotated[str, typer.Option(help="Model1002 subspec to export.")] = "ss10",
    data_vintage: Annotated[
        str,
        typer.Option("--data-vintage", help="Model data vintage setting."),
    ] = "181115",
    forecast_start: Annotated[
        str,
        typer.Option("--forecast-start", help="First forecast quarter, e.g. 2018-Q4."),
    ] = "2018-Q4",
    backend: Annotated[str, typer.Option(help="Runtime backend.")] = "auto",
    device: Annotated[str, typer.Option(help="Runtime device.")] = "auto",
    method: Annotated[
        str,
        typer.Option(help="Canonical solver method: auto, direct, or gensys."),
    ] = "auto",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Export Model1002 canonical, transition, and system fixtures for oracle comparison."""
    solve_method = _parse_canonical_solve_method(method)
    runtime = RuntimeConfig(backend=_parse_backend(backend), device=_parse_device(device))
    model = Model1002(
        subspec=subspec,
        runtime=runtime,
        settings=_model1002_settings(
            data_vintage=data_vintage,
            forecast_start=forecast_start,
        ),
    )
    try:
        canonical = model.equilibrium_matrices()
        solved = solve_canonical(canonical, method=solve_method)
        system = compute_system(model, method=solve_method)
        parameter_path = save_parameter_fixture(model.parameters, output_dir)
        steady_state_path = save_steady_state_fixture(model.steady_state, output_dir)
        canonical_path = save_canonical_fixture(canonical, output_dir)
        transition_path = save_transition_fixture(solved, output_dir)
        system_path = save_system_fixture(system, output_dir)
        manifest_path = save_fixture_manifest(
            output_dir,
            _matrix_manifest(
                model,
                parameter_path=parameter_path,
                steady_state_path=steady_state_path,
                canonical_path=canonical_path,
                transition_path=transition_path,
                system_path=system_path,
                backend=backend,
                device=device,
                method=solved.method,
                eu=solved.eu,
                canonical_shapes={
                    "Gamma0": canonical.Gamma0.shape,
                    "Gamma1": canonical.Gamma1.shape,
                    "C": canonical.C.shape,
                    "Psi": canonical.Psi.shape,
                    "Pi": canonical.Pi.shape,
                },
                transition_shapes={
                    "TTT": solved.transition.TTT.shape,
                    "RRR": solved.transition.RRR.shape,
                    "CCC": solved.transition.CCC.shape,
                    "eu": np.asarray(solved.eu, dtype=np.int64).shape,
                },
                system_shapes={
                    "TTT": system.transition.TTT.shape,
                    "RRR": system.transition.RRR.shape,
                    "CCC": system.transition.CCC.shape,
                    "ZZ": system.measurement.ZZ.shape,
                    "DD": system.measurement.DD.shape,
                    "QQ": system.measurement.QQ.shape,
                    "EE": system.measurement.EE.shape,
                },
            ),
        )
    except (NotPortedError, ValueError) as err:
        console.print(f"[yellow]{err}[/yellow]")
        raise typer.Exit(code=2) from err

    payload = {
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "parameters": str(parameter_path),
        "steady_state": str(steady_state_path),
        "canonical": str(canonical_path),
        "transition": str(transition_path),
        "system": str(system_path),
        "subspec": subspec,
        "data_vintage": data_vintage,
        "forecast_start": forecast_start,
        "method": solved.method,
        "eu": list(solved.eu),
        "canonical_shape": list(canonical.Gamma0.shape),
        "system_shape": list(system.transition.TTT.shape),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return

    table = Table(title="Python matrix fixture export")
    table.add_column("Output")
    table.add_column("Parameters")
    table.add_column("Steady state")
    table.add_column("Canonical")
    table.add_column("Transition")
    table.add_column("System")
    table.add_row(
        str(output_dir),
        str(len(model.parameters)),
        str(len(model.steady_state)),
        str(canonical.Gamma0.shape),
        str(solved.transition.TTT.shape),
        str(system.transition.TTT.shape),
    )
    console.print(table)


@vv_app.command("export-forecast")
def vv_export_forecast(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory where Python forecast fixtures will be written."),
    ] = Path("tests/fixtures/candidate"),
    subspec: Annotated[str, typer.Option(help="Model1002 subspec to export.")] = "ss10",
    data_vintage: Annotated[
        str,
        typer.Option("--data-vintage", help="Model data vintage setting."),
    ] = "181115",
    forecast_start: Annotated[
        str,
        typer.Option("--forecast-start", help="First forecast quarter, e.g. 2018-Q4."),
    ] = "2018-Q4",
    input_type: Annotated[str, typer.Option(help="Forecast input type.")] = "mode",
    cond_type: Annotated[str, typer.Option(help="Conditioning type.")] = "none",
    horizon: Annotated[int, typer.Option(help="Forecast horizon.")] = 40,
    data_path: Annotated[
        Path | None,
        typer.Option("--data", help="CSV data path used for filter/smoother-backed histories."),
    ] = None,
    include_history: Annotated[
        bool,
        typer.Option("--include-history", help="Include history-backed histstates and histobs."),
    ] = False,
    history_method: Annotated[
        str,
        typer.Option("--history-method", help="History method: filtered or smoothed."),
    ] = "filtered",
    include_pseudo: Annotated[
        bool,
        typer.Option("--include-pseudo", help="Include pseudo-observable forecast outputs."),
    ] = False,
    draws: Annotated[
        int,
        typer.Option("--draws", help="Full forecast shock draws; required for input_type='full'."),
    ] = 0,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Full forecast random seed."),
    ] = None,
    sampler_draws: Annotated[
        Path | None,
        typer.Option("--sampler-draws", help="Sampler .npz archive for parameter-draw forecasts."),
    ] = None,
    shock_samples_path: Annotated[
        Path | None,
        typer.Option(
            "--shock-samples",
            help="Shock sample .npy/.npz/.h5 archive for full structural-shock forecasts.",
        ),
    ] = None,
    zlb_rates: Annotated[
        str | None,
        typer.Option(
            "--zlb-rates",
            help=(
                "Comma-separated policy-rate path for ZLB/full conditioning; "
                "annualized percentage units by default."
            ),
        ),
    ] = None,
    zlb_floor: Annotated[
        float,
        typer.Option("--zlb-floor", help="Lower bound applied to --zlb-rates."),
    ] = 0.0,
    zlb_rate_units: Annotated[
        str,
        typer.Option("--zlb-rate-units", help="ZLB rate units: annualized, quarterly, or model."),
    ] = "annualized",
    filename: Annotated[str, typer.Option(help="Output .npz filename.")] = "forecast.npz",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
    transformed: Annotated[
        bool,
        typer.Option("--transformed", help="Export observables in declared reporting units."),
    ] = False,
) -> None:
    """Export the Python Model1002 forecast fixture for oracle comparison."""
    model = Model1002(
        subspec=subspec,
        settings=_model1002_settings(
            data_vintage=data_vintage,
            forecast_start=forecast_start,
        ),
    )
    try:
        output_vars = ["forecastobs", "forecaststates"]
        if include_history:
            output_vars.extend(["histobs", "histstates"])
        if include_pseudo:
            output_vars.append("forecastpseudo")
            if include_history:
                output_vars.append("histpseudo")
        data = _load_cli_data(model, data_path)
        effective_cond_type = cond_type
        conditional_periods = None
        if zlb_rates is not None:
            if data is not None:
                msg = "--zlb-rates cannot be combined with --data."
                raise ValueError(msg)
            zlb_rate_path = _parse_number_list(zlb_rates)
            data = build_zlb_conditional_observations(
                model,
                zlb_rate_path,
                floor=zlb_floor,
                rate_units=zlb_rate_units,
            )
            effective_cond_type = "full"
            conditional_periods = int(zlb_rate_path.size)
        sampler = _load_sampler_draws(sampler_draws)
        shock_samples = _load_shock_samples(shock_samples_path)
        forecast = forecast_one(
            model,
            input_type=input_type,
            cond_type=effective_cond_type,
            output_vars=output_vars,
            horizon=horizon,
            data=data,
            history_method=history_method,
            conditional_periods=conditional_periods,
            draws=draws,
            seed=seed,
            shock_samples=shock_samples,
            sampler=sampler,
        )
        if transformed:
            forecast = reverse_transform_forecast(model, forecast)
        path = save_forecast_fixture(forecast, output_dir, filename=filename)
        manifest_path = save_fixture_manifest(
            output_dir,
            _forecast_manifest(
                model,
                forecast,
                input_type=input_type,
                cond_type=effective_cond_type,
                horizon=horizon,
                history_method=history_method,
                data_path=data_path,
                transformed=transformed,
                array_prefix=Path(filename).stem,
                sampler=sampler,
                sampler_path=sampler_draws,
                shock_samples=shock_samples,
                shock_samples_path=shock_samples_path,
            ),
        )
    except (
        FileNotFoundError,
        KeyError,
        NotPortedError,
        UnsupportedRuntimeError,
        ValueError,
    ) as err:
        console.print(f"[yellow]{err}[/yellow]")
        raise typer.Exit(code=2) from err

    payload = {
        "output": str(path),
        "manifest": str(manifest_path),
        "subspec": subspec,
        "data_vintage": data_vintage,
        "forecast_start": forecast_start,
        "input_type": input_type,
        "cond_type": effective_cond_type,
        "horizon": horizon,
        "history_method": history_method,
        "draws": draws,
        "seed": seed,
        "sampler_draws": None if sampler_draws is None else str(sampler_draws),
        "shock_samples": None if shock_samples_path is None else str(shock_samples_path),
        "shock_samples_shape": None if shock_samples is None else list(shock_samples.shape),
        "zlb_rates": None if zlb_rates is None else _parse_number_list(zlb_rates).tolist(),
        "zlb_floor": zlb_floor,
        "zlb_rate_units": zlb_rate_units,
        "transformed": transformed,
        "states_shape": list(forecast.states.shape),
        "observables_shape": list(forecast.observables.shape),
        "pseudo_observables_shape": (
            None if forecast.pseudo_observables is None else list(forecast.pseudo_observables.shape)
        ),
        "conditional_shocks_shape": (
            None if forecast.conditional_shocks is None else list(forecast.conditional_shocks.shape)
        ),
        "conditional_states_shape": (
            None if forecast.conditional_states is None else list(forecast.conditional_states.shape)
        ),
        "conditional_observables_shape": (
            None
            if forecast.conditional_observables is None
            else list(forecast.conditional_observables.shape)
        ),
        "history_states_shape": (
            None if forecast.history_states is None else list(forecast.history_states.shape)
        ),
        "history_observables_shape": (
            None
            if forecast.history_observables is None
            else list(forecast.history_observables.shape)
        ),
        "history_pseudo_observables_shape": (
            None
            if forecast.history_pseudo_observables is None
            else list(forecast.history_pseudo_observables.shape)
        ),
        "state_samples_shape": (
            None if forecast.state_samples is None else list(forecast.state_samples.shape)
        ),
        "observable_samples_shape": (
            None if forecast.observable_samples is None else list(forecast.observable_samples.shape)
        ),
        "pseudo_observable_samples_shape": (
            None
            if forecast.pseudo_observable_samples is None
            else list(forecast.pseudo_observable_samples.shape)
        ),
        "history_state_samples_shape": (
            None
            if forecast.history_state_samples is None
            else list(forecast.history_state_samples.shape)
        ),
        "history_observable_samples_shape": (
            None
            if forecast.history_observable_samples is None
            else list(forecast.history_observable_samples.shape)
        ),
        "history_pseudo_observable_samples_shape": (
            None
            if forecast.history_pseudo_observable_samples is None
            else list(forecast.history_pseudo_observable_samples.shape)
        ),
        "log_likelihood": forecast.log_likelihood,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return

    table = Table(title="Python forecast fixture export")
    table.add_column("Output")
    table.add_column("States")
    table.add_column("Observables")
    table.add_row(str(path), str(forecast.states.shape), str(forecast.observables.shape))
    console.print(table)


@vv_app.command("export-meansbands")
def vv_export_meansbands(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory where Python means/bands fixtures will be written."),
    ] = Path("tests/fixtures/candidate"),
    subspec: Annotated[str, typer.Option(help="Model1002 subspec to export.")] = "ss10",
    data_vintage: Annotated[
        str,
        typer.Option("--data-vintage", help="Model data vintage setting."),
    ] = "181115",
    forecast_start: Annotated[
        str,
        typer.Option("--forecast-start", help="First forecast quarter, e.g. 2018-Q4."),
    ] = "2018-Q4",
    input_type: Annotated[str, typer.Option(help="Forecast input type.")] = "mode",
    cond_type: Annotated[str, typer.Option(help="Conditioning type.")] = "none",
    horizon: Annotated[int, typer.Option(help="Forecast horizon.")] = 40,
    source: Annotated[
        str, typer.Option(help="Band source: observables or states.")
    ] = "observables",
    draws: Annotated[
        int,
        typer.Option("--draws", help="Full forecast shock draws; required for input_type='full'."),
    ] = 0,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Full forecast random seed."),
    ] = None,
    sampler_draws: Annotated[
        Path | None,
        typer.Option("--sampler-draws", help="Sampler .npz archive for parameter-draw bands."),
    ] = None,
    shock_samples_path: Annotated[
        Path | None,
        typer.Option(
            "--shock-samples",
            help="Shock sample .npy/.npz/.h5 archive for full structural-shock bands.",
        ),
    ] = None,
    lower_quantile: Annotated[
        float,
        typer.Option("--lower-quantile", help="Lower quantile for full forecast bands."),
    ] = 0.05,
    upper_quantile: Annotated[
        float,
        typer.Option("--upper-quantile", help="Upper quantile for full forecast bands."),
    ] = 0.95,
    data_path: Annotated[
        Path | None,
        typer.Option("--data", help="CSV data path used for filter/smoother-backed history bands."),
    ] = None,
    history_method: Annotated[
        str,
        typer.Option("--history-method", help="History method: filtered or smoothed."),
    ] = "filtered",
    filename: Annotated[str, typer.Option(help="Output .npz filename.")] = "meansbands.npz",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
    transformed: Annotated[
        bool,
        typer.Option("--transformed", help="Export observable bands in declared reporting units."),
    ] = False,
) -> None:
    """Export the Python Model1002 means/bands fixture for oracle comparison."""
    model = Model1002(
        subspec=subspec,
        settings=_model1002_settings(
            data_vintage=data_vintage,
            forecast_start=forecast_start,
        ),
    )
    try:
        output_vars = _forecast_output_vars_for_source(source)
        sampler = _load_sampler_draws(sampler_draws)
        shock_samples = _load_shock_samples(shock_samples_path)
        bands = compute_meansbands(
            model,
            input_type=input_type,
            cond_type=cond_type,
            output_vars=output_vars,
            horizon=horizon,
            source=source,
            data=_load_cli_data(model, data_path),
            history_method=history_method,
            draws=draws,
            seed=seed,
            shock_samples=shock_samples,
            sampler=sampler,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
        if transformed:
            if source not in _transformable_band_sources():
                msg = "--transformed is only valid with observable or pseudo-observable sources."
                raise ValueError(msg)
            bands = reverse_transform_meansbands(model, bands, source=source)
        path = save_meansbands_fixture(bands, output_dir, filename=filename)
        manifest_path = save_fixture_manifest(
            output_dir,
            _meansbands_manifest(
                model,
                bands,
                input_type=input_type,
                cond_type=cond_type,
                horizon=horizon,
                source=source,
                history_method=history_method,
                data_path=data_path,
                transformed=transformed,
                array_prefix=Path(filename).stem,
                sampler=sampler,
                sampler_path=sampler_draws,
                shock_samples=shock_samples,
                shock_samples_path=shock_samples_path,
            ),
        )
    except (
        FileNotFoundError,
        KeyError,
        NotPortedError,
        UnsupportedRuntimeError,
        ValueError,
    ) as err:
        console.print(f"[yellow]{err}[/yellow]")
        raise typer.Exit(code=2) from err

    payload = {
        "output": str(path),
        "manifest": str(manifest_path),
        "subspec": subspec,
        "data_vintage": data_vintage,
        "forecast_start": forecast_start,
        "input_type": input_type,
        "cond_type": cond_type,
        "horizon": horizon,
        "source": source,
        "history_method": history_method,
        "draws": draws,
        "seed": seed,
        "sampler_draws": None if sampler_draws is None else str(sampler_draws),
        "shock_samples": None if shock_samples_path is None else str(shock_samples_path),
        "shock_samples_shape": None if shock_samples is None else list(shock_samples.shape),
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "transformed": transformed,
        "mean_shape": list(bands.mean.shape),
        "lower_shape": list(bands.lower.shape),
        "upper_shape": list(bands.upper.shape),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return

    table = Table(title="Python means/bands fixture export")
    table.add_column("Output")
    table.add_column("Mean")
    table.add_column("Lower")
    table.add_column("Upper")
    table.add_row(str(path), str(bands.mean.shape), str(bands.lower.shape), str(bands.upper.shape))
    console.print(table)


@vv_app.command("export-kalman")
def vv_export_kalman(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory where Python Kalman fixtures will be written."),
    ] = Path("tests/fixtures/candidate"),
    subspec: Annotated[str, typer.Option(help="Model1002 subspec to export.")] = "ss10",
    data_vintage: Annotated[
        str,
        typer.Option("--data-vintage", help="Model data vintage setting."),
    ] = "181115",
    forecast_start: Annotated[
        str,
        typer.Option("--forecast-start", help="First forecast quarter, e.g. 2018-Q4."),
    ] = "2018-Q4",
    data_path: Annotated[
        Path,
        typer.Option("--data", help="CSV data path used for Kalman filter parity."),
    ] = Path("observables.csv"),
    filename: Annotated[str, typer.Option(help="Output .npz filename.")] = "kalman.npz",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Export Python Model1002 Kalman filter arrays for oracle comparison."""
    model = Model1002(
        subspec=subspec,
        settings=_model1002_settings(
            data_vintage=data_vintage,
            forecast_start=forecast_start,
        ),
    )
    try:
        data = _load_cli_data(model, data_path)
        if data is None:
            msg = "--data is required for Kalman fixture export."
            raise ValueError(msg)
        start_date = _sample_start_date(model, data, in_sample=True)
        observations = df_to_matrix(model, cast(Any, data), in_sample=True)
        system = compute_system(model)
        kalman = kalman_log_likelihood(
            system,
            observations,
            process_covariances=model_process_covariances(
                model,
                system,
                observations.shape[0],
                start_date=start_date,
            ),
        )
        path = save_kalman_fixture(kalman, output_dir, filename=filename)
        manifest_path = save_fixture_manifest(
            output_dir,
            _kalman_manifest(
                model,
                kalman,
                data_path=data_path,
                array_prefix=Path(filename).stem,
            ),
        )
    except (FileNotFoundError, KeyError, UnsupportedRuntimeError, ValueError) as err:
        console.print(f"[yellow]{err}[/yellow]")
        raise typer.Exit(code=2) from err

    payload = {
        "output": str(path),
        "manifest": str(manifest_path),
        "subspec": subspec,
        "data_vintage": data_vintage,
        "forecast_start": forecast_start,
        "data": str(data_path),
        "log_likelihood": kalman.log_likelihood,
        "periods": int(kalman.filtered_states.shape[0]),
        "states": int(kalman.filtered_states.shape[1]),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return

    table = Table(title="Python Kalman fixture export")
    table.add_column("Output")
    table.add_column("Log likelihood")
    table.add_column("Filtered states")
    table.add_row(str(path), f"{kalman.log_likelihood:.12g}", str(kalman.filtered_states.shape))
    console.print(table)


@vv_app.command("export-posterior")
def vv_export_posterior(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory where Python posterior fixtures will be written."),
    ] = Path("tests/fixtures/candidate"),
    subspec: Annotated[str, typer.Option(help="Model1002 subspec to export.")] = "ss10",
    data_vintage: Annotated[
        str,
        typer.Option("--data-vintage", help="Model data vintage setting."),
    ] = "181115",
    forecast_start: Annotated[
        str,
        typer.Option("--forecast-start", help="First forecast quarter, e.g. 2018-Q4."),
    ] = "2018-Q4",
    data_path: Annotated[
        Path,
        typer.Option("--data", help="CSV data path used for posterior parity."),
    ] = Path("observables.csv"),
    filename: Annotated[str, typer.Option(help="Output .npz filename.")] = "posterior.npz",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Export Python Model1002 current-parameter posterior arrays for oracle comparison."""
    model = Model1002(
        subspec=subspec,
        settings=_model1002_settings(
            data_vintage=data_vintage,
            forecast_start=forecast_start,
        ),
    )
    try:
        data = _load_cli_data(model, data_path)
        if data is None:
            msg = "--data is required for posterior fixture export."
            raise ValueError(msg)
        start_date = _sample_start_date(model, data, in_sample=True)
        observations = df_to_matrix(model, cast(Any, data), in_sample=True)
        result = estimate_model(model, observations, start_date=start_date)
        path = save_posterior_fixture(result, model.parameters, output_dir, filename=filename)
        manifest_path = save_fixture_manifest(
            output_dir,
            _posterior_manifest(
                model,
                result,
                data_path=data_path,
                array_prefix=Path(filename).stem,
            ),
        )
    except (FileNotFoundError, KeyError, UnsupportedRuntimeError, ValueError) as err:
        console.print(f"[yellow]{err}[/yellow]")
        raise typer.Exit(code=2) from err

    payload = {
        "output": str(path),
        "manifest": str(manifest_path),
        "subspec": subspec,
        "data_vintage": data_vintage,
        "forecast_start": forecast_start,
        "data": str(data_path),
        "log_posterior": result.log_posterior,
        "log_likelihood": result.log_likelihood,
        "log_prior": result.log_prior,
        "periods": int(result.kalman.log_likelihood_by_period.shape[0]),
        "parameters": len(result.parameter_values),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return

    table = Table(title="Python posterior fixture export")
    table.add_column("Output")
    table.add_column("Log posterior")
    table.add_column("Log likelihood")
    table.add_column("Log prior")
    table.add_row(
        str(path),
        f"{result.log_posterior:.12g}",
        f"{result.log_likelihood:.12g}",
        f"{result.log_prior:.12g}",
    )
    console.print(table)


@vv_app.command("export-suite")
def vv_export_suite(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory where Python candidate suite fixtures will be written."),
    ] = Path("tests/fixtures/candidate"),
    oracle_dir: Annotated[
        Path | None,
        typer.Option("--oracle-dir", help="Optional Julia oracle directory to compare."),
    ] = None,
    subspec: Annotated[str, typer.Option(help="Model1002 subspec to export.")] = "ss10",
    data_vintage: Annotated[
        str,
        typer.Option("--data-vintage", help="Model data vintage setting."),
    ] = "181115",
    forecast_start: Annotated[
        str,
        typer.Option("--forecast-start", help="First forecast quarter, e.g. 2018-Q4."),
    ] = "2018-Q4",
    horizon: Annotated[int, typer.Option(help="Forecast horizon.")] = 40,
    data_path: Annotated[
        Path | None,
        typer.Option("--data", help="CSV data path for histobs/history parity artifacts."),
    ] = None,
    history_method: Annotated[
        str,
        typer.Option("--history-method", help="History method: filtered or smoothed."),
    ] = "filtered",
    full_draws: Annotated[
        int,
        typer.Option("--full-draws", help="Full forecast structural shock draws to export."),
    ] = 0,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Full forecast random seed."),
    ] = None,
    sampler_draws: Annotated[
        Path | None,
        typer.Option("--sampler-draws", help="Sampler .npz archive for full parameter draws."),
    ] = None,
    shock_samples_path: Annotated[
        Path | None,
        typer.Option(
            "--shock-samples",
            help="Shock sample .npy/.npz/.h5 archive for full structural-shock fixtures.",
        ),
    ] = None,
    allow_empty_data_columns: Annotated[
        bool,
        typer.Option(
            "--allow-empty-data-columns",
            help=(
                "Allow all-empty observable columns in --data. This mirrors Julia "
                "load_data(..., check_empty_columns=false) for V&V oracle replay."
            ),
        ),
    ] = False,
    tolerance_profile: Annotated[
        str,
        typer.Option(
            "--tolerance-profile",
            help="Named tolerance profile used when --oracle-dir is present.",
        ),
    ] = "forecast",
    require_oracle: Annotated[
        bool,
        typer.Option("--require-oracle", help="Fail if --oracle-dir is missing or absent."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Export the standard Model1002 Python candidate parity suite."""
    try:
        exported = _export_model1002_candidate_suite(
            output_dir=output_dir,
            subspec=subspec,
            data_vintage=data_vintage,
            forecast_start=forecast_start,
            horizon=horizon,
            data_path=data_path,
            history_method=history_method,
            full_draws=full_draws,
            seed=seed,
            sampler_draws=sampler_draws,
            shock_samples_path=shock_samples_path,
            allow_empty_data_columns=allow_empty_data_columns,
        )
        comparison: dict[str, object]
        if oracle_dir is None or not oracle_dir.exists():
            if require_oracle:
                msg = f"Oracle directory is required but not available: {oracle_dir}"
                raise FileNotFoundError(msg)
            comparison = {
                "status": "skipped",
                "reason": "oracle directory was not provided or does not exist",
            }
        else:
            profile = resolve_tolerance_profile(tolerance_profile)
            report = compare_fixture_dirs(
                oracle_dir,
                output_dir,
                atol=profile.atol,
                rtol=profile.rtol,
            )
            comparison = {
                "status": "passed" if report.passed else "failed",
                "tolerance_profile": profile.to_dict(),
                "report": report.to_dict(),
            }
    except (
        FileNotFoundError,
        KeyError,
        NotPortedError,
        UnsupportedRuntimeError,
        ValueError,
    ) as err:
        console.print(f"[yellow]{err}[/yellow]")
        raise typer.Exit(code=2) from err

    payload = {
        "output_dir": str(output_dir),
        "subspec": subspec,
        "data_vintage": data_vintage,
        "forecast_start": forecast_start,
        "horizon": horizon,
        "data": None if data_path is None else str(data_path),
        "history_method": history_method,
        "full_draws": full_draws,
        "seed": seed,
        "sampler_draws": None if sampler_draws is None else str(sampler_draws),
        "shock_samples": None if shock_samples_path is None else str(shock_samples_path),
        "allow_empty_data_columns": allow_empty_data_columns,
        "exported": exported,
        "comparison": comparison,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
    else:
        table = Table(title="Model1002 candidate parity suite")
        table.add_column("Artifact")
        table.add_column("Path")
        for artifact in exported:
            table.add_row(str(artifact["kind"]), str(artifact["path"]))
        table.add_row("comparison", str(comparison["status"]))
        console.print(table)
    if comparison["status"] == "failed":
        raise typer.Exit(code=1)


@vv_app.command("export-hard-target-inputs")
def vv_export_hard_target_inputs(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory where deterministic hard-target smoke inputs are written."),
    ] = Path("tests/fixtures/hard_target_smoke"),
    subspec: Annotated[str, typer.Option(help="Model1002 subspec for the smoke inputs.")] = "ss10",
    data_vintage: Annotated[
        str,
        typer.Option("--data-vintage", help="Model data vintage setting recorded in manifest."),
    ] = "181115",
    forecast_start: Annotated[
        str,
        typer.Option("--forecast-start", help="First forecast quarter, e.g. 2018-Q4."),
    ] = "2018-Q4",
    periods: Annotated[
        int,
        typer.Option("--periods", help="Number of pre-forecast history rows to write."),
    ] = 2,
    horizon: Annotated[
        int, typer.Option(help="Forecast horizon for follow-on smoke commands.")
    ] = 2,
    draws: Annotated[
        int,
        typer.Option("--draws", help="Repeated zero-shock full-forecast draws to write."),
    ] = 2,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table."),
    ] = False,
) -> None:
    """Write deterministic CSV/NPZ inputs for the Julia/Python hard-target smoke."""
    try:
        payload = _export_hard_target_smoke_inputs(
            output_dir=output_dir,
            subspec=subspec,
            data_vintage=data_vintage,
            forecast_start=forecast_start,
            periods=periods,
            horizon=horizon,
            draws=draws,
        )
    except (NotPortedError, ValueError) as err:
        console.print(f"[yellow]{err}[/yellow]")
        raise typer.Exit(code=2) from err

    if json_output:
        typer.echo(json.dumps(payload, indent=2))
    else:
        table = Table(title="Deterministic hard-target smoke inputs")
        table.add_column("Artifact")
        table.add_column("Path")
        table.add_row("observables", str(payload["observables_csv"]))
        table.add_row("shock_samples", str(payload["shock_samples"]))
        table.add_row("manifest", str(payload["manifest"]))
        console.print(table)


def _not_ported_exit(message: str) -> NotPortedError:
    console.print(f"[yellow]{message}[/yellow]")
    raise typer.Exit(code=2)


def _format_comparison_location(
    index: tuple[int, ...] | None,
    label: tuple[str | None, ...] | None,
) -> str:
    if index is None:
        return ""
    if label is None:
        return str(index)
    parts = [
        str(label_value) if label_value is not None else str(index_value)
        for index_value, label_value in zip(index, label, strict=True)
    ]
    return " / ".join(parts)


def _model1002_settings(*, data_vintage: str, forecast_start: str) -> dict[str, str]:
    return {
        "data_vintage": data_vintage,
        "date_forecast_start": forecast_start,
    }


def _financial_frictions_fixture_arrays() -> tuple[np.ndarray, np.ndarray]:
    inputs = np.asarray(
        [[z, sigma, spr] for _, z, sigma, spr in FINANCIAL_FRICTIONS_CASES],
        dtype=np.float64,
    )
    values = np.asarray(
        [
            _evaluate_financial_frictions_case(z, sigma, spr)
            for _, z, sigma, spr in FINANCIAL_FRICTIONS_CASES
        ],
        dtype=np.float64,
    )
    return inputs, values


def _evaluate_financial_frictions_case(z: float, sigma: float, spr: float) -> list[float]:
    return [
        omega_fn(z, sigma),
        g_fn(z, sigma),
        gamma_fn(z, sigma),
        dg_domega_fn(z, sigma),
        d2g_domega2_fn(z, sigma),
        dgamma_domega_fn(z),
        d2gamma_domega2_fn(z, sigma),
        dg_dsigma_fn(z, sigma),
        d2g_domega_dsigma_fn(z, sigma),
        dgamma_dsigma_fn(z, sigma),
        d2gamma_domega_dsigma_fn(z, sigma),
        mu_fn(z, sigma, spr),
        nk_fn(z, sigma, spr),
        zeta_bomega_fn(z, sigma, spr),
        zeta_zomega_fn(z, sigma, spr),
        zeta_spb_fn(z, sigma, spr),
    ]


def _financial_frictions_labels() -> dict[str, dict[str, list[str]]]:
    case_names = [case[0] for case in FINANCIAL_FRICTIONS_CASES]
    return {
        "financial_frictions/inputs": {
            "axis0": case_names,
            "axis1": list(FINANCIAL_FRICTIONS_INPUT_NAMES),
        },
        "financial_frictions/values": {
            "axis0": case_names,
            "axis1": list(FINANCIAL_FRICTIONS_FUNCTION_NAMES),
        },
    }


def _export_hard_target_smoke_inputs(
    *,
    output_dir: Path,
    subspec: str,
    data_vintage: str,
    forecast_start: str,
    periods: int,
    horizon: int,
    draws: int,
) -> dict[str, object]:
    if periods <= 0:
        msg = "--periods must be positive."
        raise ValueError(msg)
    if horizon < 0:
        msg = "Forecast horizon must be nonnegative."
        raise ValueError(msg)
    if draws <= 0:
        msg = "--draws must be positive."
        raise ValueError(msg)

    model = Model1002(
        subspec=subspec,
        settings=_model1002_settings(
            data_vintage=data_vintage,
            forecast_start=forecast_start,
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = _quarters_before(forecast_start, periods)
    observables_path = output_dir / "observables.csv"
    shock_samples_path = output_dir / "zero_shocks.npz"
    manifest_path = output_dir / "hard_target_smoke_manifest.json"
    oracle_path = output_dir / "oracle" / f"m1002_{subspec}_hardtarget.h5"
    candidate_dir = output_dir / "candidate"
    observable_names = list(model.observables)
    shock_names = list(model.indexes.exogenous_shocks)

    with observables_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", *observable_names])
        for date in dates:
            writer.writerow([date, *([0.0] * len(observable_names))])

    shock_samples = np.zeros((draws, horizon, len(shock_names)), dtype=np.float64)
    np.savez(shock_samples_path, shock_samples=shock_samples)

    julia_oracle_command = [
        "julia",
        "+1.8",
        "--project=tools/oracle_julia",
        "tools/oracle_julia/export_model1002.jl",
        "--out",
        str(oracle_path),
        "--include-history",
        "true",
        "--include-forecast",
        "true",
        "--include-full-forecast",
        "true",
        "--full-draws",
        str(draws),
        "--include-posterior",
        "true",
        "--data-in",
        str(observables_path),
        "--horizon",
        str(horizon),
    ]
    python_candidate_command = [
        "uv",
        "run",
        "nydsge",
        "vv",
        "export-suite",
        "--output-dir",
        str(candidate_dir),
        "--data",
        str(observables_path),
        "--shock-samples",
        str(shock_samples_path),
        "--horizon",
        str(horizon),
        "--json",
    ]
    compare_command = [
        "uv",
        "run",
        "nydsge",
        "vv",
        "compare",
        "--oracle-dir",
        str(oracle_path.parent),
        "--candidate-dir",
        str(candidate_dir),
        "--profile",
        "hard-target",
        "--tolerance-profile",
        "strict",
        "--json",
    ]
    payload: dict[str, object] = {
        "output_dir": str(output_dir),
        "subspec": subspec,
        "data_vintage": data_vintage,
        "forecast_start": forecast_start,
        "history_periods": periods,
        "horizon": horizon,
        "draws": draws,
        "dates": dates,
        "observables": len(observable_names),
        "shocks": len(shock_names),
        "observables_csv": str(observables_path),
        "shock_samples": str(shock_samples_path),
        "manifest": str(manifest_path),
        "julia_oracle_command": julia_oracle_command,
        "python_candidate_command": python_candidate_command,
        "compare_command": compare_command,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _quarters_before(forecast_start: str, periods: int) -> list[str]:
    forecast_index = _quarter_index_from_label(forecast_start)
    first_index = forecast_index - periods
    return [_quarter_label_from_index(first_index + offset) for offset in range(periods)]


def _quarter_index_from_label(value: str) -> int:
    text = str(value).strip()
    try:
        year_text, quarter_text = text.split("-Q", 1)
        year = int(year_text)
        quarter = int(quarter_text)
    except ValueError as err:
        msg = "Quarter labels must use YYYY-QN format, for example 2018-Q4."
        raise ValueError(msg) from err
    if quarter < 1 or quarter > 4:
        msg = "Quarter labels must use quarter 1, 2, 3, or 4."
        raise ValueError(msg)
    return year * 4 + quarter


def _quarter_label_from_index(value: int) -> str:
    year = (value - 1) // 4
    quarter = value - year * 4
    return f"{year}-Q{quarter}"


def _export_model1002_candidate_suite(
    *,
    output_dir: Path,
    subspec: str,
    data_vintage: str,
    forecast_start: str,
    horizon: int,
    data_path: Path | None,
    history_method: str,
    full_draws: int,
    seed: int | None,
    sampler_draws: Path | None,
    shock_samples_path: Path | None,
    allow_empty_data_columns: bool = False,
) -> list[dict[str, object]]:
    if horizon < 0:
        msg = "Forecast horizon must be nonnegative."
        raise ValueError(msg)
    if full_draws < 0:
        msg = "--full-draws must be nonnegative."
        raise ValueError(msg)
    model = Model1002(
        subspec=subspec,
        settings=_model1002_settings(
            data_vintage=data_vintage,
            forecast_start=forecast_start,
        ),
    )
    exported: list[dict[str, object]] = []

    canonical = model.equilibrium_matrices()
    solved = solve_canonical(canonical)
    system = compute_system(model)
    parameter_path = save_parameter_fixture(model.parameters, output_dir)
    steady_state_path = save_steady_state_fixture(model.steady_state, output_dir)
    canonical_path = save_canonical_fixture(canonical, output_dir)
    transition_path = save_transition_fixture(solved, output_dir)
    system_path = save_system_fixture(system, output_dir)
    manifest_path = save_fixture_manifest(
        output_dir,
        _matrix_manifest(
            model,
            parameter_path=parameter_path,
            steady_state_path=steady_state_path,
            canonical_path=canonical_path,
            transition_path=transition_path,
            system_path=system_path,
            backend="auto",
            device="auto",
            method=solved.method,
            eu=solved.eu,
            canonical_shapes={
                "Gamma0": canonical.Gamma0.shape,
                "Gamma1": canonical.Gamma1.shape,
                "C": canonical.C.shape,
                "Psi": canonical.Psi.shape,
                "Pi": canonical.Pi.shape,
            },
            transition_shapes={
                "TTT": solved.transition.TTT.shape,
                "RRR": solved.transition.RRR.shape,
                "CCC": solved.transition.CCC.shape,
                "eu": np.asarray(solved.eu, dtype=np.int64).shape,
            },
            system_shapes={
                "TTT": system.transition.TTT.shape,
                "RRR": system.transition.RRR.shape,
                "CCC": system.transition.CCC.shape,
                "ZZ": system.measurement.ZZ.shape,
                "DD": system.measurement.DD.shape,
                "QQ": system.measurement.QQ.shape,
                "EE": system.measurement.EE.shape,
            },
        ),
    )
    exported.extend(
        [
            {"kind": "canonical", "path": str(canonical_path)},
            {"kind": "parameters", "path": str(parameter_path)},
            {"kind": "steady_state", "path": str(steady_state_path)},
            {"kind": "transition", "path": str(transition_path)},
            {"kind": "system", "path": str(system_path)},
            {"kind": "manifest", "path": str(manifest_path)},
        ]
    )

    data = _load_cli_data(
        model,
        data_path,
        allow_empty_data_columns=allow_empty_data_columns,
    )
    mode_output_vars = ["forecastobs", "forecaststates"]
    if data_path is not None:
        mode_output_vars.extend(["histobs", "histstates"])
    mode_forecast = forecast_one(
        model,
        input_type="mode",
        cond_type="none",
        output_vars=mode_output_vars,
        horizon=horizon,
        data=data,
        history_method=history_method,
    )
    _save_suite_forecast(
        model,
        mode_forecast,
        output_dir=output_dir,
        filename="forecast_mode.npz",
        input_type="mode",
        cond_type="none",
        horizon=horizon,
        history_method=history_method,
        data_path=data_path,
        exported=exported,
        allow_empty_data_columns=allow_empty_data_columns,
    )
    mode_bands = compute_meansbands(
        model,
        "mode",
        "none",
        ["forecastobs"],
        horizon=horizon,
        source="forecastobs",
        data=data,
        history_method=history_method,
    )
    _save_suite_meansbands(
        model,
        mode_bands,
        output_dir=output_dir,
        filename="meansbands_mode_forecastobs.npz",
        input_type="mode",
        cond_type="none",
        horizon=horizon,
        source="forecastobs",
        history_method=history_method,
        data_path=data_path,
        exported=exported,
        allow_empty_data_columns=allow_empty_data_columns,
    )
    if data_path is not None:
        mode_history_bands = compute_meansbands(
            model,
            "mode",
            "none",
            ["histobs"],
            horizon=horizon,
            source="histobs",
            data=data,
            history_method=history_method,
        )
        _save_suite_meansbands(
            model,
            mode_history_bands,
            output_dir=output_dir,
            filename="meansbands_mode_histobs.npz",
            input_type="mode",
            cond_type="none",
            horizon=horizon,
            source="histobs",
            history_method=history_method,
            data_path=data_path,
            exported=exported,
            allow_empty_data_columns=allow_empty_data_columns,
        )

        observations = df_to_matrix(model, cast(Any, data), in_sample=True)
        posterior = estimate_model(
            model,
            observations,
            start_date=_sample_start_date(model, data, in_sample=True),
        )
        _save_suite_posterior(
            model,
            posterior,
            output_dir=output_dir,
            filename="posterior.npz",
            data_path=data_path,
            exported=exported,
            allow_empty_data_columns=allow_empty_data_columns,
        )

    sampler = _load_sampler_draws(sampler_draws)
    shock_samples = _load_shock_samples(shock_samples_path)
    if full_draws > 0 or sampler is not None or shock_samples is not None:
        full_forecast = forecast_one(
            model,
            input_type="full",
            cond_type="none",
            output_vars=mode_output_vars,
            horizon=horizon,
            data=data,
            history_method=history_method,
            draws=full_draws,
            seed=seed,
            shock_samples=shock_samples,
            sampler=sampler,
        )
        _save_suite_forecast(
            model,
            full_forecast,
            output_dir=output_dir,
            filename="forecast_full.npz",
            input_type="full",
            cond_type="none",
            horizon=horizon,
            history_method=history_method,
            data_path=data_path,
            exported=exported,
            sampler=sampler,
            sampler_path=sampler_draws,
            shock_samples=shock_samples,
            shock_samples_path=shock_samples_path,
            allow_empty_data_columns=allow_empty_data_columns,
        )
        full_bands = compute_meansbands(
            model,
            "full",
            "none",
            ["forecastobs"],
            horizon=horizon,
            source="forecastobs",
            data=data,
            history_method=history_method,
            draws=full_draws,
            seed=seed,
            shock_samples=shock_samples,
            sampler=sampler,
        )
        _save_suite_meansbands(
            model,
            full_bands,
            output_dir=output_dir,
            filename="meansbands_full_forecastobs.npz",
            input_type="full",
            cond_type="none",
            horizon=horizon,
            source="forecastobs",
            history_method=history_method,
            data_path=data_path,
            exported=exported,
            sampler=sampler,
            sampler_path=sampler_draws,
            shock_samples=shock_samples,
            shock_samples_path=shock_samples_path,
            allow_empty_data_columns=allow_empty_data_columns,
        )
        if data_path is not None:
            full_history_bands = compute_meansbands(
                model,
                "full",
                "none",
                ["histobs"],
                horizon=horizon,
                source="histobs",
                data=data,
                history_method=history_method,
                draws=full_draws,
                seed=seed,
                shock_samples=shock_samples,
                sampler=sampler,
            )
            _save_suite_meansbands(
                model,
                full_history_bands,
                output_dir=output_dir,
                filename="meansbands_full_histobs.npz",
                input_type="full",
                cond_type="none",
                horizon=horizon,
                source="histobs",
                history_method=history_method,
                data_path=data_path,
                exported=exported,
                sampler=sampler,
                sampler_path=sampler_draws,
                shock_samples=shock_samples,
                shock_samples_path=shock_samples_path,
                allow_empty_data_columns=allow_empty_data_columns,
            )
    else:
        exported.append(
            {
                "kind": "full",
                "path": None,
                "status": "skipped",
                "reason": (
                    "pass --full-draws, --shock-samples, or --sampler-draws to export full fixtures"
                ),
            }
        )
    return exported


def _save_suite_forecast(
    model: Model1002,
    forecast: ForecastOutput,
    *,
    output_dir: Path,
    filename: str,
    input_type: str,
    cond_type: str,
    horizon: int,
    history_method: str,
    data_path: Path | None,
    exported: list[dict[str, object]],
    sampler: MetropolisHastingsResult | None = None,
    sampler_path: Path | None = None,
    shock_samples: np.ndarray | None = None,
    shock_samples_path: Path | None = None,
    allow_empty_data_columns: bool = False,
) -> None:
    path = save_forecast_fixture(forecast, output_dir, filename=filename)
    manifest_path = save_fixture_manifest(
        output_dir,
        _forecast_manifest(
            model,
            forecast,
            input_type=input_type,
            cond_type=cond_type,
            horizon=horizon,
            history_method=history_method,
            data_path=data_path,
            transformed=False,
            array_prefix=Path(filename).stem,
            sampler=sampler,
            sampler_path=sampler_path,
            shock_samples=shock_samples,
            shock_samples_path=shock_samples_path,
            allow_empty_data_columns=allow_empty_data_columns,
        ),
    )
    exported.append({"kind": Path(filename).stem, "path": str(path)})
    exported.append({"kind": "manifest", "path": str(manifest_path)})


def _save_suite_meansbands(
    model: Model1002,
    bands: MeansBands,
    *,
    output_dir: Path,
    filename: str,
    input_type: str,
    cond_type: str,
    horizon: int,
    source: str,
    history_method: str,
    data_path: Path | None,
    exported: list[dict[str, object]],
    sampler: MetropolisHastingsResult | None = None,
    sampler_path: Path | None = None,
    shock_samples: np.ndarray | None = None,
    shock_samples_path: Path | None = None,
    allow_empty_data_columns: bool = False,
) -> None:
    path = save_meansbands_fixture(bands, output_dir, filename=filename)
    manifest_path = save_fixture_manifest(
        output_dir,
        _meansbands_manifest(
            model,
            bands,
            input_type=input_type,
            cond_type=cond_type,
            horizon=horizon,
            source=source,
            history_method=history_method,
            data_path=data_path,
            transformed=False,
            array_prefix=Path(filename).stem,
            sampler=sampler,
            sampler_path=sampler_path,
            shock_samples=shock_samples,
            shock_samples_path=shock_samples_path,
            allow_empty_data_columns=allow_empty_data_columns,
        ),
    )
    exported.append({"kind": Path(filename).stem, "path": str(path)})
    exported.append({"kind": "manifest", "path": str(manifest_path)})


def _save_suite_posterior(
    model: Model1002,
    result: EstimateResult,
    *,
    output_dir: Path,
    filename: str,
    data_path: Path,
    exported: list[dict[str, object]],
    allow_empty_data_columns: bool = False,
) -> None:
    path = save_posterior_fixture(result, model.parameters, output_dir, filename=filename)
    manifest_path = save_fixture_manifest(
        output_dir,
        _posterior_manifest(
            model,
            result,
            data_path=data_path,
            array_prefix=Path(filename).stem,
            allow_empty_data_columns=allow_empty_data_columns,
        ),
    )
    exported.append({"kind": Path(filename).stem, "path": str(path)})
    exported.append({"kind": "manifest", "path": str(manifest_path)})


def _matrix_manifest(
    model: Model1002,
    *,
    parameter_path: Path,
    steady_state_path: Path,
    canonical_path: Path,
    transition_path: Path,
    system_path: Path,
    backend: str,
    device: str,
    method: str,
    eu: tuple[int, int],
    canonical_shapes: dict[str, tuple[int, ...]],
    transition_shapes: dict[str, tuple[int, ...]],
    system_shapes: dict[str, tuple[int, ...]],
) -> dict[str, object]:
    manifest = _parameter_manifest(model, parameter_path=parameter_path)
    steady_state_manifest = _steady_state_manifest(model, steady_state_path=steady_state_path)
    system_manifest = _system_manifest(model, backend=backend, device=device)
    manifest.update(
        {key: value for key, value in system_manifest.items() if key not in {"kind", "labels"}}
    )
    manifest["labels"] = {
        **cast(dict[str, dict[str, list[str]]], manifest["labels"]),
        **cast(dict[str, dict[str, list[str]]], steady_state_manifest["labels"]),
        **cast(dict[str, dict[str, list[str]]], system_manifest["labels"]),
    }
    manifest.update(
        {
            "kind": "model1002_matrix_candidate",
            "method": method,
            "eu": list(eu),
            "parameters": str(parameter_path.name),
            "steady_state": str(steady_state_path.name),
            "canonical": str(canonical_path.name),
            "transition": str(transition_path.name),
            "system": str(system_path.name),
            "shapes": {
                "canonical": {name: list(shape) for name, shape in canonical_shapes.items()},
                "transition": {name: list(shape) for name, shape in transition_shapes.items()},
                "system": {name: list(shape) for name, shape in system_shapes.items()},
            },
        }
    )
    labels = manifest["labels"]
    equation_labels = list(model.indexes.equilibrium_conditions)
    core_state_labels = list(model.indexes.endogenous_states)
    shock_labels = list(model.indexes.exogenous_shocks)
    expected_shock_labels = list(model.indexes.expected_shocks)
    _add_array_labels(
        labels,
        "canonical/Gamma0",
        canonical_shapes["Gamma0"],
        {0: equation_labels, 1: core_state_labels},
    )
    _add_array_labels(
        labels,
        "canonical/Gamma1",
        canonical_shapes["Gamma1"],
        {0: equation_labels, 1: core_state_labels},
    )
    _add_array_labels(labels, "canonical/C", canonical_shapes["C"], {0: equation_labels})
    _add_array_labels(
        labels,
        "canonical/Psi",
        canonical_shapes["Psi"],
        {0: equation_labels, 1: shock_labels},
    )
    _add_array_labels(
        labels,
        "canonical/Pi",
        canonical_shapes["Pi"],
        {0: equation_labels, 1: expected_shock_labels},
    )
    _add_array_labels(
        labels,
        "transition/TTT",
        transition_shapes["TTT"],
        {0: core_state_labels, 1: core_state_labels},
    )
    _add_array_labels(
        labels,
        "transition/RRR",
        transition_shapes["RRR"],
        {0: core_state_labels, 1: shock_labels},
    )
    _add_array_labels(labels, "transition/CCC", transition_shapes["CCC"], {0: core_state_labels})
    _add_array_labels(
        labels,
        "transition/eu",
        transition_shapes["eu"],
        {0: ["existence", "uniqueness"]},
    )
    return manifest


def _steady_state_manifest(model: Model1002, *, steady_state_path: Path) -> dict[str, object]:
    steady_state_labels = list(model.steady_state)
    labels: dict[str, dict[str, list[str]]] = {}
    _add_array_labels(
        labels,
        "steady_state/values",
        (len(steady_state_labels),),
        {0: steady_state_labels},
    )
    return {
        "kind": "model1002_steady_state_candidate",
        "model": "Model1002",
        "subspec": model.subspec,
        "data_vintage": str(model.get_setting("data_vintage")),
        "forecast_start": str(model.get_setting("date_forecast_start")),
        "steady_state": str(steady_state_path.name),
        "steady_state_count": len(steady_state_labels),
        "labels": labels,
    }


def _parameter_manifest(model: Model1002, *, parameter_path: Path) -> dict[str, object]:
    parameter_labels = list(model.parameters)
    labels: dict[str, dict[str, list[str]]] = {}
    _add_array_labels(labels, "parameters/values", (len(parameter_labels),), {0: parameter_labels})
    _add_array_labels(
        labels,
        "parameters/scaled_values",
        (len(parameter_labels),),
        {0: parameter_labels},
    )
    _add_array_labels(labels, "parameters/fixed", (len(parameter_labels),), {0: parameter_labels})
    _add_array_labels(
        labels,
        "parameters/bounds",
        (len(parameter_labels), 2),
        {0: parameter_labels, 1: ["lower", "upper"]},
    )
    return {
        "kind": "model1002_parameter_candidate",
        "model": "Model1002",
        "subspec": model.subspec,
        "data_vintage": str(model.get_setting("data_vintage")),
        "forecast_start": str(model.get_setting("date_forecast_start")),
        "parameters": str(parameter_path.name),
        "parameter_count": len(parameter_labels),
        "parameter_metadata": [
            {
                "name": parameter.name,
                "fixed": parameter.fixed,
                "bounds": (
                    None if parameter.value_bounds is None else list(parameter.value_bounds)
                ),
                "transform": parameter.transform,
                "scaling": parameter.scaling,
                "prior": _parameter_prior_metadata(parameter.prior),
                "description": parameter.description,
                "tex_label": parameter.tex_label,
                "category": parameter.category,
                "regime": parameter.regime,
            }
            for parameter in model.parameters.values()
        ],
        "labels": labels,
    }


def _parameter_prior_metadata(prior: object | None) -> object | None:
    if prior is None:
        return None
    if isinstance(prior, dict):
        return prior
    if hasattr(prior, "__dict__"):
        return {
            key: value
            for key, value in vars(prior).items()
            if isinstance(value, str | int | float | bool) or value is None
        }
    return str(prior)


def _system_manifest(
    model: Model1002,
    *,
    backend: str,
    device: str,
) -> dict[str, object]:
    labels: dict[str, dict[str, list[str]]] = {}
    state_labels = _state_labels(model)
    shock_labels = list(model.indexes.exogenous_shocks)
    observable_labels = list(model.observables)
    pseudo_labels = list(model.pseudo_observables)
    _add_array_labels(
        labels,
        "system/TTT",
        (len(state_labels), len(state_labels)),
        {0: state_labels, 1: state_labels},
    )
    _add_array_labels(
        labels,
        "system/RRR",
        (len(state_labels), len(shock_labels)),
        {0: state_labels, 1: shock_labels},
    )
    _add_array_labels(labels, "system/CCC", (len(state_labels),), {0: state_labels})
    _add_array_labels(
        labels,
        "system/ZZ",
        (len(observable_labels), len(state_labels)),
        {0: observable_labels, 1: state_labels},
    )
    _add_array_labels(labels, "system/DD", (len(observable_labels),), {0: observable_labels})
    _add_array_labels(
        labels,
        "system/QQ",
        (len(shock_labels), len(shock_labels)),
        {0: shock_labels, 1: shock_labels},
    )
    _add_array_labels(
        labels,
        "system/EE",
        (len(observable_labels), len(observable_labels)),
        {0: observable_labels, 1: observable_labels},
    )
    _add_array_labels(
        labels,
        "system/ZZ_pseudo",
        (len(pseudo_labels), len(state_labels)),
        {0: pseudo_labels, 1: state_labels},
    )
    _add_array_labels(labels, "system/DD_pseudo", (len(pseudo_labels),), {0: pseudo_labels})
    return {
        "kind": "model1002_system_candidate",
        "model": "Model1002",
        "subspec": model.subspec,
        "data_vintage": str(model.get_setting("data_vintage")),
        "forecast_start": str(model.get_setting("date_forecast_start")),
        "backend": backend,
        "device": device,
        "labels": labels,
    }


def _kalman_manifest(
    model: Model1002,
    kalman: KalmanResult,
    *,
    data_path: Path,
    array_prefix: str = "kalman",
) -> dict[str, object]:
    history_dates = _history_date_labels(model, data_path)
    state_labels = _state_labels(model)
    labels: dict[str, dict[str, list[str]]] = {}
    _add_array_labels(
        labels,
        f"{array_prefix}/log_likelihood",
        kalman.log_likelihood_by_period.shape,
        {0: history_dates},
    )
    _add_array_labels(
        labels,
        f"{array_prefix}/predicted_states",
        kalman.predicted_states.shape,
        {0: history_dates, 1: state_labels},
    )
    _add_array_labels(
        labels,
        f"{array_prefix}/filtered_states",
        kalman.filtered_states.shape,
        {0: history_dates, 1: state_labels},
    )
    covariance_axes = {0: history_dates, 1: state_labels, 2: state_labels}
    _add_array_labels(
        labels,
        f"{array_prefix}/predicted_covariances",
        kalman.predicted_covariances.shape,
        covariance_axes,
    )
    _add_array_labels(
        labels,
        f"{array_prefix}/filtered_covariances",
        kalman.filtered_covariances.shape,
        covariance_axes,
    )
    _add_array_labels(
        labels,
        f"{array_prefix}/final_filtered_state",
        kalman.final_filtered_state.shape,
        {0: state_labels},
    )
    _add_array_labels(
        labels,
        f"{array_prefix}/total_log_likelihood",
        (1,),
        {0: ["total"]},
    )
    return {
        "kind": "model1002_kalman_candidate",
        "model": "Model1002",
        "subspec": model.subspec,
        "data_vintage": str(model.get_setting("data_vintage")),
        "forecast_start": str(model.get_setting("date_forecast_start")),
        "data": str(data_path),
        "periods": int(kalman.filtered_states.shape[0]),
        "state_count": int(kalman.filtered_states.shape[1]),
        "log_likelihood": kalman.log_likelihood,
        "labels": labels,
    }


def _posterior_manifest(
    model: Model1002,
    result: EstimateResult,
    *,
    data_path: Path,
    array_prefix: str = "posterior",
    allow_empty_data_columns: bool = False,
) -> dict[str, object]:
    history_dates = _history_date_labels(
        model,
        data_path,
        allow_empty_data_columns=allow_empty_data_columns,
    )
    parameter_labels = list(result.parameter_values)
    labels: dict[str, dict[str, list[str]]] = {}
    for name in ("log_posterior", "log_likelihood", "log_prior"):
        _add_array_labels(
            labels,
            f"{array_prefix}/{name}",
            (1,),
            {0: ["value"]},
        )
    _add_array_labels(
        labels,
        f"{array_prefix}/log_likelihood_by_period",
        result.kalman.log_likelihood_by_period.shape,
        {0: history_dates},
    )
    _add_array_labels(
        labels,
        f"{array_prefix}/log_prior_by_parameter",
        (len(parameter_labels),),
        {0: parameter_labels},
    )
    _add_array_labels(
        labels,
        f"{array_prefix}/parameter_values",
        (len(parameter_labels),),
        {0: parameter_labels},
    )
    return {
        "kind": "model1002_posterior_candidate",
        "model": "Model1002",
        "subspec": model.subspec,
        "data_vintage": str(model.get_setting("data_vintage")),
        "forecast_start": str(model.get_setting("date_forecast_start")),
        "data": str(data_path),
        "periods": int(result.kalman.log_likelihood_by_period.shape[0]),
        "parameter_count": len(result.parameter_values),
        "log_posterior": result.log_posterior,
        "log_likelihood": result.log_likelihood,
        "log_prior": result.log_prior,
        "labels": labels,
    }


def _forecast_manifest(
    model: Model1002,
    forecast: ForecastOutput,
    *,
    input_type: str,
    cond_type: str,
    horizon: int,
    history_method: str,
    data_path: Path | None,
    transformed: bool,
    array_prefix: str = "forecast",
    sampler: MetropolisHastingsResult | None = None,
    sampler_path: Path | None = None,
    shock_samples: np.ndarray | None = None,
    shock_samples_path: Path | None = None,
    allow_empty_data_columns: bool = False,
) -> dict[str, object]:
    conditional_periods = (
        0 if forecast.conditional_states is None else forecast.conditional_states.shape[0]
    )
    forecast_dates = _forecast_date_labels(
        model,
        forecast.states.shape[0],
        offset=conditional_periods,
    )
    conditional_dates = _forecast_date_labels(model, conditional_periods)
    history_dates = _history_date_labels(
        model,
        data_path,
        allow_empty_data_columns=allow_empty_data_columns,
    )
    state_labels = _state_labels(model)
    observable_labels = list(model.observables)
    pseudo_labels = list(model.pseudo_observables)
    labels: dict[str, dict[str, list[str]]] = {}
    _add_array_labels(
        labels,
        f"{array_prefix}/states",
        forecast.states.shape,
        {0: forecast_dates, 1: state_labels},
    )
    _add_array_labels(
        labels,
        f"{array_prefix}/observables",
        forecast.observables.shape,
        {0: forecast_dates, 1: observable_labels},
    )
    if forecast.pseudo_observables is not None:
        _add_array_labels(
            labels,
            f"{array_prefix}/pseudo_observables",
            forecast.pseudo_observables.shape,
            {0: forecast_dates, 1: pseudo_labels},
        )
    if forecast.conditional_shocks is not None:
        _add_array_labels(
            labels,
            f"{array_prefix}/conditional_shocks",
            forecast.conditional_shocks.shape,
            {0: conditional_dates, 1: list(model.indexes.exogenous_shocks)},
        )
    if forecast.conditional_states is not None:
        _add_array_labels(
            labels,
            f"{array_prefix}/conditional_states",
            forecast.conditional_states.shape,
            {0: conditional_dates, 1: state_labels},
        )
    if forecast.conditional_observables is not None:
        _add_array_labels(
            labels,
            f"{array_prefix}/conditional_observables",
            forecast.conditional_observables.shape,
            {0: conditional_dates, 1: observable_labels},
        )
    if forecast.state_samples is not None:
        _add_array_labels(
            labels,
            f"{array_prefix}/state_samples",
            forecast.state_samples.shape,
            {
                0: _draw_labels(forecast.state_samples.shape[0]),
                1: forecast_dates,
                2: state_labels,
            },
        )
    if forecast.observable_samples is not None:
        _add_array_labels(
            labels,
            f"{array_prefix}/observable_samples",
            forecast.observable_samples.shape,
            {
                0: _draw_labels(forecast.observable_samples.shape[0]),
                1: forecast_dates,
                2: observable_labels,
            },
        )
    if forecast.pseudo_observable_samples is not None:
        _add_array_labels(
            labels,
            f"{array_prefix}/pseudo_observable_samples",
            forecast.pseudo_observable_samples.shape,
            {
                0: _draw_labels(forecast.pseudo_observable_samples.shape[0]),
                1: forecast_dates,
                2: pseudo_labels,
            },
        )
    if forecast.history_states is not None:
        _add_array_labels(
            labels,
            f"{array_prefix}/history_states",
            forecast.history_states.shape,
            {0: history_dates, 1: state_labels},
        )
    if forecast.history_observables is not None:
        _add_array_labels(
            labels,
            f"{array_prefix}/history_observables",
            forecast.history_observables.shape,
            {0: history_dates, 1: observable_labels},
        )
    if forecast.history_pseudo_observables is not None:
        _add_array_labels(
            labels,
            f"{array_prefix}/history_pseudo_observables",
            forecast.history_pseudo_observables.shape,
            {0: history_dates, 1: pseudo_labels},
        )
    if forecast.history_state_samples is not None:
        _add_array_labels(
            labels,
            f"{array_prefix}/history_state_samples",
            forecast.history_state_samples.shape,
            {
                0: _draw_labels(forecast.history_state_samples.shape[0]),
                1: history_dates,
                2: state_labels,
            },
        )
    if forecast.history_observable_samples is not None:
        _add_array_labels(
            labels,
            f"{array_prefix}/history_observable_samples",
            forecast.history_observable_samples.shape,
            {
                0: _draw_labels(forecast.history_observable_samples.shape[0]),
                1: history_dates,
                2: observable_labels,
            },
        )
    if forecast.history_pseudo_observable_samples is not None:
        _add_array_labels(
            labels,
            f"{array_prefix}/history_pseudo_observable_samples",
            forecast.history_pseudo_observable_samples.shape,
            {
                0: _draw_labels(forecast.history_pseudo_observable_samples.shape[0]),
                1: history_dates,
                2: pseudo_labels,
            },
        )
    manifest: dict[str, object] = {
        "kind": "model1002_forecast_candidate",
        "model": "Model1002",
        "subspec": model.subspec,
        "data_vintage": str(model.get_setting("data_vintage")),
        "forecast_start": str(model.get_setting("date_forecast_start")),
        "input_type": input_type,
        "cond_type": cond_type,
        "horizon": horizon,
        "history_method": history_method,
        "transformed": transformed,
        "labels": labels,
    }
    sampler_metadata = _sampler_manifest(sampler, sampler_path)
    if sampler_metadata is not None:
        manifest["samplers"] = {array_prefix: sampler_metadata}
    shock_sample_metadata = _shock_samples_manifest(shock_samples, shock_samples_path)
    if shock_sample_metadata is not None:
        manifest["shock_samples"] = {array_prefix: shock_sample_metadata}
    return manifest


def _meansbands_manifest(
    model: Model1002,
    bands: MeansBands,
    *,
    input_type: str,
    cond_type: str,
    horizon: int,
    source: str,
    history_method: str,
    data_path: Path | None,
    transformed: bool,
    array_prefix: str = "meansbands",
    sampler: MetropolisHastingsResult | None = None,
    sampler_path: Path | None = None,
    shock_samples: np.ndarray | None = None,
    shock_samples_path: Path | None = None,
    allow_empty_data_columns: bool = False,
) -> dict[str, object]:
    axis0_labels = (
        _history_date_labels(
            model,
            data_path,
            allow_empty_data_columns=allow_empty_data_columns,
        )
        if source in {"history_observables", "histobs", "history_states", "histstates"}
        or source in {"history_pseudo_observables", "histpseudo", "histpseudoobs"}
        else _forecast_date_labels(model, bands.mean.shape[0])
    )
    variable_labels = _variable_labels_for_source(model, source)
    labels: dict[str, dict[str, list[str]]] = {}
    for name, values in (
        (f"{array_prefix}/mean", bands.mean),
        (f"{array_prefix}/lower", bands.lower),
        (f"{array_prefix}/upper", bands.upper),
    ):
        _add_array_labels(labels, name, values.shape, {0: axis0_labels, 1: variable_labels})
    manifest: dict[str, object] = {
        "kind": "model1002_meansbands_candidate",
        "model": "Model1002",
        "subspec": model.subspec,
        "data_vintage": str(model.get_setting("data_vintage")),
        "forecast_start": str(model.get_setting("date_forecast_start")),
        "input_type": input_type,
        "cond_type": cond_type,
        "horizon": horizon,
        "source": source,
        "history_method": history_method,
        "transformed": transformed,
        "labels": labels,
    }
    sampler_metadata = _sampler_manifest(sampler, sampler_path)
    if sampler_metadata is not None:
        manifest["samplers"] = {array_prefix: sampler_metadata}
    shock_sample_metadata = _shock_samples_manifest(shock_samples, shock_samples_path)
    if shock_sample_metadata is not None:
        manifest["shock_samples"] = {array_prefix: shock_sample_metadata}
    return manifest


def _sampler_manifest(
    sampler: MetropolisHastingsResult | None,
    sampler_path: Path | None,
) -> dict[str, object] | None:
    if sampler is None:
        return None
    return {
        "kind": "metropolis_hastings_result",
        "source_path": None if sampler_path is None else str(sampler_path),
        "parameter_names": list(sampler.parameter_names),
        "estimation_draws_shape": list(sampler.estimation_draws.shape),
        "parameter_draws_shape": list(sampler.parameter_draws.shape),
        "log_posterior_shape": list(sampler.log_posterior.shape),
        "accepted_shape": list(sampler.accepted.shape),
        "proposal_covariance_shape": list(sampler.proposal_covariance.shape),
        "draws": int(sampler.parameter_draws.shape[0]),
        "parameter_count": int(sampler.parameter_draws.shape[1]),
        "acceptance_rate": float(sampler.acceptance_rate),
        "burnin": int(sampler.burnin),
        "seed": None if sampler.seed is None else int(sampler.seed),
    }


def _shock_samples_manifest(
    shock_samples: np.ndarray | None,
    shock_samples_path: Path | None,
) -> dict[str, object] | None:
    if shock_samples is None:
        return None
    sample_array = np.asarray(shock_samples, dtype=np.float64)
    return {
        "kind": "structural_shock_samples",
        "source_path": None if shock_samples_path is None else str(shock_samples_path),
        "shape": list(sample_array.shape),
        "draws": int(sample_array.shape[0]),
    }


def _add_array_labels(
    labels: dict[str, dict[str, list[str]]],
    name: str,
    shape: tuple[int, ...],
    axis_labels: dict[int, list[str]],
) -> None:
    valid = {
        f"axis{axis}": values
        for axis, values in axis_labels.items()
        if axis < len(shape) and len(values) == shape[axis]
    }
    if valid:
        labels[name] = valid


def _forecast_date_labels(model: Model1002, periods: int, *, offset: int = 0) -> list[str]:
    if offset < 0:
        msg = "Forecast label offset must be nonnegative."
        raise ValueError(msg)
    labels = quarter_labels_from_start(model.get_setting("date_forecast_start"), periods + offset)
    return labels[offset:]


def _sample_start_date(model: Model1002, data: Any, *, in_sample: bool) -> str | None:
    if not hasattr(data, "columns"):
        return None
    labels = date_labels_for_sample(model, data, in_sample=in_sample)
    if not labels:
        return None
    return labels[0]


def _history_date_labels(
    model: Model1002,
    data_path: Path | None,
    *,
    allow_empty_data_columns: bool = False,
) -> list[str]:
    if data_path is None:
        return []
    return date_labels_for_sample(
        model,
        load_data(
            model,
            path=data_path,
            check_empty_columns=not allow_empty_data_columns,
        ),
    )


def _draw_labels(count: int) -> list[str]:
    return [f"draw_{index}" for index in range(count)]


def _variable_labels_for_source(model: Model1002, source: str) -> list[str]:
    if source in {"observables", "forecastobs", "history_observables", "histobs"}:
        return list(model.observables)
    if source in {"states", "forecaststates", "history_states", "histstates"}:
        return _state_labels(model)
    if source in {
        "pseudo_observables",
        "forecastpseudo",
        "forecastpseudoobs",
        "pseudoobs",
        "history_pseudo_observables",
        "histpseudo",
        "histpseudoobs",
    }:
        return list(model.pseudo_observables)
    return []


def _state_labels(model: Model1002) -> list[str]:
    return list(model.indexes.endogenous_states) + list(model.indexes.endogenous_states_augmented)


def _load_cli_data(
    model: Model1002,
    data_path: Path | None,
    *,
    allow_empty_data_columns: bool = False,
) -> object | None:
    if data_path is None:
        return None
    return load_data(
        model,
        path=data_path,
        check_empty_columns=not allow_empty_data_columns,
    )


def _forecast_output_vars_for_source(source: str) -> list[str]:
    if source in {"observables", "forecastobs"}:
        return ["forecastobs"]
    if source in {"states", "forecaststates"}:
        return ["forecaststates"]
    if source in {"pseudo_observables", "forecastpseudo", "forecastpseudoobs", "pseudoobs"}:
        return ["forecastpseudo"]
    if source in {"history_observables", "histobs"}:
        return ["histobs"]
    if source in {"history_states", "histstates"}:
        return ["histstates"]
    if source in {"history_pseudo_observables", "histpseudo", "histpseudoobs"}:
        return ["histpseudo"]
    msg = f"Unsupported forecast source: {source}"
    raise ValueError(msg)


def _transformable_band_sources() -> set[str]:
    return {
        "observables",
        "forecastobs",
        "history_observables",
        "histobs",
        "pseudo_observables",
        "forecastpseudo",
        "forecastpseudoobs",
        "pseudoobs",
        "history_pseudo_observables",
        "histpseudo",
        "histpseudoobs",
    }


def _parse_backend(value: str) -> BackendName:
    if value not in get_args(BackendName):
        msg = f"Unsupported backend: {value}"
        raise typer.BadParameter(msg)
    return cast(BackendName, value)


def _parse_device(value: str) -> DeviceName:
    if value not in get_args(DeviceName):
        msg = f"Unsupported device: {value}"
        raise typer.BadParameter(msg)
    return cast(DeviceName, value)


def _parse_canonical_solve_method(value: str) -> CanonicalSolveMethod:
    if value not in get_args(CanonicalSolveMethod):
        msg = f"Unsupported canonical solve method: {value}"
        raise typer.BadParameter(msg)
    return cast(CanonicalSolveMethod, value)


def _parse_dtype(value: str) -> DTypeName:
    if value not in get_args(DTypeName):
        msg = f"Unsupported dtype: {value}"
        raise typer.BadParameter(msg)
    return cast(DTypeName, value)


def _parse_parity_kernel(value: str) -> ParityKernel:
    if value not in {"forecast", "kalman", "all"}:
        msg = f"Unsupported parity kernel: {value}"
        raise typer.BadParameter(msg)
    return cast(ParityKernel, value)


def _parse_parameter_names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        msg = "--parameters must include at least one parameter name."
        raise typer.BadParameter(msg)
    return names


def _load_proposal_covariance(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    if not path.exists():
        msg = f"Proposal covariance path does not exist: {path}"
        raise FileNotFoundError(msg)
    if path.suffix == ".npy":
        return np.atleast_2d(np.asarray(np.load(path), dtype=np.float64))
    if path.suffix == ".npz":
        with np.load(path) as archive:
            if "proposal_covariance" in archive.files:
                return np.atleast_2d(np.asarray(archive["proposal_covariance"], dtype=np.float64))
            if len(archive.files) == 1:
                return np.atleast_2d(np.asarray(archive[archive.files[0]], dtype=np.float64))
        msg = "NPZ proposal covariance must contain proposal_covariance or exactly one array."
        raise ValueError(msg)
    if path.suffix == ".csv":
        return np.atleast_2d(np.loadtxt(path, delimiter=",", dtype=np.float64))
    msg = "Proposal covariance must be a .csv, .npy, or .npz file."
    raise ValueError(msg)


def _load_sampler_draws(path: Path | None) -> MetropolisHastingsResult | None:
    if path is None:
        return None
    if not path.exists():
        msg = f"Sampler draw archive does not exist: {path}"
        raise FileNotFoundError(msg)
    return load_sampler_result(path)


def _load_shock_samples(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    if not path.exists():
        msg = f"Shock sample archive does not exist: {path}"
        raise FileNotFoundError(msg)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        samples = np.asarray(np.load(path), dtype=np.float64)
    elif suffix == ".npz":
        with np.load(path) as archive:
            if "shock_samples" in archive.files:
                samples = np.asarray(archive["shock_samples"], dtype=np.float64)
            elif len(archive.files) == 1:
                samples = np.asarray(archive[archive.files[0]], dtype=np.float64)
            else:
                msg = "NPZ shock sample archive must contain shock_samples or exactly one array."
                raise ValueError(msg)
    elif suffix in {".h5", ".hdf5"}:
        import h5py

        samples = None
        with h5py.File(path, "r") as file:
            for dataset_name in ("shock_samples", "forecast_full/shock_samples"):
                if dataset_name in file:
                    samples = np.asarray(file[dataset_name], dtype=np.float64)
                    break
            if samples is None:
                dataset_names: list[str] = []

                def collect_dataset(name: str, item: object) -> None:
                    if isinstance(item, h5py.Dataset):
                        dataset_names.append(name)

                file.visititems(collect_dataset)
                if len(dataset_names) == 1:
                    samples = np.asarray(file[dataset_names[0]], dtype=np.float64)
        if samples is None:
            msg = (
                "HDF5 shock sample archive must contain shock_samples, "
                "forecast_full/shock_samples, or exactly one dataset."
            )
            raise ValueError(msg)
    else:
        msg = "Shock sample archive must be a .npy, .npz, .h5, or .hdf5 file."
        raise ValueError(msg)
    if samples.ndim != 3:
        msg = "Shock samples must be a 3D array."
        raise ValueError(msg)
    return samples


def _parse_number_list(value: str) -> np.ndarray:
    items = [item.strip() for item in value.split(",")]
    if not items or any(item == "" for item in items):
        msg = "Number list options must be comma-separated numeric values."
        raise ValueError(msg)
    try:
        return np.asarray([float(item) for item in items], dtype=np.float64)
    except ValueError as err:
        msg = "Number list options must be comma-separated numeric values."
        raise ValueError(msg) from err


app.add_typer(data_app, name="data")
app.add_typer(vv_app, name="vv")
