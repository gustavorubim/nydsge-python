# Migration-only benchmark producer for FRBNY-DSGE/DSGE.jl.
#
# This script is not part of the Python runtime. It runs the Julia oracle path
# and writes a JSON baseline file consumed by `nydsge bench --baseline`.

using Dates

const EXPORTER_PATH = joinpath(@__DIR__, "export_model1002.jl")
include(EXPORTER_PATH)

function parse_benchmark_args(args)
    options = Dict{String,String}(
        "out" => "tests/fixtures/oracle/julia_benchmark_model1002.json",
        "kernel" => "forecast",
        "subspec" => "ss10",
        "data-vintage" => "181115",
        "forecast-start" => "2018-Q4",
        "horizon" => "40",
        "repeats" => "3",
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key = arg[3:end]
            if i == length(args)
                error("Missing value for option $(arg)")
            end
            options[key] = args[i + 1]
            i += 2
        else
            error("Unexpected positional argument: $(arg)")
        end
    end
    return options
end

function benchmark_model1002(subspec::String, data_vintage::String, forecast_start::String)
    custom_settings = Dict{Symbol,Setting}(
        :data_vintage => Setting(:data_vintage, data_vintage),
        :date_forecast_start => Setting(:date_forecast_start, quartertodate(forecast_start)),
    )
    return Model1002(subspec; custom_settings = custom_settings)
end

function benchmark_forecast_entry(options)
    horizon = parse(Int, options["horizon"])
    repeats = parse(Int, options["repeats"])
    if horizon < 0
        error("Benchmark horizon must be nonnegative")
    end
    if repeats <= 0
        error("Benchmark repeats must be positive")
    end

    model = benchmark_model1002(
        options["subspec"],
        options["data-vintage"],
        options["forecast-start"],
    )
    model <= Setting(:forecast_horizons, horizon, "Number of periods to forecast ahead")
    system = compute_system(model)
    z0 = zeros(Float64, size(system[:TTT], 2))

    states, observables, pseudo_observables, shocks = forecast(
        model,
        system,
        z0;
        cond_type = :none,
        draw_shocks = false,
    )

    elapsed_samples = Float64[]
    for _ in 1:repeats
        elapsed = @elapsed begin
            states, observables, pseudo_observables, shocks = forecast(
                model,
                system,
                z0;
                cond_type = :none,
                draw_shocks = false,
            )
        end
        push!(elapsed_samples, elapsed)
    end

    return Dict{String,Any}(
        "name" => "julia-model1002-$(options["subspec"])-forecast",
        "backend" => "julia",
        "device" => "cpu",
        "kernel" => "forecast",
        "horizon" => horizon,
        "repeats" => repeats,
        "dtype" => "float64",
        "elapsed_seconds" => minimum(elapsed_samples),
        "mean_elapsed_seconds" => sum(elapsed_samples) / length(elapsed_samples),
        "elapsed_samples_seconds" => elapsed_samples,
        "states_shape" => collect(size(states)),
        "observables_shape" => collect(size(observables)),
        "pseudo_observables_shape" => collect(size(pseudo_observables)),
        "shocks_shape" => collect(size(shocks)),
    )
end

function json_string(value)
    escaped = replace(
        string(value),
        "\\" => "\\\\",
        "\"" => "\\\"",
        "\n" => "\\n",
        "\r" => "\\r",
        "\t" => "\\t",
    )
    return "\"" * escaped * "\""
end

json_value(value::AbstractString) = json_string(value)
json_value(value::Bool) = value ? "true" : "false"
json_value(value::Integer) = string(value)

function json_value(value::AbstractFloat)
    if !isfinite(value)
        error("Cannot write non-finite JSON float $(value)")
    end
    return string(value)
end

json_value(::Nothing) = "null"

function json_value(values::AbstractVector)
    return "[" * join([json_value(value) for value in values], ",") * "]"
end

function json_value(values::Tuple)
    return json_value(collect(values))
end

function json_value(values::AbstractDict)
    entries = String[]
    for (key, value) in values
        push!(entries, json_string(key) * ":" * json_value(value))
    end
    return "{" * join(entries, ",") * "}"
end

function write_json_report(path::String, report)
    mkpath(dirname(path))
    open(path, "w") do io
        write(io, json_value(report))
        write(io, "\n")
    end
end

function benchmark_model1002_main(args = ARGS)
    options = parse_benchmark_args(args)
    kernel = lowercase(strip(options["kernel"]))
    if !(kernel in ("forecast", "all"))
        error("Supported benchmark kernels are forecast and all")
    end

    results = Any[]
    if kernel in ("forecast", "all")
        push!(results, benchmark_forecast_entry(options))
    end

    report = Dict{String,Any}(
        "name" => "julia-model1002-$(options["subspec"])",
        "source" => "tools/oracle_julia/benchmark_model1002.jl",
        "exporter" => "tools/oracle_julia/export_model1002.jl",
        "model" => "Model1002",
        "subspec" => options["subspec"],
        "data_vintage" => options["data-vintage"],
        "forecast_start" => options["forecast-start"],
        "julia_version" => string(VERSION),
        "created_utc" => string(Dates.now(Dates.UTC)),
        "timing_notes" => (
            "Forecast timings exclude model construction and system solve, " *
            "include one untimed warmup, and report the minimum repeat as elapsed_seconds."
        ),
        "results" => results,
    )
    write_json_report(options["out"], report)
    println("Wrote Model1002 Julia benchmark baseline to $(options["out"])")
end

function is_benchmark_main_script()
    if isempty(PROGRAM_FILE)
        return false
    end
    return lowercase(normpath(abspath(PROGRAM_FILE))) == lowercase(normpath(abspath(@__FILE__)))
end

if is_benchmark_main_script()
    benchmark_model1002_main()
end
