# Migration-only oracle export for FRBNY-DSGE/DSGE.jl.
#
# Usage from a Julia environment with DSGE.jl available:
#
#   julia tools/oracle_julia/export_model1002.jl --out tests/fixtures/oracle/m1002_ss10.h5
#
# This script is not part of the Python runtime. It exists only to produce
# Python-readable fixtures for parity tests.

using Dates
using CSV
using DataFrames
using DSGE
using HDF5
using LinearAlgebra
using ModelConstructors
using Random
using Statistics

function parse_args(args)
    options = Dict{String,String}(
        "out" => "tests/fixtures/oracle/m1002_ss10.h5",
        "subspec" => "ss10",
        "data-vintage" => "181115",
        "forecast-start" => "2018-Q4",
        "horizon" => "40",
        "include-forecast" => "false",
        "include-full-forecast" => "false",
        "full-draws" => "0",
        "shock-samples-in" => "",
        "include-kalman" => "false",
        "include-posterior" => "false",
        "include-history" => "false",
        "include-financial-frictions" => "false",
        "include-sampler" => "false",
        "sampler-seed" => "",
        "sampler-draws" => "5000",
        "sampler-burnin" => "2",
        "sampler-blocks" => "22",
        "sampler-param-blocks" => "1",
        "sampler-thin" => "5",
        "sampler-adaptive-accept" => "false",
        "sampler-target-accept" => "0.25",
        "sampler-cc" => "0.09",
        "sampler-alpha" => "1.0",
        "sampler-c" => "0.5",
        "sampler-cc0" => "0.01",
        "sampler-calculate-hessian" => "true",
        "sampler-reoptimize" => "false",
        "sampler-run-csminwel" => "false",
        "sampler-proposal-scale" => "",
        "sampler-mode-in" => "",
        "sampler-hessian-in" => "",
        "data-in" => "",
        "data-out" => "",
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

function parse_bool(value::String)
    normalized = lowercase(strip(value))
    if normalized in ("true", "1", "yes", "y")
        return true
    elseif normalized in ("false", "0", "no", "n")
        return false
    end
    error("Expected boolean value, got $(value)")
end

function parse_int(value::String)
    return parse(Int, strip(value))
end

function parse_float(value::String)
    return parse(Float64, strip(value))
end

function parse_optional_int(value::String)
    text = strip(value)
    if isempty(text)
        return nothing
    end
    return parse(Int, text)
end

function parse_optional_float(value::String)
    text = strip(value)
    if isempty(text)
        return nothing
    end
    return parse(Float64, text)
end

function write_sampler_dataset(file, name, value)
    if haskey(file, name)
        delete_object(file, name)
    end
    file[name] = Array(value)
end

function write_dataset(file, name, value)
    if haskey(file, name)
        delete_object(file, name)
    end
    array = Array(value)
    file[name] = ndims(array) == 2 ? permutedims(array) : array
end

function write_forecast_dataset(file, name, value)
    if haskey(file, name)
        delete_object(file, name)
    end
    file[name] = Array(value)
end

function write_file_attribute(file, name::String, value)
    if name in keys(attributes(file))
        delete_attribute(file, name)
    end
    write_attribute(file, name, value)
end

function maybe_property(object, names; default = nothing)
    for name in names
        if hasproperty(object, name)
            return getproperty(object, name)
        end
        if name in fieldnames(typeof(object))
            return getfield(object, name)
        end
    end
    return default
end

function maybe_constructor_function(source_module, names, object; default = nothing)
    for name in names
        if isdefined(source_module, name)
            fn = getfield(source_module, name)
            try
                return fn(object)
            catch
            end
        end
    end
    return default
end

function call_if_defined(source_module, names, object)
    for name in names
        if isdefined(source_module, name)
            fn = getfield(source_module, name)
            try
                fn(object)
                return
            catch
            end
        end
    end
end

function parameter_name(item, parameter)
    if item isa Pair
        return string(item.first)
    end
    value = maybe_property(parameter, (:key, :name, :symbol); default = nothing)
    if value === nothing
        error("Could not determine parameter name for $(parameter)")
    end
    return string(value)
end

function parameter_value(parameter)
    value = maybe_property(parameter, (:value, :val); default = nothing)
    if value === nothing
        value = maybe_constructor_function(ModelConstructors, (:value, :get_value), parameter)
    end
    if value === nothing
        error("Could not determine parameter value for $(parameter)")
    end
    return Float64(value)
end

function parameter_scaled_value(parameter)
    value = maybe_constructor_function(
        ModelConstructors,
        (:scaledvalue, :scaled_value, :get_scaled_value),
        parameter,
    )
    if value === nothing
        value = maybe_property(parameter, (:scaled_value, :scaledvalue); default = nothing)
    end
    if value === nothing
        value = parameter_value(parameter)
    end
    return Float64(value)
end

function parameter_fixed_bool(parameter)
    regimes = maybe_property(parameter, (:regimes,); default = nothing)
    if regimes !== nothing && haskey(regimes, :fixed)
        return all(value -> Bool(value), values(regimes[:fixed]))
    end
    value = maybe_property(parameter, (:fixed, :is_fixed); default = false)
    return Bool(value)
end

function parameter_fixed(parameter)
    return parameter_fixed_bool(parameter) ? 1.0 : 0.0
end

function parameter_bounds(parameter)
    bounds = maybe_property(
        parameter,
        (:valuebounds, :value_bounds, :bounds);
        default = nothing,
    )
    if bounds === nothing
        return (NaN, NaN)
    end
    return (Float64(bounds[1]), Float64(bounds[2]))
end

function ordered_mapping_names(mapping)
    items = collect(mapping)
    if all(item -> item isa Pair, items)
        sorted_items = sort(items; by = item -> item.second)
        return [string(item.first) for item in sorted_items]
    end
    return [parameter_name(item, item) for item in items]
end

function quarter_label(date)
    quarter = ((Dates.month(date) - 1) ÷ 3) + 1
    return "$(Dates.year(date))-Q$(quarter)"
end

function forecast_date_labels(forecast_start::String, horizon::Int)
    if horizon == 0
        return String[]
    end
    start_date = quartertodate(forecast_start)
    return [quarter_label(start_date + Dates.Month(3 * offset)) for offset in 0:(horizon - 1)]
end

function export_label_attributes(file, model)
    endogenous_state_names = ordered_mapping_names(model.endogenous_states)
    augmented_state_names = ordered_mapping_names(model.endogenous_states_augmented)
    write_file_attribute(file, "endogenous_state_names", join(endogenous_state_names, ","))
    write_file_attribute(file, "augmented_state_names", join(augmented_state_names, ","))
    write_file_attribute(file, "state_names", join(
        [endogenous_state_names; augmented_state_names],
        ",",
    ))
    write_file_attribute(file, "exogenous_shock_names", join(
        ordered_mapping_names(model.exogenous_shocks),
        ",",
    ))
    write_file_attribute(file, "expected_shock_names", join(
        ordered_mapping_names(model.expected_shocks),
        ",",
    ))
    write_file_attribute(file, "equation_names", join(
        ordered_mapping_names(model.equilibrium_conditions),
        ",",
    ))
    write_file_attribute(file, "observable_names", join(ordered_mapping_names(model.observables), ","))
    write_file_attribute(file, "pseudo_observable_names", join(
        ordered_mapping_names(model.pseudo_observables),
        ",",
    ))
end

function export_mode_forecast(
    file,
    model;
    forecast_start::String,
    horizon::Int,
    initial_state = nothing,
)
    if horizon < 0
        error("Forecast horizon must be nonnegative")
    end
    model <= Setting(:forecast_horizons, horizon, "Number of periods to forecast ahead")
    system = compute_system(model)
    z0 = initial_state === nothing ? zeros(Float64, size(system[:TTT], 2)) : initial_state
    states, observables, pseudo_observables, shocks = forecast(
        model,
        system,
        z0;
        cond_type = :none,
        draw_shocks = false,
    )
    write_file_attribute(file, "forecast_mode_dates", join(
        forecast_date_labels(forecast_start, horizon),
        ",",
    ))
    write_forecast_dataset(file, "forecast_mode/states", states)
    write_forecast_dataset(file, "forecast_mode/observables", observables)
    write_forecast_dataset(file, "forecast_mode/pseudo_observables", pseudo_observables)
    write_forecast_dataset(file, "forecast_mode/shocks", shocks)
    write_forecast_dataset(file, "meansbands_mode_forecastobs/mean", observables)
    write_forecast_dataset(file, "meansbands_mode_forecastobs/lower", observables)
    write_forecast_dataset(file, "meansbands_mode_forecastobs/upper", observables)
end

function read_hdf5_shock_samples(path::String)
    available_keys = String[]
    result = h5open(path, "r") do sample_file
        available_keys = sort(string.(collect(keys(sample_file))))
        if any(key -> key == "shock_samples", available_keys)
            return read(sample_file["shock_samples"])
        end
        if any(key -> key == "forecast_full", available_keys)
            group = sample_file["forecast_full"]
            group_keys = string.(collect(keys(group)))
            if any(key -> key == "shock_samples", group_keys)
                return read(group["shock_samples"])
            end
        end
        return nothing
    end
    if result !== nothing
        return result
    end
    error(
        "HDF5 shock sample archive $(path) must contain shock_samples or " *
        "forecast_full/shock_samples; available top-level keys: " *
        join(available_keys, ","),
    )
end

function normalize_shock_samples(raw_samples; nshocks::Int, horizon::Int)
    sample_array = Array{Float64}(raw_samples)
    if ndims(sample_array) != 3
        error("Shock samples must be a 3D array")
    end
    if size(sample_array, 1) == nshocks
        shocks_by_periods_by_draws = sample_array
    elseif size(sample_array, 3) == nshocks
        shocks_by_periods_by_draws = permutedims(sample_array, (3, 2, 1))
    elseif size(sample_array, 2) == nshocks
        shocks_by_periods_by_draws = permutedims(sample_array, (2, 3, 1))
    else
        error(
            "Shock samples must have one non-draw dimension equal to nshocks=$(nshocks)",
        )
    end
    draws = size(shocks_by_periods_by_draws, 3)
    normalized = zeros(Float64, nshocks, horizon, draws)
    copied_periods = min(horizon, size(shocks_by_periods_by_draws, 2))
    normalized[:, 1:copied_periods, :] =
        shocks_by_periods_by_draws[:, 1:copied_periods, :]
    return normalized
end

function full_forecast_shock_samples(model; horizon::Int, full_draws::Int, shock_samples_in::String)
    nshocks = n_shocks_exogenous(model)
    if isempty(shock_samples_in)
        if full_draws <= 0
            error("--include-full-forecast true requires --full-draws or --shock-samples-in")
        end
        return zeros(Float64, nshocks, horizon, full_draws)
    end
    shock_samples = normalize_shock_samples(
        read_hdf5_shock_samples(shock_samples_in);
        nshocks = nshocks,
        horizon = horizon,
    )
    if full_draws > 0 && full_draws != size(shock_samples, 3)
        error(
            "--full-draws must match explicit shock sample draws: " *
            "$(full_draws) != $(size(shock_samples, 3))",
        )
    end
    return shock_samples
end

function sample_quantile(sample_cube, probability::Float64)
    output = zeros(Float64, size(sample_cube, 1), size(sample_cube, 2))
    for variable in axes(sample_cube, 1)
        for period in axes(sample_cube, 2)
            output[variable, period] = quantile(vec(sample_cube[variable, period, :]), probability)
        end
    end
    return output
end

function export_full_forecast(
    file,
    model;
    forecast_start::String,
    horizon::Int,
    full_draws::Int,
    shock_samples_in::String,
    history_observables = nothing,
    initial_state = nothing,
)
    if horizon < 0
        error("Forecast horizon must be nonnegative")
    end
    model <= Setting(:forecast_horizons, horizon, "Number of periods to forecast ahead")
    system = compute_system(model)
    z0 = initial_state === nothing ? zeros(Float64, size(system[:TTT], 2)) : initial_state
    shock_samples = full_forecast_shock_samples(
        model;
        horizon = horizon,
        full_draws = full_draws,
        shock_samples_in = shock_samples_in,
    )
    draws = size(shock_samples, 3)
    nstates = size(system[:TTT], 1)
    nobs = size(system[:ZZ], 1)
    npseudo = size(system[:ZZ_pseudo], 1)
    nshocks = size(shock_samples, 1)
    state_samples = zeros(Float64, nstates, horizon, draws)
    observable_samples = zeros(Float64, nobs, horizon, draws)
    pseudo_observable_samples = zeros(Float64, npseudo, horizon, draws)
    used_shock_samples = zeros(Float64, nshocks, horizon, draws)
    for draw in 1:draws
        states, observables, pseudo_observables, shocks = forecast(
            model,
            system,
            z0;
            cond_type = :none,
            shocks = shock_samples[:, :, draw],
        )
        state_samples[:, :, draw] = states
        observable_samples[:, :, draw] = observables
        pseudo_observable_samples[:, :, draw] = pseudo_observables
        used_shock_samples[:, :, draw] = shocks
    end
    states = dropdims(mean(state_samples; dims = 3); dims = 3)
    observables = dropdims(mean(observable_samples; dims = 3); dims = 3)
    pseudo_observables = dropdims(mean(pseudo_observable_samples; dims = 3); dims = 3)
    write_file_attribute(file, "forecast_full_dates", join(
        forecast_date_labels(forecast_start, horizon),
        ",",
    ))
    write_file_attribute(file, "forecast_full_draws", string(draws))
    write_forecast_dataset(file, "forecast_full/states", states)
    write_forecast_dataset(file, "forecast_full/observables", observables)
    write_forecast_dataset(file, "forecast_full/pseudo_observables", pseudo_observables)
    write_forecast_dataset(file, "forecast_full/state_samples", state_samples)
    write_forecast_dataset(file, "forecast_full/observable_samples", observable_samples)
    write_forecast_dataset(
        file,
        "forecast_full/pseudo_observable_samples",
        pseudo_observable_samples,
    )
    write_forecast_dataset(file, "forecast_full/shock_samples", used_shock_samples)
    write_forecast_dataset(file, "meansbands_full_forecastobs/mean", observables)
    write_forecast_dataset(
        file,
        "meansbands_full_forecastobs/lower",
        sample_quantile(observable_samples, 0.05),
    )
    write_forecast_dataset(
        file,
        "meansbands_full_forecastobs/upper",
        sample_quantile(observable_samples, 0.95),
    )
    if history_observables !== nothing
        history_samples = repeat(
            reshape(
                history_observables,
                size(history_observables, 1),
                size(history_observables, 2),
                1,
            ),
            1,
            1,
            draws,
        )
        write_forecast_dataset(
            file,
            "forecast_full/history_observables",
            history_observables,
        )
        write_forecast_dataset(
            file,
            "forecast_full/history_observable_samples",
            history_samples,
        )
        write_forecast_dataset(file, "meansbands_full_histobs/mean", history_observables)
        write_forecast_dataset(file, "meansbands_full_histobs/lower", history_observables)
        write_forecast_dataset(file, "meansbands_full_histobs/upper", history_observables)
    end
end

function coerce_quarter_date(value)
    if value isa Dates.Date
        return Dates.lastdayofquarter(value)
    end
    text = String(strip(string(value)))
    if occursin(r"^[0-9]{4}-[qQ][1-4]$", text) || occursin(r"^[0-9]{4}[qQ][1-4]$", text)
        return quartertodate(text)
    end
    return Dates.lastdayofquarter(Dates.Date(text))
end

function load_or_read_history_data(model; data_in::String = "")
    if isempty(data_in)
        return load_data(
            model;
            try_disk = true,
            check_empty_columns = false,
            summary_statistics = :none,
            verbose = :none,
        )
    end
    df = CSV.read(data_in, DataFrame, copycols = true)
    if :date in propertynames(df)
        df[!, :date] = coerce_quarter_date.(df[!, :date])
    end
    return df
end

function export_history_observables(file, model; data_in::String = "", data_out::String = "")
    df = load_or_read_history_data(model; data_in = data_in)
    if !isempty(data_out)
        mkpath(dirname(data_out))
        CSV.write(data_out, df)
    end
    history_df = df[date_mainsample_start(model) .<= df[!, :date] .<= date_mainsample_end(model), :]
    history_observables = df_to_matrix(model, history_df)
    write_file_attribute(file, "history_dates", join(
        [quarter_label(date) for date in history_df[!, :date]],
        ",",
    ))
    write_forecast_dataset(file, "forecast_mode/history_observables", history_observables)
    write_forecast_dataset(file, "meansbands_mode_histobs/mean", history_observables)
    write_forecast_dataset(file, "meansbands_mode_histobs/lower", history_observables)
    write_forecast_dataset(file, "meansbands_mode_histobs/upper", history_observables)
    return history_observables, history_df
end

function filtered_forecast_start_state(model, system; data_in::String = "", history_df = nothing)
    if history_df === nothing
        if isempty(data_in)
            return nothing
        end
        df = load_or_read_history_data(model; data_in = data_in)
        history_df = df[
            date_mainsample_start(model) .<= df[!, :date] .<= date_mainsample_end(model),
            :,
        ]
    end
    if nrow(history_df) == 0
        return nothing
    end
    kal = DSGE.filter(model, history_df, system; cond_type = :none)
    return kal[:s_T]
end

function export_kalman(file, model, system; data_in::String = "")
    df = load_or_read_history_data(model; data_in = data_in)
    history_df = df[date_mainsample_start(model) .<= df[!, :date] .<= date_mainsample_end(model), :]
    if nrow(history_df) == 0
        error("Kalman export requires at least one in-sample data row")
    end
    kal = DSGE.filter(model, history_df, system; cond_type = :none)
    write_file_attribute(file, "history_dates", join(
        [quarter_label(date) for date in history_df[!, :date]],
        ",",
    ))
    write_forecast_dataset(file, "kalman/log_likelihood", kal[:loglh])
    write_forecast_dataset(file, "kalman/predicted_states", kal[:s_pred])
    write_forecast_dataset(file, "kalman/filtered_states", kal[:s_filt])
    write_forecast_dataset(file, "kalman/predicted_covariances", kal[:P_pred])
    write_forecast_dataset(file, "kalman/filtered_covariances", kal[:P_filt])
    write_forecast_dataset(file, "kalman/final_filtered_state", kal[:s_T])
    write_forecast_dataset(file, "kalman/total_log_likelihood", [kal[:total_loglh]])
end

function export_posterior(file, model, system; data_in::String = "")
    df = load_or_read_history_data(model; data_in = data_in)
    history_df = df[date_mainsample_start(model) .<= df[!, :date] .<= date_mainsample_end(model), :]
    if nrow(history_df) == 0
        error("Posterior export requires at least one in-sample data row")
    end
    kal = DSGE.filter(model, history_df, system; cond_type = :none)
    log_likelihood = Float64(kal[:total_loglh])
    log_prior = Float64(prior(model))
    log_posterior = log_likelihood + log_prior

    items = collect(model.parameters)
    values = Float64[]
    prior_contributions = Float64[]
    for item in items
        parameter = item isa Pair ? item.second : item
        push!(values, parameter_value(parameter))
        if parameter_fixed(parameter) == 1.0 || !ModelConstructors.hasprior(parameter)
            push!(prior_contributions, 0.0)
        else
            push!(prior_contributions, Float64(ModelConstructors.logpdf(parameter)))
        end
    end

    write_file_attribute(file, "history_dates", join(
        [quarter_label(date) for date in history_df[!, :date]],
        ",",
    ))
    write_forecast_dataset(file, "posterior/log_posterior", [log_posterior])
    write_forecast_dataset(file, "posterior/log_likelihood", [log_likelihood])
    write_forecast_dataset(file, "posterior/log_prior", [log_prior])
    write_forecast_dataset(file, "posterior/log_likelihood_by_period", kal[:loglh])
    write_forecast_dataset(file, "posterior/log_prior_by_parameter", prior_contributions)
    write_forecast_dataset(file, "posterior/parameter_values", values)
end

function export_parameters(file, model)
    items = collect(model.parameters)
    names = String[]
    values = Float64[]
    scaled_values = Float64[]
    fixed = Float64[]
    bounds = fill(NaN, length(items), 2)
    for (index, item) in enumerate(items)
        parameter = item isa Pair ? item.second : item
        push!(names, parameter_name(item, parameter))
        push!(values, parameter_value(parameter))
        push!(scaled_values, parameter_scaled_value(parameter))
        push!(fixed, parameter_fixed(parameter))
        lower, upper = parameter_bounds(parameter)
        bounds[index, 1] = lower
        bounds[index, 2] = upper
    end
    write_file_attribute(file, "parameter_names", join(names, ","))
    write_dataset(file, "parameters/values", values)
    write_dataset(file, "parameters/scaled_values", scaled_values)
    write_dataset(file, "parameters/fixed", fixed)
    write_dataset(file, "parameters/bounds", bounds)
end

function steady_state_name(item, value)
    if item isa Pair
        return string(item.first)
    end
    name = maybe_property(value, (:key, :name, :symbol); default = nothing)
    if name === nothing
        error("Could not determine steady-state name for $(value)")
    end
    return string(name)
end

function steady_state_value(value)
    if value isa Number
        return Float64(value)
    end
    raw = maybe_property(value, (:value, :val); default = nothing)
    if raw === nothing
        raw = maybe_constructor_function(ModelConstructors, (:value, :get_value), value)
    end
    if raw === nothing
        error("Could not determine steady-state value for $(value)")
    end
    return Float64(raw)
end

function export_steady_state(file, model)
    call_if_defined(DSGE, (Symbol("steadystate!"), :steadystate), model)
    steady_state = maybe_property(
        model,
        (:steady_state, :steady_state_values, :steadystate);
        default = nothing,
    )
    if steady_state === nothing
        @warn "Steady-state values were not exported because no steady-state container was found"
        return
    end
    items = collect(steady_state)
    names = String[]
    values = Float64[]
    for item in items
        value = item isa Pair ? item.second : item
        push!(names, steady_state_name(item, value))
        push!(values, steady_state_value(value))
    end
    write_file_attribute(file, "steady_state_names", join(names, ","))
    write_dataset(file, "steady_state/values", values)
end

function export_financial_frictions(file)
    input_names = ["z", "sigma", "spr"]
    case_names = ["default", "lower_sigma", "wide_spread"]
    function_names = [
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
    ]
    inputs = [
        -2.42825276274453 0.5 (1.0 + 1.7444 / 100.0)^0.25;
        -2.42825276274453 0.45 (1.0 + 1.7444 / 100.0)^0.25;
        -2.1 0.6 (1.0 + 2.5 / 100.0)^0.25;
    ]
    values = Matrix{Float64}(undef, size(inputs, 1), length(function_names))
    omega_fn = getfield(DSGE, Symbol("ω_fn"))
    g_fn = getfield(DSGE, Symbol("G_fn"))
    gamma_fn = getfield(DSGE, Symbol("Γ_fn"))
    dg_domega_fn = getfield(DSGE, Symbol("dG_dω_fn"))
    d2g_domega2_fn = getfield(DSGE, Symbol("d2G_dω2_fn"))
    dgamma_domega_fn = getfield(DSGE, Symbol("dΓ_dω_fn"))
    d2gamma_domega2_fn = getfield(DSGE, Symbol("d2Γ_dω2_fn"))
    dg_dsigma_fn = getfield(DSGE, Symbol("dG_dσ_fn"))
    d2g_domega_dsigma_fn = getfield(DSGE, Symbol("d2G_dωdσ_fn"))
    dgamma_dsigma_fn = getfield(DSGE, Symbol("dΓ_dσ_fn"))
    d2gamma_domega_dsigma_fn = getfield(DSGE, Symbol("d2Γ_dωdσ_fn"))
    mu_fn = getfield(DSGE, Symbol("μ_fn"))
    nk_fn = getfield(DSGE, Symbol("nk_fn"))
    zeta_bomega_fn = getfield(DSGE, Symbol("ζ_bω_fn"))
    zeta_zomega_fn = getfield(DSGE, Symbol("ζ_zω_fn"))
    zeta_spb_fn = getfield(DSGE, Symbol("ζ_spb_fn"))

    for row in axes(inputs, 1)
        z = inputs[row, 1]
        sigma = inputs[row, 2]
        spr = inputs[row, 3]
        values[row, :] = [
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
    end

    write_file_attribute(file, "financial_frictions_input_names", join(input_names, ","))
    write_file_attribute(file, "financial_frictions_case_names", join(case_names, ","))
    write_file_attribute(
        file,
        "financial_frictions_function_names",
        join(function_names, ","),
    )
    write_dataset(file, "financial_frictions/inputs", inputs)
    write_dataset(file, "financial_frictions/values", values)
end

function metropolis_hastings_with_trace(
    proposal_dist,
    model,
    data::Matrix{Float64},
    cc0::Float64,
    cc::Float64;
    n_blocks::Int = 1,
    n_param_blocks::Int = 1,
    n_sim::Int = 100,
    n_burn::Int = 0,
    mhthin::Int = 1,
    savepath::String = "mhsave.h5",
    rng = MersenneTwister(0),
    regime_switching::Bool = false,
    toggle::Bool = true,
)
    if regime_switching
        error("Sampler trace export does not yet support regime-switching parameters.")
    end
    if n_blocks <= n_burn
        error("Sampler trace export requires n_blocks > n_burn.")
    end

    mu_symbol = Symbol(string(Char(0x03bc)))
    sigma_symbol = Symbol(string(Char(0x03c3)))
    lambda_symbol = Symbol(string(Char(0x03bb)), "_vals")
    propdist = DSGE.init_deg_mvnormal(
        getfield(proposal_dist, mu_symbol),
        getfield(proposal_dist, sigma_symbol),
    )
    parameters = DSGE.get_parameters(model)
    use_chand_recursion = !any(isnan.(data)) && !regime_switching

    function loglikelihood(parameter_vector, data_matrix::Matrix{Float64})::Float64
        DSGE.update!(
            model,
            parameter_vector;
            regime_switching = regime_switching,
            toggle = toggle,
        )
        return DSGE.likelihood(
            model,
            data_matrix;
            sampler = true,
            catch_errors = false,
            use_chand_recursion = use_chand_recursion,
        )
    end

    function trace_parameter_is_fixed(parameter)
        return parameter_fixed_bool(parameter)
    end

    function trace_log_prior(parameters, values::Vector{Float64})::Float64
        if length(values) != length(parameters)
            error("Sampler trace prior vector has the wrong parameter count.")
        end

        total = 0.0
        for i = 1:length(parameters)
            parameter = parameters[i]
            if !trace_parameter_is_fixed(parameter) && ModelConstructors.hasprior(parameter)
                total += ModelConstructors.logpdf(ModelConstructors.parameter(parameter, values[i]))
            end
        end
        return total
    end

    para_old = rand(propdist, rng; cc = cc0)
    post_old = -Inf
    initialized = false
    while !initialized
        post_old = ModelConstructors.posterior!(
            loglikelihood,
            parameters,
            para_old,
            data;
            sampler = true,
        )
        if post_old > -Inf
            setfield!(propdist, mu_symbol, para_old)
            initialized = true
        else
            para_old = rand(propdist, rng; cc = cc0)
        end
    end

    free_para_inds = ModelConstructors.get_free_para_inds(
        parameters;
        regime_switching = regime_switching,
        toggle = toggle,
    )
    n_params = length(parameters)
    if n_param_blocks == 1
        blocks_free = Vector{Int}[free_para_inds]
        reblock = false
    else
        n_free_para = length(free_para_inds)
        blocks_free = Vector{Int}[]
        reblock = true
    end

    rows_per_block = n_sim * n_param_blocks
    mhparams = zeros(rows_per_block, n_params)
    mhaccepted = zeros(Int8, rows_per_block)
    mhlogpost = fill(-Inf, rows_per_block)
    mhproposal = zeros(rows_per_block, n_params)
    mhprevious = zeros(rows_per_block, n_params)
    mhproposal_logpost = fill(-Inf, rows_per_block)
    mhprevious_logpost = fill(-Inf, rows_per_block)
    mhproposal_loglikelihood = fill(-Inf, rows_per_block)
    mhprevious_loglikelihood = fill(-Inf, rows_per_block)
    mhproposal_logprior = fill(-Inf, rows_per_block)
    mhprevious_logprior = fill(-Inf, rows_per_block)
    mhuniform = fill(NaN, rows_per_block)
    mhlog_acceptance = fill(-Inf, rows_per_block)
    block_acceptance_rates = zeros(Float64, n_blocks)
    all_rejections = 0

    simfile = h5open(savepath, "w")
    n_saved_obs = rows_per_block * (n_blocks - n_burn)
    saved_accepted = zeros(Int8, n_saved_obs)
    saved_logpost = fill(-Inf, n_saved_obs)
    saved_proposal = zeros(n_saved_obs, n_params)
    saved_previous = zeros(n_saved_obs, n_params)
    saved_proposal_logpost = fill(-Inf, n_saved_obs)
    saved_previous_logpost = fill(-Inf, n_saved_obs)
    saved_proposal_loglikelihood = fill(-Inf, n_saved_obs)
    saved_previous_loglikelihood = fill(-Inf, n_saved_obs)
    saved_proposal_logprior = fill(-Inf, n_saved_obs)
    saved_previous_logprior = fill(-Inf, n_saved_obs)
    saved_uniform = fill(NaN, n_saved_obs)
    saved_log_acceptance = fill(-Inf, n_saved_obs)
    parasim = isdefined(HDF5, :create_dataset) ?
        HDF5.create_dataset(
            simfile,
            "mhparams",
            datatype(Float64),
            dataspace(n_saved_obs, n_params);
            chunk = (rows_per_block, n_params),
        ) :
        HDF5.d_create(
            simfile,
            "mhparams",
            datatype(Float64),
            dataspace(n_saved_obs, n_params),
            "chunk",
            (rows_per_block, n_params),
        )

    try
        for block = 1:n_blocks
            block_rejections = 0

            for j = 1:(n_sim * mhthin)
                if reblock
                    blocks_free = DSGE.SMC.generate_free_blocks(n_free_para, n_param_blocks)
                    for block_f in blocks_free
                        sort!(block_f)
                    end
                end

                for (k, block_a) in enumerate(blocks_free)
                    para_previous = deepcopy(para_old)
                    post_previous = post_old
                    para_subset = para_old[block_a]
                    proposal_mu = getfield(propdist, mu_symbol)
                    proposal_sigma = getfield(propdist, sigma_symbol)
                    proposal_lambda = getfield(propdist, lambda_symbol)
                    subset_sigma = (
                        proposal_sigma[block_a, block_a] +
                        proposal_sigma[block_a, block_a]'
                    ) / 2.0
                    d_subset = DegenerateMvNormal(
                        proposal_mu[block_a],
                        subset_sigma,
                        inv(subset_sigma),
                        proposal_lambda[block_a],
                    )

                    para_draw = rand(d_subset, rng; cc = cc)
                    para_new = deepcopy(para_old)
                    para_new[block_a] = para_draw
                    post_new = ModelConstructors.posterior!(
                        loglikelihood,
                        parameters,
                        para_new,
                        data;
                        sampler = true,
                    )
                    prior_new = trace_log_prior(parameters, para_new)
                    prior_previous = trace_log_prior(parameters, para_previous)

                    accepted_step = false
                    log_acceptance = post_new - post_old
                    r = exp(log_acceptance)
                    x = rand(rng)
                    if x < min(1.0, r)
                        para_old = para_new
                        post_old = post_new
                        setfield!(propdist, mu_symbol, para_new)
                        accepted_step = true
                    else
                        block_rejections += 1
                    end

                    if j % mhthin == 0
                        draw_index = convert(Int, ((j / mhthin) - 1) * n_param_blocks + k)
                        mhparams[draw_index, :] = para_old'
                        mhaccepted[draw_index] = accepted_step ? Int8(1) : Int8(0)
                        mhlogpost[draw_index] = post_old
                        mhproposal[draw_index, :] = para_new'
                        mhprevious[draw_index, :] = para_previous'
                        mhproposal_logpost[draw_index] = post_new
                        mhprevious_logpost[draw_index] = post_previous
                        mhproposal_logprior[draw_index] = prior_new
                        mhprevious_logprior[draw_index] = prior_previous
                        mhproposal_loglikelihood[draw_index] = post_new - prior_new
                        mhprevious_loglikelihood[draw_index] = post_previous - prior_previous
                        mhuniform[draw_index] = x
                        mhlog_acceptance[draw_index] = log_acceptance
                    end
                end
            end

            all_rejections += block_rejections
            block_acceptance_rates[block] =
                1.0 - block_rejections / (n_sim * mhthin * n_param_blocks)

            block_start = rows_per_block * (block - n_burn - 1) + 1
            block_end = block_start + rows_per_block - 1
            if block > n_burn
                parasim[block_start:block_end, :] = map(Float64, mhparams)
                saved_accepted[block_start:block_end] = mhaccepted
                saved_logpost[block_start:block_end] = mhlogpost
                saved_proposal[block_start:block_end, :] = mhproposal
                saved_previous[block_start:block_end, :] = mhprevious
                saved_proposal_logpost[block_start:block_end] = mhproposal_logpost
                saved_previous_logpost[block_start:block_end] = mhprevious_logpost
                saved_proposal_loglikelihood[block_start:block_end] = mhproposal_loglikelihood
                saved_previous_loglikelihood[block_start:block_end] = mhprevious_loglikelihood
                saved_proposal_logprior[block_start:block_end] = mhproposal_logprior
                saved_previous_logprior[block_start:block_end] = mhprevious_logprior
                saved_uniform[block_start:block_end] = mhuniform
                saved_log_acceptance[block_start:block_end] = mhlog_acceptance
            end
        end

        simfile["accepted"] = saved_accepted
        simfile["log_posterior"] = saved_logpost
        simfile["proposal_parameters"] = saved_proposal
        simfile["previous_parameters"] = saved_previous
        simfile["proposal_log_posterior"] = saved_proposal_logpost
        simfile["previous_log_posterior"] = saved_previous_logpost
        simfile["proposal_log_likelihood"] = saved_proposal_loglikelihood
        simfile["previous_log_likelihood"] = saved_previous_loglikelihood
        simfile["proposal_log_prior"] = saved_proposal_logprior
        simfile["previous_log_prior"] = saved_previous_logprior
        simfile["uniform_draw"] = saved_uniform
        simfile["log_acceptance"] = saved_log_acceptance

        write_attribute(
            simfile,
            "acceptance_rate",
            string(1.0 - all_rejections / (n_blocks * n_sim * mhthin * n_param_blocks)),
        )
        write_attribute(
            simfile,
            "block_acceptance_rates",
            join(string.(block_acceptance_rates), ","),
        )
    finally
        close(simfile)
    end
end

function export_sampler(
    file,
    model;
    data_in::String = "",
    data_out::String = "",
    draws::Int = 5000,
    burnin::Int = 2,
    blocks::Int = 22,
    param_blocks::Int = 1,
    thin::Int = 5,
    adaptive_accept::Bool = false,
    target_accept::Float64 = 0.25,
    c::Float64 = 0.5,
    cc::Float64 = 0.09,
    cc0::Float64 = 0.01,
    alpha::Float64 = 1.0,
    calculate_hessian::Bool = true,
    reoptimize::Bool = false,
    run_csminwel::Bool = false,
    proposal_scale::Union{Float64,Nothing} = nothing,
    mode_in::String = "",
    hessian_in::String = "",
    seed::Union{Int,Nothing} = nothing,
)
    history_data = load_or_read_history_data(model; data_in = data_in)
    if !isempty(data_out)
        mkpath(dirname(data_out))
        CSV.write(data_out, history_data)
    end
    model <= Setting(:sampling_method, :MH)
    model <= Setting(:calculate_hessian, calculate_hessian)
    model <= Setting(:reoptimize, reoptimize)
    model <= Setting(:n_mh_simulations, draws, "Metropolis-Hastings draws")
    model <= Setting(:n_mh_blocks, blocks, "Metropolis-Hastings block count")
    model <= Setting(:n_mh_param_blocks, param_blocks, "Metropolis-Hastings parameter blocks")
    model <= Setting(:n_mh_burn, burnin, "Metropolis-Hastings burn-in blocks")
    model <= Setting(:mh_thin, thin, "Metropolis-Hastings thinning")
    model <= Setting(:mh_adaptive_accept, adaptive_accept, "Adaptive MH proposal acceptance")
    model <= Setting(:mh_target_accept, target_accept, "Target MH acceptance rate")
    model <= Setting(:mh_cc, cc, "Metropolis-Hastings cc parameter")
    model <= Setting(Symbol("mh_", string(Char(0x03b1))), alpha, "Metropolis-Hastings alpha parameter")
    model <= Setting(:mh_c, c, "Metropolis-Hastings c parameter")
    model <= Setting(:mh_cc0, cc0, "Metropolis-Hastings cc0 parameter")
    if !isempty(mode_in)
        specify_mode!(model, mode_in; verbose = :none)
    end
    if !isempty(hessian_in)
        specify_hessian!(model, hessian_in; verbose = :none)
    end
    if seed !== nothing
        model.rng = MersenneTwister(seed)
    end

    input_proposal_covariance = Matrix{Float64}(undef, 0, 0)
    if proposal_scale !== nothing
        n_params = length(DSGE.get_parameters(model))
        input_proposal_covariance = Matrix{Float64}(I, n_params, n_params) .* proposal_scale
    end

    if proposal_scale !== nothing
        data_matrix = history_data isa DataFrame ? df_to_matrix(model, history_data) : Matrix{Float64}(history_data)
        params = ModelConstructors.get_values(
            DSGE.get_parameters(model);
            regime_switching = false,
        )
        propdist = DegenerateMvNormal(params, input_proposal_covariance)
        if adaptive_accept
            @warn "Sampler trace export is skipped when adaptive_accept=true."
            metropolis_hastings(propdist, model, data_matrix, cc0, cc; verbose = :none)
        else
            metropolis_hastings_with_trace(
                propdist,
                model,
                data_matrix,
                cc0,
                cc;
                n_blocks = blocks,
                n_param_blocks = param_blocks,
                n_sim = draws,
                n_burn = burnin,
                mhthin = thin,
                savepath = rawpath(model, "estimate", "mhsave.h5"),
                rng = model.rng,
            )
        end
        compute_parameter_covariance(model)
    else
        estimate(
            model,
            history_data;
            old_data = Matrix{Float64}(undef, size(history_data, 1), 0),
            run_csminwel = run_csminwel,
            sampling = true,
            verbose = :none,
        )
    end

    sampler_accepted = Int8[]
    sampler_log_posterior = Float64[]
    sampler_proposal_parameters = Matrix{Float64}(undef, 0, 0)
    sampler_previous_parameters = Matrix{Float64}(undef, 0, 0)
    sampler_proposal_log_posterior = Float64[]
    sampler_previous_log_posterior = Float64[]
    sampler_proposal_log_likelihood = Float64[]
    sampler_previous_log_likelihood = Float64[]
    sampler_proposal_log_prior = Float64[]
    sampler_previous_log_prior = Float64[]
    sampler_uniform_draw = Float64[]
    sampler_log_acceptance = Float64[]
    sampler_acceptance_rate = ""
    sampler_block_acceptance_rates = ""
    sampler_fixed = [parameter_fixed(parameter) for parameter in DSGE.get_parameters(model)]
    mhparams = h5open(rawpath(model, "estimate", "mhsave.h5"), "r") do handle
        if haskey(handle, "accepted")
            sampler_accepted = vec(read(handle["accepted"]))
        end
        if haskey(handle, "log_posterior")
            sampler_log_posterior = vec(read(handle["log_posterior"]))
        end
        if haskey(handle, "proposal_parameters")
            sampler_proposal_parameters = read(handle["proposal_parameters"])
        end
        if haskey(handle, "previous_parameters")
            sampler_previous_parameters = read(handle["previous_parameters"])
        end
        if haskey(handle, "proposal_log_posterior")
            sampler_proposal_log_posterior = vec(read(handle["proposal_log_posterior"]))
        end
        if haskey(handle, "previous_log_posterior")
            sampler_previous_log_posterior = vec(read(handle["previous_log_posterior"]))
        end
        if haskey(handle, "proposal_log_likelihood")
            sampler_proposal_log_likelihood = vec(read(handle["proposal_log_likelihood"]))
        end
        if haskey(handle, "previous_log_likelihood")
            sampler_previous_log_likelihood = vec(read(handle["previous_log_likelihood"]))
        end
        if haskey(handle, "proposal_log_prior")
            sampler_proposal_log_prior = vec(read(handle["proposal_log_prior"]))
        end
        if haskey(handle, "previous_log_prior")
            sampler_previous_log_prior = vec(read(handle["previous_log_prior"]))
        end
        if haskey(handle, "uniform_draw")
            sampler_uniform_draw = vec(read(handle["uniform_draw"]))
        end
        if haskey(handle, "log_acceptance")
            sampler_log_acceptance = vec(read(handle["log_acceptance"]))
        end
        attrs = attributes(handle)
        if "acceptance_rate" in keys(attrs)
            sampler_acceptance_rate = read_attribute(handle, "acceptance_rate")
        end
        if "block_acceptance_rates" in keys(attrs)
            sampler_block_acceptance_rates = read_attribute(handle, "block_acceptance_rates")
        end
        read(handle["mhparams"])
    end
    draw_covariance = h5open(
        workpath(model, "estimate", "parameter_covariance.h5"),
        "r",
    ) do handle
        read(handle["mhcov"])
    end

    write_file_attribute(
        file,
        "sampler_parameter_names",
        join(ordered_mapping_names(model.parameters), ","),
    )
    write_file_attribute(file, "sampler_sampling_method", "MH")
    write_file_attribute(file, "sampler_draws", string(draws))
    write_file_attribute(file, "sampler_burnin", string(burnin))
    write_file_attribute(file, "sampler_blocks", string(blocks))
    write_file_attribute(file, "sampler_param_blocks", string(param_blocks))
    write_file_attribute(file, "sampler_thin", string(thin))
    write_file_attribute(file, "sampler_adaptive_accept", string(adaptive_accept))
    write_file_attribute(file, "sampler_target_accept", string(target_accept))
    write_file_attribute(file, "sampler_cc", string(cc))
    write_file_attribute(file, "sampler_alpha", string(alpha))
    write_file_attribute(file, "sampler_c", string(c))
    write_file_attribute(file, "sampler_cc0", string(cc0))
    write_file_attribute(file, "sampler_calculate_hessian", string(calculate_hessian))
    write_file_attribute(file, "sampler_reoptimize", string(reoptimize))
    write_file_attribute(file, "sampler_run_csminwel", string(run_csminwel))
    write_file_attribute(
        file,
        "sampler_proposal_scale",
        proposal_scale === nothing ? "" : string(proposal_scale),
    )
    write_file_attribute(file, "sampler_mode_in", mode_in)
    write_file_attribute(file, "sampler_hessian_in", hessian_in)
    write_file_attribute(file, "sampler_covariance_source", "saved_draw_covariance")
    write_file_attribute(
        file,
        "sampler_trace_available",
        string(!isempty(sampler_accepted) && !isempty(sampler_log_posterior)),
    )
    write_file_attribute(
        file,
        "sampler_proposal_trace_available",
        string(!isempty(sampler_proposal_parameters) && !isempty(sampler_proposal_log_posterior)),
    )
    write_file_attribute(file, "sampler_acceptance_rate", sampler_acceptance_rate)
    write_file_attribute(
        file,
        "sampler_block_acceptance_rates",
        sampler_block_acceptance_rates,
    )
    write_file_attribute(
        file,
        "sampler_input_proposal_covariance_available",
        string(!isempty(input_proposal_covariance)),
    )
    write_file_attribute(file, "sampler_seed", seed === nothing ? "" : string(seed))

    write_sampler_dataset(file, "sampler/mhparams", mhparams)
    write_sampler_dataset(file, "sampler/fixed", sampler_fixed)
    if !isempty(sampler_accepted)
        write_sampler_dataset(file, "sampler/accepted", sampler_accepted)
    end
    if !isempty(sampler_log_posterior)
        write_sampler_dataset(file, "sampler/log_posterior", sampler_log_posterior)
    end
    if !isempty(sampler_proposal_parameters)
        write_sampler_dataset(
            file,
            "sampler/proposal_parameters",
            sampler_proposal_parameters,
        )
    end
    if !isempty(sampler_previous_parameters)
        write_sampler_dataset(
            file,
            "sampler/previous_parameters",
            sampler_previous_parameters,
        )
    end
    if !isempty(sampler_proposal_log_posterior)
        write_sampler_dataset(
            file,
            "sampler/proposal_log_posterior",
            sampler_proposal_log_posterior,
        )
    end
    if !isempty(sampler_previous_log_posterior)
        write_sampler_dataset(
            file,
            "sampler/previous_log_posterior",
            sampler_previous_log_posterior,
        )
    end
    if !isempty(sampler_proposal_log_likelihood)
        write_sampler_dataset(
            file,
            "sampler/proposal_log_likelihood",
            sampler_proposal_log_likelihood,
        )
    end
    if !isempty(sampler_previous_log_likelihood)
        write_sampler_dataset(
            file,
            "sampler/previous_log_likelihood",
            sampler_previous_log_likelihood,
        )
    end
    if !isempty(sampler_proposal_log_prior)
        write_sampler_dataset(
            file,
            "sampler/proposal_log_prior",
            sampler_proposal_log_prior,
        )
    end
    if !isempty(sampler_previous_log_prior)
        write_sampler_dataset(
            file,
            "sampler/previous_log_prior",
            sampler_previous_log_prior,
        )
    end
    if !isempty(sampler_uniform_draw)
        write_sampler_dataset(file, "sampler/uniform_draw", sampler_uniform_draw)
    end
    if !isempty(sampler_log_acceptance)
        write_sampler_dataset(file, "sampler/log_acceptance", sampler_log_acceptance)
    end
    write_sampler_dataset(file, "sampler/proposal_covariance", draw_covariance)
    write_sampler_dataset(file, "sampler/draw_covariance", draw_covariance)
    if !isempty(input_proposal_covariance)
        write_sampler_dataset(
            file,
            "sampler/input_proposal_covariance",
            input_proposal_covariance,
        )
    end
end

function export_model1002(;
    out,
    subspec,
    data_vintage,
    forecast_start,
    horizon,
    include_forecast,
    include_full_forecast,
    full_draws,
    shock_samples_in,
    include_kalman,
    include_posterior,
    include_history,
    include_financial_frictions,
    include_sampler,
    sampler_seed,
    sampler_draws,
    sampler_burnin,
    sampler_blocks,
    sampler_param_blocks,
    sampler_thin,
    sampler_adaptive_accept,
    sampler_target_accept,
    sampler_cc,
    sampler_alpha,
    sampler_c,
    sampler_cc0,
    sampler_calculate_hessian,
    sampler_reoptimize,
    sampler_run_csminwel,
    sampler_proposal_scale,
    sampler_mode_in,
    sampler_hessian_in,
    data_in,
    data_out,
)
    custom_settings = Dict{Symbol,Setting}(
        :data_vintage => Setting(:data_vintage, data_vintage),
        :date_forecast_start => Setting(:date_forecast_start, quartertodate(forecast_start)),
    )
    model = Model1002(subspec; custom_settings = custom_settings)

    gamma0, gamma1, c, psi, pi = eqcond(model)
    transition_ttt, transition_ccc, transition_rrr, transition_eu = gensys(
        gamma0,
        gamma1,
        c,
        psi,
        pi,
        1.0 + 1.0e-6,
    )
    system = compute_system(model)

    mkpath(dirname(out))
    h5open(out, "w") do file
        write_file_attribute(file, "source", "FRBNY-DSGE/DSGE.jl")
        write_file_attribute(file, "model", "Model1002")
        write_file_attribute(file, "subspec", subspec)
        write_file_attribute(file, "data_vintage", data_vintage)
        write_file_attribute(file, "forecast_start", forecast_start)
        write_file_attribute(file, "created_utc", string(Dates.now(Dates.UTC)))
        export_label_attributes(file, model)

        export_parameters(file, model)
        export_steady_state(file, model)
        if include_financial_frictions
            export_financial_frictions(file)
        end
        if include_sampler
            export_sampler(
                file,
                model;
                data_in = data_in,
                data_out = data_out,
                draws = sampler_draws,
                burnin = sampler_burnin,
                blocks = sampler_blocks,
                param_blocks = sampler_param_blocks,
                thin = sampler_thin,
                adaptive_accept = sampler_adaptive_accept,
                target_accept = sampler_target_accept,
                c = sampler_c,
                cc = sampler_cc,
                cc0 = sampler_cc0,
                alpha = sampler_alpha,
                calculate_hessian = sampler_calculate_hessian,
                reoptimize = sampler_reoptimize,
                run_csminwel = sampler_run_csminwel,
                proposal_scale = sampler_proposal_scale,
                mode_in = sampler_mode_in,
                hessian_in = sampler_hessian_in,
                seed = sampler_seed,
            )
        end

        write_dataset(file, "canonical/Gamma0", gamma0)
        write_dataset(file, "canonical/Gamma1", gamma1)
        write_dataset(file, "canonical/C", c)
        write_dataset(file, "canonical/Psi", psi)
        write_dataset(file, "canonical/Pi", pi)

        write_dataset(file, "transition/TTT", transition_ttt)
        write_dataset(file, "transition/RRR", transition_rrr)
        write_dataset(file, "transition/CCC", transition_ccc)
        write_dataset(file, "transition/eu", Int64.(transition_eu))

        write_dataset(file, "system/TTT", system[:TTT])
        write_dataset(file, "system/RRR", system[:RRR])
        write_dataset(file, "system/CCC", system[:CCC])
        write_dataset(file, "system/ZZ", system[:ZZ])
        write_dataset(file, "system/DD", system[:DD])
        write_dataset(file, "system/QQ", system[:QQ])
        write_dataset(file, "system/EE", system[:EE])

        try
            write_dataset(file, "system/ZZ_pseudo", system[:ZZ_pseudo])
            write_dataset(file, "system/DD_pseudo", system[:DD_pseudo])
        catch err
            @warn "Pseudo-measurement matrices were not exported" exception = (err, catch_backtrace())
        end

        history_observables = nothing
        history_df = nothing
        if include_history
            history_observables, history_df = export_history_observables(
                file,
                model;
                data_in = data_in,
                data_out = data_out,
            )
        end
        if include_kalman
            export_kalman(file, model, system; data_in = data_in)
        end
        if include_posterior
            export_posterior(file, model, system; data_in = data_in)
        end
        forecast_initial_state = include_history ? filtered_forecast_start_state(
                model,
                system;
                data_in = data_in,
                history_df = history_df,
            ) : nothing
        if include_forecast
            export_mode_forecast(
                file,
                model;
                forecast_start = forecast_start,
                horizon = horizon,
                initial_state = forecast_initial_state,
            )
        end
        if include_full_forecast
            export_full_forecast(
                file,
                model;
                forecast_start = forecast_start,
                horizon = horizon,
                full_draws = full_draws,
                shock_samples_in = shock_samples_in,
                history_observables = history_observables,
                initial_state = forecast_initial_state,
            )
        end
    end
    println("Exported Model1002 oracle fixture to $(out)")
end

function export_model1002_main(args = ARGS)
    options = parse_args(args)
    export_model1002(
        out = options["out"],
        subspec = options["subspec"],
        data_vintage = options["data-vintage"],
        forecast_start = options["forecast-start"],
        horizon = parse(Int, options["horizon"]),
        include_forecast = parse_bool(options["include-forecast"]),
        include_full_forecast = parse_bool(options["include-full-forecast"]),
        full_draws = parse(Int, options["full-draws"]),
        shock_samples_in = options["shock-samples-in"],
        include_kalman = parse_bool(options["include-kalman"]),
        include_posterior = parse_bool(options["include-posterior"]),
        include_history = parse_bool(options["include-history"]),
        include_financial_frictions = parse_bool(options["include-financial-frictions"]),
        include_sampler = parse_bool(options["include-sampler"]),
        sampler_seed = parse_optional_int(options["sampler-seed"]),
        sampler_draws = parse_int(options["sampler-draws"]),
        sampler_burnin = parse_int(options["sampler-burnin"]),
        sampler_blocks = parse_int(options["sampler-blocks"]),
        sampler_param_blocks = parse_int(options["sampler-param-blocks"]),
        sampler_thin = parse_int(options["sampler-thin"]),
        sampler_adaptive_accept = parse_bool(options["sampler-adaptive-accept"]),
        sampler_target_accept = parse_float(options["sampler-target-accept"]),
        sampler_cc = parse_float(options["sampler-cc"]),
        sampler_alpha = parse_float(options["sampler-alpha"]),
        sampler_c = parse_float(options["sampler-c"]),
        sampler_cc0 = parse_float(options["sampler-cc0"]),
        sampler_calculate_hessian = parse_bool(options["sampler-calculate-hessian"]),
        sampler_reoptimize = parse_bool(options["sampler-reoptimize"]),
        sampler_run_csminwel = parse_bool(options["sampler-run-csminwel"]),
        sampler_proposal_scale = parse_optional_float(options["sampler-proposal-scale"]),
        sampler_mode_in = options["sampler-mode-in"],
        sampler_hessian_in = options["sampler-hessian-in"],
        data_in = options["data-in"],
        data_out = options["data-out"],
    )
end

function is_main_script()
    if isempty(PROGRAM_FILE)
        return false
    end
    return lowercase(normpath(abspath(PROGRAM_FILE))) == lowercase(normpath(abspath(@__FILE__)))
end

if is_main_script()
    export_model1002_main()
end
