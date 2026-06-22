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
using ModelConstructors
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

function parameter_fixed(parameter)
    value = maybe_property(parameter, (:fixed, :is_fixed); default = false)
    return Bool(value) ? 1.0 : 0.0
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
    sorted_items = sort(items; by = item -> item.second)
    return [string(item.first) for item in sorted_items]
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

options = parse_args(ARGS)
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
    data_in = options["data-in"],
    data_out = options["data-out"],
)
