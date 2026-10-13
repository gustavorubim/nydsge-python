# Actual local-data ss104 forecast at source defaults. No estimation or sampling.
# Usage: julia --project=tools/oracle_julia forecast_ss104_nyfed_params.jl DSGE_ROOT SSR_ROOT INPUT_DIR OUT_H5
using DSGE, HDF5, JSON, Dates, LinearAlgebra, CSV, DataFrames, SHA
length(ARGS) == 4 || error("Expected DSGE_ROOT SSR_ROOT INPUT_DIR OUT_H5")
dsge_root, ssr_root, input_dir, output = abspath.(ARGS)
include(joinpath(@__DIR__, "ss104_reference.jl"))
meta = JSON.parsefile(joinpath(input_dir, "forecast_metadata.json"))
function source_commit(root)
    strip(read(`git -c safe.directory=$root -C $root rev-parse HEAD`, String))
end
source_commit(dsge_root) == meta["dsge_commit"] || error("Wrong DSGE source commit")
source_commit(ssr_root) == meta["statespace_commit"] || error("Wrong filter source commit")
data_path = joinpath(input_dir, "conditioned_observations.csv")
bytes2hex(sha256(read(data_path))) == meta["conditioned_observations_sha256"] || error("Data hash mismatch")
meta["cond_type"] == "semi" || error("Require semi conditioning")
meta["horizon_including_conditional"] == 40 || error("Require 40 quarters")

m = DSGE.Model1002("ss104"; custom_settings=[
    DSGE.Setting(:date_presample_start, Date(1959,9,30)),
    DSGE.Setting(:date_mainsample_start, Date(1960,3,31)),
    DSGE.Setting(:date_forecast_start, Date(2026,9,30)),
    DSGE.Setting(:date_conditional_end, Date(2026,9,30)),
    DSGE.Setting(:saveroot, dirname(output)), DSGE.Setting(:dataroot, dirname(output))])
function ascii_name(key)
    name = string(key)
    for (a,b) in ("α"=>"alpha", "β"=>"beta", "ρ"=>"rho", "σ"=>"sigma", "π"=>"pi",
                  "ζ"=>"zeta", "ι"=>"iota", "μ"=>"mu", "ω"=>"omega", "γ"=>"gamma",
                  "λ"=>"lambda", "δ"=>"delta", "Φ"=>"Phi", "Γ"=>"Gamma", "ν"=>"nu",
                  "ϵ"=>"epsilon", "φ"=>"phi", "S′′"=>"Spp", "ψ"=>"psi", "η"=>"eta")
        name = replace(name, a=>b)
    end
    name
end
parameter_error = 0.0
for r in 1:29
    map = DSGE.get_setting(m,:model2para_regime)
    actual = Dict{String,Float64}()
    for p in m.parameters
        if !isempty(p.regimes)
            DSGE.ModelConstructors.toggle_regime!(p, r,
                haskey(map,p.key) ? map[p.key] : Dict(i=>i for i in 1:29))
        end
        actual[ascii_name(p.key)] = p.value
    end
    requested = meta["parameters_by_regime"][string(r)]
    Set(keys(actual)) == Set(keys(requested)) || error("Parameter names differ in regime $r")
    global parameter_error = max(parameter_error, maximum(abs(actual[k]-requested[k]) for k in keys(actual)))
end
parameter_error <= 1e-12 || error("Parameter mismatch: $parameter_error")
println("Verified all 130 parameters in all 29 regimes; max error = ", parameter_error)
s = DSGE.compute_system(m; tvis=true, verbose=:none)
frame = CSV.read(data_path, DataFrame)
ordered_names(field) = string.(first.(sort(collect(getfield(m,field)); by=last)))
obs_names = ordered_names(:observables)
Set(names(frame)) == Set(vcat(["date"], obs_names)) || error("Observable columns differ")
quarter_date(q) = lastdayofmonth(Date(parse(Int,q[1:4]), 3*parse(Int,q[end:end]), 1))
dates = quarter_date.(string.(frame.date))
dates[1] == Date(1959,9,30) && dates[end] == Date(2026,9,30) || error("History dates differ")
all(dates[t] == lastdayofmonth(dates[t-1] + Month(3)) for t in 2:length(dates)) || error("Noncontiguous dates")
y = permutedims(Float64.(coalesce.(Matrix(frame[:,obs_names]), NaN)))
regime_dates = DSGE.get_setting(m,:regime_dates)
regimes = [maximum(r for (r,d) in regime_dates if d <= date) for date in dates]
regimes == meta["history_regimes"] || error("History regimes differ")
keep = union(Set(string.(DSGE.get_setting(m,:cond_semi_names))),
             Set("obs_exp_nominalrate$i" for i in 1:6))
all(isnan(y[i,end]) for (i,n) in enumerate(obs_names) if !(n in keep)) || error("Semi mask differs")

# Split regime 1 at the historical 2008Q4 ZLB: anticipated Taylor innovations
# have zero variance before that date, exactly as in the Python modal recipe.
tags = [(regimes[t], dates[t] < DSGE.get_setting(m,:date_zlb_start)) for t in eachindex(dates)]
starts = vcat([1], [t for t in 2:length(tags) if tags[t] != tags[t-1]])
stops = vcat(starts[2:end] .- 1, [length(tags)])
segments = [a:b for (a,b) in zip(starts,stops)]
Ts,Rs,Cs,Qs,Zs,Ds,Es = (Any[] for _ in 1:7)
for a in starts
    r, pre_zlb = tags[a]
    push!(Ts,s[r,:TTT]); push!(Rs,s[r,:RRR]); push!(Cs,s[r,:CCC])
    q = copy(s[r,:QQ])
    if pre_zlb
        ant = [m.exogenous_shocks[Symbol("rm_shl$i")] for i in 1:6]
        q[ant,:] .= 0.; q[:,ant] .= 0.
    end
    push!(Qs,q); push!(Zs,s[r,:ZZ]); push!(Ds,s[r,:DD]); push!(Es,s[r,:EE])
end
# Concrete arrays match the upstream dispatch signatures.
Ts,Rs,Qs,Zs,Es = (Matrix{Float64}[v...] for v in (Ts,Rs,Qs,Zs,Es))
Cs,Ds = (Vector{Float64}[v...] for v in (Cs,Ds))
s0,P0 = SS104Filter.init_stationary_states(Ts[1],Rs[1],Cs[1],Qs[1])
presample = count(d -> d < Date(1960,3,31), dates)
filtered = SS104Filter.kalman_filter(segments,y,Ts,Rs,Cs,Qs,Zs,Ds,Es,s0,P0; Nt0=presample)
smoothed,_ = SS104Filter.durbin_koopman_smoother(segments,y,Ts,Rs,Cs,Qs,Zs,Ds,Es,s0,P0; draw_states=false)
forecast = DSGE.forecast(m,s,smoothed[:,end],zeros(48,39); cond_type=:semi)
cond_state = smoothed[:,end:end]
cond_obs = s[28,:ZZ]*cond_state .+ s[28,:DD]
cond_pseudo = s[28,:ZZ_pseudo]*cond_state .+ s[28,:DD_pseudo]
mkpath(dirname(output))
h5open(output,"w") do f
    write(f,"metadata/json",JSON.json(Dict(
        "dsge_commit"=>source_commit(dsge_root), "statespace_commit"=>source_commit(ssr_root),
        "julia_version"=>string(VERSION), "parameter_source"=>meta["parameter_source"],
        "parameter_max_abs_error"=>parameter_error,
        "conditioned_observations_sha256"=>bytes2hex(sha256(read(data_path))),
        "cond_type"=>"semi", "presample_periods"=>presample,
        "forecast_start"=>"2026Q3", "forecast_end"=>"2036Q2", "estimation_runs"=>0)))
    for field in (:observables,:pseudo_observables)
        write(f,"names/"*string(field),ordered_names(field))
    end
    write(f,"workflow/data",y)
    write(f,"workflow/dates",string.(frame.date))
    write(f,"workflow/regimes",regimes)
    write(f,"workflow/loglik",filtered[1])
    write(f,"forecast/states",hcat(cond_state,forecast[1]))
    write(f,"forecast/observables",hcat(cond_obs,forecast[2]))
    write(f,"forecast/pseudo",hcat(cond_pseudo,forecast[3]))
end
println("Wrote same-input Julia forecast: ",output)
