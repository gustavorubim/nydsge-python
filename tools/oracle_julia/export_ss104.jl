# Fixed-default ss104 oracle. Never estimates or samples parameters.
# Usage: julia --project=tools/oracle_julia export_ss104.jl DSGE_ROOT SSR_ROOT OUT_H5
# Bootstrap package types/dependencies from the pinned oracle environment, then
# replace economic and solve/state-space methods with the supplied reference source.
using DSGE, HDF5, JSON, Dates, LinearAlgebra
length(ARGS) == 3 || error("Expected DSGE_ROOT SSR_ROOT OUT_H5")
dsge_root, ssr_root, output = abspath.(ARGS)
include(joinpath(@__DIR__, "ss104_reference.jl"))
m = DSGE.Model1002("ss104"; custom_settings=[
    DSGE.Setting(:date_forecast_start, Date(2026,9,30)),
    DSGE.Setting(:date_conditional_end, Date(2026,9,30)),
    DSGE.Setting(:saveroot, dirname(output)), DSGE.Setting(:dataroot, dirname(output))])
println("Model dimensions: ", DSGE.n_states(m), " / ", DSGE.n_states_augmented(m),
        " / ", DSGE.n_shocks_exogenous(m), " / ", DSGE.n_observables(m))
s = DSGE.compute_system(m; tvis=true, verbose=:none)
println("Computed ", length(s.transitions), " regimes")

# Controlled finite observations, not a production vintage. All regimes are
# independently matrix-checked; history covers 2019Q4 through 2026Q2.
y = fill(NaN, 28, 28)
for t in 1:28, name in (:obs_gdp,:obs_hours,:obs_wages,:obs_gdpdeflator,:obs_corepce,
                        :obs_nominalrate,:obs_consumption,:obs_investment,:obs_spread,
                        :obs_longinflation,:obs_longrate,:obs_tfp,:obs_gdi)
    i = m.observables[name]
    y[i,t] = s[1,:DD][i] + .01*sin((t-1)+(i-1))
end
y[m.observables[:obs_pgap],3] = -.125
y[m.observables[:obs_ygap],3] = -12.
y[m.observables[:obs_nominalrate],5:9] .= NaN
y[:,28] .= NaN
for name in (:obs_gdp,:obs_corepce,:obs_spread,:obs_nominalrate,:obs_longrate)
    i = m.observables[name]
    y[i,28] = s[1,:DD][i] + .01*sin(27+i-1)
end
y[m.observables[:obs_gdp],28] = .65
y[m.observables[:obs_corepce],28] = .70
for h in 1:6
    y[m.observables[Symbol("obs_exp_nominalrate$h")],28] = h == 6 ? .845 : .9075
end
Ts,Rs,Cs,Qs,Zs,Ds,Es = ([s[r,k] for r in 1:28] for k in (:TTT,:RRR,:CCC,:QQ,:ZZ,:DD,:EE))
function replay(n)
    regs = [i:i for i in 1:n]
    loglh, spred, Ppred, sfilt, Pfilt = SS104Filter.kalman_filter(regs, y[:,1:n], Ts[1:n],Rs[1:n],Cs[1:n],Qs[1:n],Zs[1:n],Ds[1:n],Es[1:n])
    s0,P0 = SS104Filter.init_stationary_states(Ts[1],Rs[1],Cs[1],Qs[1])
    smth,_ = SS104Filter.durbin_koopman_smoother(regs,y[:,1:n],Ts[1:n],Rs[1:n],Cs[1:n],Qs[1:n],Zs[1:n],Ds[1:n],Es[1:n],s0,P0; draw_states=false)
    return loglh,spred,Ppred,sfilt,Pfilt,smth
end
history = replay(27)
conditional = replay(28)
forecast = DSGE.forecast(m,s,history[4][:,end],zeros(48,40); cond_type=:none)
conditional_forecast = DSGE.forecast(m,s,conditional[6][:,end],zeros(48,39); cond_type=:full)
println("History loglik=", sum(history[1]), "; conditional loglik=", sum(conditional[1]))
h5open(output, "w") do f
    write(f, "metadata/json", JSON.json(Dict(
        "dsge_commit"=>strip(read(`git -c safe.directory=$dsge_root -C $dsge_root rev-parse HEAD`, String)),
        "statespace_commit"=>strip(read(`git -c safe.directory=$ssr_root -C $ssr_root rev-parse HEAD`, String)),
        "julia_version"=>string(VERSION), "forecast_start"=>"2026Q3",
        "scope"=>"ss104 default params; reference methods over pinned package types; no estimation")))
    for key in (:endogenous_states, :endogenous_states_augmented, :exogenous_shocks,
                :equilibrium_conditions, :expected_shocks, :observables, :pseudo_observables)
        names = sort(collect(getfield(m,key)); by=last)
        write(f, "names/" * string(key), string.(first.(names)))
    end
    for r in 1:29
        canonical = DSGE.eqcond(m, r)
        for (name, array) in zip(("Gamma0", "Gamma1", "C", "Psi", "Pi"), canonical)
            write(f, "canonical/$r/$name", array)
        end
        for name in (:TTT, :RRR, :CCC, :ZZ, :DD, :QQ, :EE, :ZZ_pseudo, :DD_pseudo)
            write(f, "system/$r/" * string(name), s[r,name])
        end
        ModelConstructors = DSGE.ModelConstructors
        for p in m.parameters
            if !isempty(p.regimes)
                map = DSGE.get_setting(m,:model2para_regime)
                ModelConstructors.toggle_regime!(p, r, haskey(map,p.key) ? map[p.key] : Dict(i=>i for i in 1:29))
            end
        end
        write(f, "parameters/$r/names", string.([p.key for p in m.parameters]))
        write(f, "parameters/$r/values", [p.value for p in m.parameters])
    end
    write(f,"workflow/data",y)
    for (prefix,replay_result) in (("history",history),("conditional",conditional))
        for (name,array) in zip(("loglik", "predicted_states", "predicted_covariances", "filtered_states", "filtered_covariances", "smoothed_states"),replay_result)
            write(f,"workflow/$prefix/$name",array)
        end
    end
    for (prefix,result) in (("forecast",forecast),("conditional_forecast",conditional_forecast))
        for (name,array) in zip(("states","observables","pseudo","shocks"),result)
            write(f,"workflow/$prefix/$name",array)
        end
    end
end
println("Wrote ss104 oracle: ", output)
