# Offline kernel oracle. No Julia or fixture access is used by Python runtime.
# Usage: julia --startup-file=no export_ss104_m1.jl DSGE_ROOT SSR_ROOT OUTPUT_JSON
# Sources: DSGE.jl c09dfe8 and StateSpaceRoutines.jl 794fb02 (see fixture metadata).
# Only upstream linear kernels are included. Unused Tracker dispatches are given
# abstract placeholder types to avoid installing particle-filter/AD dependencies.
using LinearAlgebra, SparseArrays

abstract type AbstractDSGEModel{T} end
struct OracleModel <: AbstractDSGEModel{Float64} end
get_settings(::OracleModel) = Dict{Symbol, Any}()
get_setting(::OracleModel, key) = get_settings(OracleModel())[key]
struct System{T} end
struct RegimeSwitchingSystem{T} end
const VERBOSITY = Dict(:none => 0, :low => 1, :high => 2)

module Tracker
    abstract type TrackedReal{T} <: Real end
end
abstract type TrackedArray{T} <: AbstractMatrix{T} end
const TrackedMatrix = TrackedArray
abstract type TrackedVector{T} <: AbstractVector{T} end

length(ARGS) == 3 || error("Expected DSGE_ROOT SSR_ROOT OUTPUT_JSON")
dsge_root, ssr_root, output = abspath.(ARGS)
include(joinpath(dsge_root, "src", "solve", "gensys.jl"))
include(joinpath(dsge_root, "src", "solve", "gensys2.jl"))
include(joinpath(dsge_root, "src", "statespace", "util.jl"))
include(joinpath(ssr_root, "src", "filters", "util.jl"))
include(joinpath(ssr_root, "src", "filters", "kalman_filter.jl"))
include(joinpath(ssr_root, "src", "smoothers", "util.jl"))
include(joinpath(ssr_root, "src", "smoothers", "hamilton_smoother.jl"))

function canonical(phi = 1.5, rate = 0.)
    g0 = [1. 0. 1. -1.; -.1 1. 0. -.99; 0. -phi 1. 0.; 0. 1. 0. 0.]
    g1 = zeros(4, 4)
    g1[1, 1] = .5
    g1[4, 4] = 1.
    return g0, g1, [0., 0., rate, 0.], reshape([1., 0., 0., 0.], 4, 1), reshape([0., 0., 0., 1.], 4, 1)
end
T, C, R, eu = gensys(canonical()...; verbose = :none)
eu == [1, 1] || error("Permanent policy failed gensys: $eu")
Tcal, Rcal, Ccal = gensys2(OracleModel(), canonical(0., -.25)..., real(T), real(R), real(C), 3)
zlb_path = [Dict("TTT" => Tcal[i], "RRR" => Rcal[i], "CCC" => Ccal[i]) for i in 1:3]

# All seven matrices change, with partial and entirely missing observations.
Ts = [[.6 .1; 0. .3], [.2 -.2; .1 .7]]
Rs = [Matrix{Float64}(I, 2, 2), [.5 .1; 0. 1.2]]
Cs = [[.2, -.1], [-.1, .4]]
Qs = [Matrix(Diagonal([.4, .2])), Matrix(Diagonal([.8, .1]))]
Zs = [Matrix{Float64}(I, 2, 2), [1. .5; -.2 1.]]
Ds = [[.3, .1], [-.4, .2]]
Es = [Matrix(Diagonal([.1, .2])), Matrix(Diagonal([.3, .05]))]
y = [.5 NaN .2 NaN -.4; -.2 .3 NaN NaN .6]
regime_indices = [1:2, 3:5]
loglh, s_pred, P_pred, s_filt, P_filt = kalman_filter(regime_indices, y, Ts, Rs, Cs, Qs, Zs, Ds, Es)
s0, P0 = init_stationary_states(Ts[1], Rs[1], Cs[1], Qs[1])
s_smth, _ = hamilton_smoother(regime_indices, y, Ts, Rs, Cs, Qs, Zs, Ds, Es, s0, P0)
kalman = Dict("log_likelihood_by_period" => loglh,
    "predicted_states" => permutedims(s_pred),
    "predicted_covariances" => permutedims(P_pred, (3, 1, 2)),
    "filtered_states" => permutedims(s_filt),
    "filtered_covariances" => permutedims(P_filt, (3, 1, 2)),
    "smoothed_states" => permutedims(s_smth))
power, constant = k_periods_ahead_expectations(Ts[1], Cs[1], Ts, Cs, 1, 3, 2)
sum_power, sum_constant = k_periods_ahead_expected_sums(Ts[1], Cs[1], Ts, Cs, 1, 3, 2)

# Small dependency-free JSON writer; arrays preserve their written dimensions.
function json(x)
    if x isa AbstractDict
        return "{" * join([json(string(k)) * ":" * json(x[k]) for k in sort(collect(keys(x)))], ",") * "}"
    elseif x isa AbstractString
        return "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
    elseif x isa AbstractArray
        values = ndims(x) == 1 ? x : [selectdim(x, 1, i) for i in axes(x, 1)]
        return "[" * join(json.(values), ",") * "]"
    else
        return string(x)
    end
end
fixture = Dict("metadata" => Dict(
    "dsge_commit" => strip(read(`git -C $dsge_root rev-parse HEAD`, String)),
    "statespace_commit" => strip(read(`git -C $ssr_root rev-parse HEAD`, String)),
    "julia_version" => string(VERSION), "orientation" => "periods x states",
    "scope" => "small kernel oracle; not Model1002 ss104 parity"),
    "gensys2" => zlb_path, "kalman" => kalman,
    "expectations" => Dict("TTT" => power, "CCC" => constant),
    "expected_sums" => Dict("TTT" => sum_power, "CCC" => sum_constant))
write(output, json(fixture) * "\n")
println("Wrote M1 kernel oracle to ", output)
