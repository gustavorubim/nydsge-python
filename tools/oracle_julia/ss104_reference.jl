# Shared c09dfe8 economic overlay and 794fb02 filter/smoother.
LinearAlgebra.BLAS.set_num_threads(1)
function overlay(relative; skip_types = false)
    path = joinpath(dsge_root, "src", relative)
    code = read(path, String)
    if skip_types
        # Filter parsed definitions, including @doc wrappers, without matching
        # type examples inside docstrings.
        function filter_types(expr)
            expr isa Expr || return expr
            if expr.head in (:struct, :abstract)
                return nothing
            elseif expr.head == :macrocall && expr.args[end] isa Expr &&
                   expr.args[end].head in (:struct, :abstract)
                return nothing
            end
            return expr
        end
        Base.include_string(filter_types, DSGE, code, path)
    else
        Base.include_string(DSGE, code, path)
    end
end
overlay("abstractdsgemodel.jl"; skip_types=true)
overlay("defaults.jl")
overlay("util.jl"; skip_types=true)
overlay("statespace/util.jl"; skip_types=true)
for file in ("gensys.jl", "gensys2.jl", "gensys_uncertain_altpol.jl", "gensys2_uncertain_altpol.jl", "solve.jl")
    overlay("solve/" * file)
end
for file in ("default_policy.jl", "flexible_ait.jl", "zlb_rule.jl")
    overlay("altpolicy/" * file)
end
for file in ("m1002.jl", "observables.jl", "pseudo_observables.jl", "subspecs_builder.jl", "subspecs.jl", "eqcond.jl", "augment_states.jl", "measurement.jl", "pseudo_measurement.jl")
    overlay("models/representative/m1002/" * file; skip_types=file == "m1002.jl")
end
overlay("statespace/statespace_functions.jl")
overlay("forecast/forecast.jl")
println("Loaded reference economic and policy methods")

module SS104Filter
    using LinearAlgebra, SparseArrays, Random
    module Tracker
        abstract type TrackedReal{T} <: Real end
    end
    abstract type TrackedArray{T} <: AbstractMatrix{T} end
    const TrackedMatrix = TrackedArray
    abstract type TrackedVector{T} <: AbstractVector{T} end
    include(joinpath(Main.ssr_root,"src","filters","util.jl"))
    include(joinpath(Main.ssr_root,"src","filters","kalman_filter.jl"))
    include(joinpath(Main.ssr_root,"src","smoothers","util.jl"))
    include(joinpath(Main.ssr_root,"src","smoothers","hamilton_smoother.jl"))
    include(joinpath(Main.ssr_root,"src","smoothers","koopman_smoother.jl"))
    include(joinpath(Main.ssr_root,"src","smoothers","durbin_koopman_smoother.jl"))
end
