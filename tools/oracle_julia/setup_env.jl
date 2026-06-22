# Bootstrap the migration-only Julia oracle environment.
#
# This intentionally lives outside the Python package runtime. It creates a
# local Julia project under tools/oracle_julia with the upstream packages needed
# to export parity fixtures.

using Pkg

Pkg.activate(@__DIR__)
Pkg.add(["CSV", "DataFrames", "DSGE", "HDF5", "ModelConstructors"])
Pkg.instantiate()
Pkg.precompile()
Pkg.status()
