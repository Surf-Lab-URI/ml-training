#!/usr/bin/env julia
# Emit params.toml as shell variable assignments, so the Slurm submitters read the same file the
# Julia code does instead of keeping a second copy of every default.
#
#     eval "$(julia --project=. scripts/params_export.jl)"
#     echo "$PIV_N_SIMS"
#
# Only the values the shell scripts actually need are exported — the physics stays in Julia's
# hands. Anything already set in the environment is left alone, so a one-off
# `PIV_N_SIMS=50 ./unity/submit_v2.sh ...` still wins over the file.
#
# Printing assignments rather than having each script parse TOML keeps the shell side
# dependency-free and means one `eval` covers every variable.
include(joinpath(@__DIR__, "..", "src", "Params.jl"))

using Printf

# (shell variable, dotted key in params.toml, fallback)
const EXPORTS = [
    ("PIV_N_SIMS",        "run.n_sims",         "10000"),
    ("PIV_BASE_SEED",     "run.base_seed",      "0"),
    ("PIV_NT",            "run.nt",             "40"),
    ("PIV_KEEP_COMBINED", "run.keep_combined",  "1"),
    ("PIV_OUTPUT_ROOT",   "run.output_root",    ""),
    ("PIV_PROJECT_DIR",   "unity.project_dir",  ""),
    ("PIV_JULIA_DEPOT",   "unity.julia_depot",  ""),
    ("PIV_PARTITION",     "unity.partition",    "uri-cpu"),
    ("PIV_CHUNK",         "unity.chunk",        "50"),
    ("PIV_TIME_SIMULATE", "unity.time_limit_simulate", "00:40:00"),
    ("PIV_TIME_RENDER",   "unity.time_limit_render",   "04:00:00"),
    ("PIV_MAX_CONCURRENT","unity.max_concurrent","100"),
    ("PIV_MEM",           "unity.mem",          "8G"),
    ("PIV_CPUS_PER_TASK", "unity.cpus_per_task","4"),
    ("PIV_PARTICLES_PER_IMAGE", "imaging.particles_per_image", "12000"),
]

"""Render a TOML value the way a shell wants it: booleans as 1/0, everything else as-is."""
shellval(v) = v isa Bool ? (v ? "1" : "0") : string(v)

# The v2 bin directory names, derived from the medians so this list can never drift away from what
# the generator actually writes. Exported as one space-separated string.
if isempty(get(ENV, "PIV_BINS", ""))
    meds = Params.get_vector("bins.v2.medians", [3.0, 6.0, 9.0, 12.0, 16.0, 20.0, 26.0, 30.0])
    println("export PIV_BINS='", join([@sprintf("med%02d", round(Int, m)) for m in meds], " "), "'")
end

for (var, key, fallback) in EXPORTS
    # Respect anything the caller already set — the environment outranks the file.
    if haskey(ENV, var) && !isempty(ENV[var])
        continue
    end
    raw = Params.get(key, "")           # "" means the key is absent from params.toml
    val = isempty(string(raw)) ? fallback : shellval(raw)
    isempty(val) && continue
    println("export $var='", replace(val, "'" => "'\\''"), "'")
end
