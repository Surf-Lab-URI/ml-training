"""
Params — the single reader for `params.toml`.

Every tunable number in this pipeline comes from here, so that a user can change behaviour by
editing one commented file instead of hunting through five scripts.

Resolution order, highest priority first:

    1. an explicit command-line flag        (handled by each script's own arg parser)
    2. an environment variable              (`PIV_<SECTION>_<KEY>`, e.g. PIV_RUN_NT=60)
    3. params.toml
    4. the `default` passed to `Params.get` — a last-resort fallback so that a params.toml
       missing a key still runs rather than erroring

Environment variables exist mainly so a Slurm script can override one value for one array task
without editing a file that other tasks are reading concurrently. Reach for the file first.

Usage:

    include(joinpath(@__DIR__, "Params.jl"))
    nt = Params.get("run.nt", 40)                 # Int
    med = Params.get_vector("bins.v2.medians", [3, 6, 9])

The file is located via the PIV_PARAMS environment variable, or `params.toml` at the repository
root, and is read once and cached.
"""
module Params

using TOML

const _CACHE = Ref{Union{Nothing,Dict{String,Any}}}(nothing)
const _ROOT  = normpath(joinpath(@__DIR__, ".."))

"""Path to the parameter file actually in use. Override with PIV_PARAMS."""
# Base.get, not our own get — this module defines a `get` that shadows it inside the module body.
paramfile() = Base.get(ENV, "PIV_PARAMS", joinpath(_ROOT, "params.toml"))

"""Load and cache params.toml. A missing file is not fatal — every lookup then falls back to
its built-in default, which keeps old scripts working in a checkout without the file."""
function config()
    if _CACHE[] === nothing
        f = paramfile()
        _CACHE[] = if isfile(f)
            @info "Params: reading $f"
            TOML.parsefile(f)
        else
            @warn "Params: $f not found — falling back to built-in defaults for every value"
            Dict{String,Any}()
        end
    end
    return _CACHE[]::Dict{String,Any}
end

"""Force a re-read (used by the self-test; not needed in normal runs)."""
reload!() = (_CACHE[] = nothing; config())

"""Environment-variable name for a dotted key: "run.nt" -> "PIV_RUN_NT"."""
envname(key::AbstractString) = "PIV_" * uppercase(replace(key, "." => "_"))

# Legacy environment variables that predate this file. Kept working so existing Slurm scripts and
# anyone's muscle memory keep behaving as before; each maps onto its new dotted key.
const _LEGACY_ENV = Dict(
    "imaging.appearance.mode"      => "PIV_LAB_APPEARANCE",  # "1"/"0" -> "lab"/"clean"
    "bins.v2.medians"              => "PIV_V2_MEDIANS",
    "bins.v2.tolerance"            => "PIV_V2_TOL",
    "bins.v2.write_out_of_tolerance" => "PIV_V2_LOOSE",
)

"""Walk a dotted key through the nested TOML dictionaries. Returns `nothing` if absent."""
function _dig(key::AbstractString)
    node::Any = config()
    for part in split(key, ".")
        node isa AbstractDict || return nothing
        haskey(node, part) || return nothing
        node = node[part]
    end
    return node
end

"""Coerce a string (from the environment) to the type of `default`."""
function _coerce(s::AbstractString, default::T) where {T}
    T <: Bool          && return lowercase(strip(s)) in ("1", "true", "yes", "on")
    T <: Integer       && return parse(T, strip(s))
    T <: AbstractFloat && return parse(T, strip(s))
    return s
end

"""
    get(key, default)

Resolve one dotted key. Environment variable wins over the file; the file wins over `default`.
The returned value is coerced to the type of `default`, so `get("run.nt", 40)` always gives an Int
even when the environment supplies the string "60".
"""
function get(key::AbstractString, default::T) where {T}
    for var in (envname(key), Base.get(_LEGACY_ENV, key, nothing))
        var === nothing && continue
        raw = Base.get(ENV, var, nothing)
        raw === nothing && continue
        # PIV_LAB_APPEARANCE is a 0/1 flag standing in for a mode name.
        if var == "PIV_LAB_APPEARANCE"
            return (strip(raw) == "1" ? "lab" : "clean")::T
        end
        return _coerce(raw, default)
    end
    v = _dig(key)
    v === nothing && return default
    v isa AbstractString && !(T <: AbstractString) && return _coerce(v, default)
    T <: AbstractFloat && v isa Integer && return T(v)   # TOML writes 1 where we want 1.0
    T <: Integer && v isa AbstractFloat && isinteger(v) && return T(v)
    return v
end

"""
    get_vector(key, default)

Vector-valued lookup. An environment override is comma-separated, e.g. PIV_BINS_V2_MEDIANS="3,9,20".
Element type follows `default`, so an Int vector stays an Int vector.
"""
function get_vector(key::AbstractString, default::Vector{T}) where {T}
    for var in (envname(key), Base.get(_LEGACY_ENV, key, nothing))
        var === nothing && continue
        raw = Base.get(ENV, var, nothing)
        (raw === nothing || isempty(strip(raw))) && continue
        return T[_coerce(p, zero(T)) for p in split(raw, ",") if !isempty(strip(p))]
    end
    v = _dig(key)
    v === nothing && return default
    v isa AbstractVector || return default
    return T[x isa AbstractString ? _coerce(x, zero(T)) : T(x) for x in v]
end

"""
    describe(io = stdout)

Print every resolved value that a run will actually use. Called by the generators at startup so
that a Slurm log records the configuration alongside the results — the log, not this file, is the
record of what a given dataset was built with.
"""
function describe(io::IO = stdout; keys_shown = String[])
    println(io, "── resolved parameters (", paramfile(), ") ─────────────────────────────")
    for k in keys_shown
        println(io, rpad("  " * k, 42), _dig(k) === nothing ? "(default)" : repr(_dig(k)))
    end
    println(io, "─"^78)
end

end # module
