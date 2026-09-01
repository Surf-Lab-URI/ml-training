using DrWatson
@quickactivate "ml-training"

using ArgParse

include(joinpath(@__DIR__, "Params.jl"))

# Command-line flags all default to `nothing` on purpose. A flag left off is then filled in from
# params.toml below, which is what makes that file authoritative: if the flags carried their own
# defaults, params.toml could never take effect for anyone who did not pass the flag.
# The resolved values are written back into `parsed_args`, so every downstream use of
# `parsed_args["..."]` keeps working exactly as it did before this file grew a parameter reader.
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--jet_amp", "-a"
            help = "amplitude of jet mode in streamfunction [params: physics.jet_amplitude]"
            arg_type = Float64
            default = nothing
        "--n_max", "-n"
            help = "number of modes in streamfunction [params: physics.n_modes]"
            arg_type = Int
            default = nothing
        "--m_jet", "-j"
            help = "wavenumber of horizontal jet [params: physics.m_jet]"
            arg_type = Int
            default = nothing
        "--t_end", "-t"
            help = "end time of simulation; omitted -> nt*dt_save [params: physics.t_end]"
            arg_type = Float64
            default = nothing
        "--nt"
            help = "number of timesteps recorded [params: run.nt]"
            arg_type = Int
            default = nothing
        "--out_dir", "-o"
            help = "output directory"
            arg_type = String
            default = "out/"
        "--no_image_gen"
            help = "skip image pair generation after combining (run sim + combine only)"
            action = :store_true
        "--seed"
            help = "master random seed for reproducibility (drives all simulation rand() calls)"
            arg_type = Int
            default = nothing
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

# --- Fill every unset flag from params.toml ----------------------------------------------------
# `something(a, b)` returns a unless it is nothing. t_end stays nothing when neither the flag nor
# the file supplies it, which is the signal to derive it as nt*dt_save (see 2DTurbulence.jl).
parsed_args["seed"]    = something(parsed_args["seed"],    Params.get("run.seed", 1234))
parsed_args["nt"]      = something(parsed_args["nt"],      Params.get("run.nt", 40))
parsed_args["jet_amp"] = something(parsed_args["jet_amp"], Params.get("physics.jet_amplitude", 300.0))
parsed_args["n_max"]   = something(parsed_args["n_max"],   Params.get("physics.n_modes", 21))
parsed_args["m_jet"]   = something(parsed_args["m_jet"],   Params.get("physics.m_jet", 2))
if parsed_args["t_end"] === nothing
    te = Params.get("physics.t_end", -1.0)
    parsed_args["t_end"] = te > 0 ? te : nothing
end

tag!(parsed_args)

# Arguments used for defining random initial conditons

# A (jet amplitude) is computed in 2DTurbulence.jl AFTER Random.seed! so the seed applies to it (BUG-8)
nmax = parsed_args["n_max"]
mmax = parsed_args["n_max"]
mjet = parsed_args["m_jet"]
automate = !parsed_args["no_image_gen"]   # BUG-3: store-true flag replaces unreliable Bool parsing

# Arguments used for defining simulation time and output
# (st is computed in 2DTurbulence.jl after `dt` exists — see BUG-1)
t_end = parsed_args["t_end"]
nt    = parsed_args["nt"]

# Grid, fluid and time-stepping constants, all from params.toml. These used to be literals inside
# 2DTurbulence.jl, which meant the only way to change the resolution or the viscosity was to edit
# the solver script.
grid_n              = Params.get("physics.grid_n", 512)
grid_m              = Params.get("physics.grid_m", 512)
viscosity_nu        = Params.get("physics.viscosity_nu", 1e-5)
advection_order     = Params.get("physics.advection_order", 5)
n_particles         = Params.get("physics.n_particles", grid_n * grid_m ÷ 16)
cfl_safety          = Params.get("physics.cfl_safety", 0.5)
save_interval_factor = Params.get("physics.save_interval_factor", 10)
wizard_cfl          = Params.get("physics.wizard_cfl", 0.7)
wizard_max_change   = Params.get("physics.wizard_max_change", 1.1)
wizard_max_dt_factor = Params.get("physics.wizard_max_dt_factor", 2)
