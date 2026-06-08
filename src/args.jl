using DrWatson
@quickactivate "ml-training"

using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--jet_amp", "-a"
            help = "amplitude of jet mode in streamfunction"
            arg_type = Float64
            default = 300.0
        "--n_max", "-n"
            help = "number of modes in streamfunction"
            arg_type = Int
            default = 21
        "--m_jet", "-j"
            help = "wavenumber of horizontal jet"
            arg_type = Int
            default = 2
        "--t_end", "-t"
            help = "end time of simulation"
            arg_type = Float64
            default = nothing
        "--nt"
            help = "number of timesteps recorded"
            arg_type = Int
            default = 20
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
            default = 1234
    end

    return parse_args(s)
end

parsed_args = parse_commandline()
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