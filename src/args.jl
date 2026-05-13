using DrWatson
@quickactivate "ml-training"

using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--jet_amp", "-a"
            help = "amplitude of jet mode in streamfunction"
            arg_type = Float64
            default = 300
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
        "--automate", "-m"
            help = "automate the entire process, from running the simulation to generating image pairs for training (overrides other arguments except for out_dir)"
            arg_type = Bool
            default = true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()
tag!(parsed_args)

# Arguments used for defining random initial conditons 

A = parsed_args["jet_amp"]*(1.5-rand()) #Amplitude of a long wave added at the end to create jets.
nmax = parsed_args["n_max"]
mmax = parsed_args["n_max"]
mjet = parsed_args["m_jet"]
automate = parsed_args["automate"]

# Arguments used for defining simulation time and output
if isnothing(parsed_args["t_end"])
    st = parsed_args["nt"]*dt
else
    st = parsed_args["t_end"]
end