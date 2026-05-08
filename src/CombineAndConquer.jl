using DrWatson
@quickactivate "ml-training"

using JLD2
using ArgParse
using Oceananigans

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--fields_file", "-f"
            help = "path to JLD2 file containing field timeseries"
            arg_type = String
            default = nothing
        "--particles_file", "-p"
            help = "path to JLD2 file containing particle timeseries"
            arg_type = String
            default = nothing
        "--sim_vars", "-s"
            help = "additional simulation variables to include in combined file, as a comma-separated list of variable names (e.g. 'u,v')"
            arg_type = String
            default = nothing
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

particles_file = parsed_args["particles_file"]
fields_file = parsed_args["fields_file"]
vars = parsed_args["sim_vars"]
out_dir = projectdir() * "/data/binary/"

# --- Combining Data into a Single File ---

fields = load(fields_file)
particles = load(particles_file)

function prefix_keys(prefix, data)
    return Dict(Symbol(prefix * "/" * key) => value for (key, value) in data)
end

combined_file = merge(prefix_keys("fields", fields), prefix_keys("particles", particles))

jldsave(out_dir * "combined" * vars * ".jld2"; combined_file...)