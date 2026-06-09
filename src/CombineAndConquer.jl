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
        "--input_dir", "-i"
            help = "directory with multiple field and particle file pairs, all of which should be combined into single jld2 files"
            arg_type = String
            default = nothing
        "--output_dir", "-o"
            help = "directory for output files."
            arg_type = String
            default = nothing
        "--delete_input_files", "-d"
            help = "deletes all input files (particle files and field files)"
            action = :store_true

    end

    return parse_args(s)
end


parsed_args = parse_commandline()

particles_file = parsed_args["particles_file"]
fields_file = parsed_args["fields_file"]
sim_vars = parsed_args["sim_vars"]
input_dir = parsed_args["input_dir"]
delete_input_files = parsed_args["delete_input_files"]
output_dir = parsed_args["output_dir"]

if isnothing(output_dir)
    # Honor the configurable output root (PIV_OUT_DIR) so HPC array tasks write
    # the combined file into their own scratch — matching 2DTurbulence.jl/ImageGen.jl.
    data_root = get(ENV, "PIV_OUT_DIR", projectdir())
    out_dir = joinpath(data_root, "data", "binary") * "/"
else
    out_dir = output_dir
end

if input_dir !== nothing
    particle_files = filter(f -> endswith(f, "_particles.jld2"), readdir(input_dir, join = true))
    jld2_files = filter(f -> endswith(f, ".jld2"), readdir(input_dir, join = true))
    combined_files = filter(f -> endswith(f, "_combined.jld2"), readdir(input_dir, join = true))
    field_files = setdiff(jld2_files, particle_files)
    field_files = setdiff(field_files, combined_files)
else
    field_files = [fields_file]
    particle_files = [particles_file]
end

# --- Combining Data into a Single File ---

for (ff, pf) in zip(field_files, particle_files)
    fields = load(ff)
    particles = load(pf)
    global sim_vars
    if isnothing(sim_vars)
        vars = split(ff, '/')[end]
        vars = vars[1:findlast(isequal('.'),vars)-1]
    else
        vars = sim_vars
    end

    function prefix_keys(prefix, data)
        return Dict(Symbol(prefix * "/" * key) => value for (key, value) in data)
    end

    combined_file = merge(prefix_keys("fields", fields), prefix_keys("particles", particles))

    jldsave(out_dir * vars * "_combined.jld2"; combined_file...)

    # removing old files
    if delete_input_files
        rm(fields_file)
        rm(particles_file)
    end
end