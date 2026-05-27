using DrWatson
@quickactivate "ml-training"

include(projectdir()*"/src/ImageGenFunc.jl")

using JLD2
using ArgParse
using Oceananigans
using Images
using FileIO
using ImageIO
using Printf
using Random
using Statistics
using Dates
using CUDA
using SpecialFunctions

# ---Main Loop to Generate Image Pairs as JLD2 Files---

infiles = if input_dir !== nothing
    filter(f -> endswith(f, "_combined.jld2"), readdir(input_dir, join = true))
elseif file !== nothing
    [file]
else
    error("Must provide either --combined_file (-f) or --input_dir (-d)")
end
last_c = 0
formatted_time = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

pix_vals = [3, 5, 11]
last_c = zeros(length(pix_vals))

for file in infiles
    global last_c
    global vars
    jldopen(file, "r") do fin

        frame_keys = get_frame_keys(fin)

        # computing image spacing
        t1 = load_time(fin, frame_keys[1])
        t2 = load_time(fin, frame_keys[2])
        Δt = t2 - t1

        u, v = load_field_frame(fin, frame_keys[1])

        speed = sqrt.(u.^2 .+ v.^2)
        smax = maximum(speed) * Δt

        
        for (i_pix,pix) in zip(collect(1:length(pix_vals)), pix_vals)
            dp = max(1, Int(floor(pix / smax)))

            # output files
            pix_dir = joinpath(projectdir(), "data", "visual", "pix" * string(pix))
            mkpath(pix_dir)

            if isnothing(name) && vars !== nothing
                out_file = joinpath(pix_dir, "image_pairs_" * vars * "_pix" * string(pix) * ".jld2")
            elseif name !== nothing
                vars = formatted_time * "_image_pairs_" * name
                out_file = joinpath(pix_dir, vars * "_pix" * string(pix) * ".jld2")
            else
                error("You must provide either the --name or the --vars argument.")
            end

            jldopen(out_file, "a+") do fout
                n_pairs = length(frame_keys) - dp

                for p in 1:n_pairs
                    c = last_c[i_pix] + p
                    keyA = frame_keys[p]
                    keyB = frame_keys[p + dp]

                    xA, yA = load_particle_frame(fin, keyA)
                    xB, yB = load_particle_frame(fin, keyB)

                    xA, yA, xB, yB = subset_particles(xA, yA, xB, yB, k_particles, rng)

                    uA, vA = load_field_frame(fin, keyA)
                    uB, vB = load_field_frame(fin, keyB)

                    tA = load_time(fin, keyA)
                    tB = load_time(fin, keyB)
                    Δt_pair = tB - tA

                    imgA, imgB = make_image_pair(
                        fout,
                        xA, yA,
                        xB, yB,
                        uA, vA,
                        uB, vB,
                        c;
                        width = 512,
                        height = 512,
                        xlim = (0.0, 2π),
                        ylim = (0.0, 2π),
                        σₚ = 1.2,
                        Δt_pair = Δt_pair
                    )

                    if save_pngs == true
                        save_image_png(imgA, imgB, c;
                            out_dir = joinpath(projectdir(), "data", "visual", "pix" * string(pix)),
                            name = "pair")
                    elseif p == 1 || p == n_pairs
                        save_image_png(imgA, imgB, c;
                            out_dir = joinpath(projectdir(), "data", "visual", "pix" * string(pix)),
                            name = "pair")
                    end
                end
                last_c[i_pix] = last_c[i_pix] + n_pairs
                
            end
        end
    end
end