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

jldopen(file, "r") do fin

    frame_keys = get_frame_keys(fin)

    # computing image spacing
    t1 = load_time(fin, frame_keys[1])
    t2 = load_time(fin, frame_keys[2])
    Δt = t2 - t1

    u, v = load_field_frame(fin, frame_keys[1])

    speed = sqrt.(u.^2 .+ v.^2)
    smax = maximum(speed) * Δt

    pix_vals = [10, 15, 20, 25, 30]
    
    for pix in pix_vals
        dp = max(1, Int(floor(pix / smax)))

        # output files
        pix_dir = joinpath(projectdir(), "data", "visual", vars, "pix" * string(pix))
        mkpath(pix_dir)

        out_file = joinpath(pix_dir, "image_pairs_" * vars * "_pix" * string(pix) * ".jld2")

        jldopen(out_file, "w") do fout
            n_pairs = length(frame_keys) - dp

            for p in 1:n_pairs
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
                    p;
                    width = 512,
                    height = 512,
                    xlim = (0.0, 2π),
                    ylim = (0.0, 2π),
                    σₚ = 1.2,
                    Δt_pair = Δt_pair
                )

                if save_pngs == true
                    save_image_png(imgA, imgB, p;
                        out_dir = joinpath(projectdir(), "data", "visual", vars, "pix" * string(pix)),
                        name = "pair")
                elseif p == 1 || p == n_pairs
                    save_image_png(imgA, imgB, p;
                        out_dir = joinpath(projectdir(), "data", "visual", vars, "pix" * string(pix)),
                        name = "pair")
                end
            end
        end
    end
end