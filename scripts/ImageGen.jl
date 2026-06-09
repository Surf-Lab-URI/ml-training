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
formatted_time = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

pix_vals = [10, 20, 30]
last_c = zeros(length(pix_vals))

for file in infiles
    global last_c
    global vars

    # Directory mode (-d): derive `vars` per-file from the combined filename so
    # outputs don't collide. (Single-file mode keeps the -v vars passed in.)
    if input_dir !== nothing
        vars = replace(basename(file), "_combined.jld2" => "")
    end

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

            # BUG-13 stopgap: dp is an integer, so achievable displacement is a
            # multiple of this sim's smax. Flag when the label is off (hot sim
            # floored to dp=1, or large quantization error) instead of silently
            # writing mislabeled data. See bugs.md BUG-13/BUG-14.
            actual = dp * smax
            rel_err = abs(actual - pix) / pix
            if rel_err > 0.2
                @warn "pix=$pix: achievable displacement ≈ $(round(actual, digits=1)) px " *
                      "(smax=$(round(smax, digits=2)), dp=$dp) — label off by $(round(Int, 100*rel_err))%"
            end

            # output files
            pix_dir = joinpath(data_root, "data", "visual", "pix" * string(pix))
            mkpath(pix_dir)

            if isnothing(name) && vars !== nothing
                out_file = joinpath(pix_dir, "image_pairs_" * vars * "_pix" * string(pix) * ".jld2")
            elseif name !== nothing
                vars = formatted_time * "_image_pairs_" * name
                out_file = joinpath(pix_dir, vars * "_pix" * string(pix) * ".jld2")
            else
                error("You must provide either the --name or the --vars argument.")
            end

            # ONE-PAIR-PER-SIM (design §2): emit only the single most-developed
            # pair — B = the last frame (latest = most cascade-developed), A = B−dp.
            # The old loop wrote every frame-offset pair (~nframes−dp per sim),
            # which were near-duplicate, correlated samples (~37× redundancy).
            if length(frame_keys) <= dp
                @warn "pix=$pix: only $(length(frame_keys)) frames but dp=$dp — cannot form a pair, skipping"
                continue
            end

            keyA = frame_keys[end - dp]
            keyB = frame_keys[end]

            jldopen(out_file, "a+") do fout
                c = Int(last_c[i_pix]) + 1

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
                    xlim = (0.0, 512.0),    # must match sim grid extent (BUG-15)
                    ylim = (0.0, 512.0),    # not (0, 2π) — particles live in [0, 512)
                    σₚ = 1.2,
                    Δt_pair = Δt_pair
                )

                if save_pngs == true
                    save_image_png(imgA, imgB, c; out_dir = pix_dir, name = "pair")
                end

                last_c[i_pix] = last_c[i_pix] + 1
            end
        end
    end
end