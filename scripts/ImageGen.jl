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
using TOML   # to append realized-displacement info to each sim's metadata sidecar
# using CUDA   # CPU-only run: omit to avoid CUDA_Runtime_jll init failures on CPU nodes + faster precompile. Re-enable for GPU.
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

# Image appearance is resolved once in src/ImageGenFunc.jl from params.toml [imaging.appearance];
# `appearance_draw(rng)` below returns one pair's values. Ranges were validated against
# ExpLCL_1_03 (bg~50, contrast~4x, particle ~2px, noise σ~5).

# Bin targets, from params.toml [bins.v1].pix_values. These are TARGET MAXIMUM displacements and
# are reached only to the nearest whole frame — that quantisation is BUG-13, and it is why
# datagen_v2/ImageGenV2.jl exists. Prefer v2 for new data.
pix_vals = Params.get_vector("bins.v1.pix_values", [10, 20, 30])
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

        # ONE PAIR PER SIM, SHARED FIRST FRAME (standard PIV convention): all bins share
        # the SAME first image A; the second image B is dp frames later (B = A + dp).
        # Anchor A so the LARGEST displacement bin lands its B on the final (most-developed)
        # frame: iA = nframes − max(dp). A is common; B moves forward per bin.
        dps = [max(1, Int(floor(pix / smax))) for pix in pix_vals]
        dp_max = maximum(dps)
        iA = length(frame_keys) - dp_max
        if iA < 1
            @warn "Only $(length(frame_keys)) frames but max dp=$dp_max — cannot form pairs, skipping $(basename(file))"
            return
        end
        keyA = frame_keys[iA]

        # Draw ONE particle subset (shared indices) so the first image is IDENTICAL across
        # bins; the same tracked particles are then followed to each bin's B frame.
        xA_full, yA_full = load_particle_frame(fin, keyA)
        npool = length(xA_full)
        k_use = min(k_particles, npool)
        idx = randperm(rng, npool)[1:k_use]
        xA, yA = xA_full[idx], yA_full[idx]
        uA, vA = load_field_frame(fin, keyA)   # label field at A (shared across bins)
        tA = load_time(fin, keyA)

        # realized-displacement metadata for this sim (for filtering)
        disp = Dict{String,Any}("smax" => smax, "frameA_index" => iA, "nframes" => length(frame_keys))

        for (i_pix,pix) in zip(collect(1:length(pix_vals)), pix_vals)
            dp = dps[i_pix]

            # BUG-13 stopgap: dp is an integer, so achievable displacement is a multiple
            # of this sim's smax. Flag when the label is off instead of silently mislabeling.
            actual = dp * smax
            rel_err = abs(actual - pix) / pix
            if rel_err > 0.2
                @warn "pix=$pix: achievable displacement ≈ $(round(actual, digits=1)) px " *
                      "(smax=$(round(smax, digits=2)), dp=$dp) — label off by $(round(Int, 100*rel_err))%"
            end
            disp["dp_pix$(pix)"]          = dp
            disp["realized_px_pix$(pix)"] = actual

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

            keyB = frame_keys[iA + dp]              # shared A; B = A + dp
            xB_full, yB_full = load_particle_frame(fin, keyB)
            xB, yB = xB_full[idx], yB_full[idx]     # same particles as A, advected to B
            uB, vB = load_field_frame(fin, keyB)
            Δt_pair = load_time(fin, keyB) - tA

            jldopen(out_file, "a+") do fout
                c = Int(last_c[i_pix]) + 1

                bg, pk, sp, nσ = appearance_draw(rng)   # params.toml [imaging.appearance]
                imgA, imgB = make_image_pair(
                    fout,
                    xA, yA,
                    xB, yB,
                    uA, vA,
                    uB, vB,
                    c;
                    width = img_width,      # params.toml [imaging]; must match the sim grid
                    height = img_height,    # extent, or particles land off-frame (BUG-15)
                    xlim = img_xlim,
                    ylim = img_ylim,
                    σₚ = sp,
                    Δt_pair = Δt_pair,
                    background = bg, peak = pk, noise_σ = nσ, rng = rng
                )

                if save_pngs == true
                    save_image_png(imgA, imgB, c; out_dir = pix_dir, name = "pair")
                end

                last_c[i_pix] = last_c[i_pix] + 1
            end
        end

        # Append the realized-displacement info to this sim's metadata sidecar so the
        # manifest can be filtered by ACTUAL displacement (not just the nominal bin name).
        metafile = joinpath(data_root, "data", "binary", "metadata" * vars * ".toml")
        try
            meta = isfile(metafile) ? TOML.parsefile(metafile) : Dict{String,Any}()
            meta["displacement"] = disp
            open(metafile, "w") do io
                TOML.print(io, meta)
            end
        catch err
            @warn "Could not update metadata with displacement info" exception=err
        end
    end
end