#=
ImageGenV3 — image pairs with the tank's measured near-surface SHEAR LAYER.

Why a v3
--------
v2 fixed the displacement *magnitude* and it did not transfer: the wide-range model was worse on the
tank everywhere. Characterising 60 frames of the traditional-PIV collection showed why -- the
mismatch was never magnitude, it was STRUCTURE (DATA_REQUIREMENTS R6):

    tank, median displacement:  8.06 px at 0-1 mm  ->  0.38 px at 24-40 mm   (21x over ~8 mm)
    tank, WHOLE-FRAME median:   0.38 px            (our slowest bin, med03, is already 8x faster)
    tank, TOP-2mm median:       7.88 px            (matched almost exactly by med09)

Our data is homogeneous 2-D periodic turbulence with no surface, so it has no such profile. v3 keeps
the simulation's turbulent structure and imposes the measured profile on it.

What changes from v2
--------------------
  * a wavy FREE SURFACE with an air region masked to exactly zero, as the lab's own images are;
  * a depth-dependent velocity scaling reproducing the 21x profile (datagen_v3/ShearProfile.jl);
  * bins indexed by TOP-2mm median displacement, not whole-frame median -- with a shear layer the
    whole-frame median is not a meaningful label, and the top 2 mm is the region we care about;
  * particles re-advected by the SCALED field, so images and labels stay consistent.

Honest limitations, both real
-----------------------------
  1. The scaled field is NOT divergence-free and is not a solution of the free-surface equations.
     Train with lambda_div reduced or 0, or the penalty fights the imposed shear.
  2. Frame B is produced by warping frame A through the scaled displacement (a frozen-field warp),
     so the pair contains no unsteadiness: a perfect algorithm could recover it exactly. Real pairs
     decorrelate slightly over dt. v2's fractional-frame sampling did carry that unsteadiness; this
     trades it for the profile. Whether that trade is worth it is exactly what the pilot tests.

Usage
-----
    julia --project=. datagen_v3/ImageGenV3.jl -f <combined.jld2> -v tag
    julia --project=. datagen_v3/ImageGenV3.jl -d <combined_dir> -s 12000
=#

using DrWatson
@quickactivate "ml-training"

include(projectdir() * "/src/ImageGenFunc.jl")      # parses ARGS; renderer, appearance, dropout
include(joinpath(@__DIR__, "..", "datagen_v2", "FracFrame.jl"))
include(joinpath(@__DIR__, "ShearProfile.jl"))
using .FracFrame, .ShearProfile

using JLD2, Printf, Random, Statistics, Dates, TOML

# Bin targets: the median displacement in the TOP 2 mm. The tank's measured value is 7.88 px on
# average over 60 frames, ranging 0.97-22.03 between frames, so this set brackets what the tank
# actually does rather than extrapolating past it -- the mistake v2 made at the high end.
const SURF_TARGETS = Params.get_vector("bins.v3.surface_medians", [2.0, 4.0, 6.0, 9.0, 12.0, 16.0, 20.0, 24.0])
const TOL     = Params.get("bins.v3.tolerance", 0.05)
const AIRFRAC = Params.get("bins.v3.air_fraction", 0.181)     # measured 18.1%, sd 0.1
const DX_MM   = Params.get("bins.v3.dx_mm", 0.0565454946380008)
const BIN_NAME = m -> @sprintf("surf%02d", round(Int, m))

infiles = if input_dir !== nothing
    filter(f -> endswith(f, "_combined.jld2"), readdir(input_dir, join = true))
elseif file !== nothing
    [file]
else
    error("Must provide either --combined_file (-f) or --input_dir (-d)")
end
isempty(infiles) && error("no *_combined.jld2 files found")

@info "v3 top-2mm median targets (px): $(join(Int.(round.(SURF_TARGETS)), ", "))  tol=$TOL  air=$(AIRFRAC)"

"""Bilinear sample of a field at (row, col) positions, periodic in both axes."""
function sample_field(F::AbstractMatrix, r::AbstractVector, c::AbstractVector)
    H, W = size(F)
    out = similar(r)
    @inbounds for k in eachindex(r)
        rr = mod(r[k] - 1, H) + 1; cc = mod(c[k] - 1, W) + 1
        i0 = floor(Int, rr); j0 = floor(Int, cc)
        fr = rr - i0; fc = cc - j0
        i1 = i0 == H ? 1 : i0 + 1; j1 = j0 == W ? 1 : j0 + 1
        out[k] = F[i0,j0]*(1-fr)*(1-fc) + F[i1,j0]*fr*(1-fc) +
                 F[i0,j1]*(1-fr)*fc     + F[i1,j1]*fr*fc
    end
    return out
end

"""Median |displacement| among particles inside the top `band` mm, for scale factor `s`."""
function top2_median(dxf, dyf, xp, yp, surf, s, band = 2.0)
    vals = Float64[]
    @inbounds for k in eachindex(xp)
        j = clamp(round(Int, xp[k]) + 1, 1, length(surf))
        d = (yp[k] - surf[j]) * DX_MM
        (d >= 0 && d < band) || continue
        push!(vals, s * hypot(dxf[k], dyf[k]))
    end
    return isempty(vals) ? NaN : median(vals)
end

formatted_time = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
counters = Dict{String,Int}(); nwritten = 0; nskipped = 0

for infile in infiles
    global vars, nwritten, nskipped
    if input_dir !== nothing
        vars = replace(basename(infile), "_combined.jld2" => "")
    end

    jldopen(infile, "r") do fin
        keys_ = FracFrame.frame_keys_of(fin)
        nfr = length(keys_)
        nfr < 4 && (@warn "only $nfr frames in $(basename(infile)) — skipping"; return)

        iA = max(1, nfr - 1)
        xA0, yA0 = FracFrame._particles(fin, keys_[iA])
        u, v = FracFrame._field(fin, keys_[iA])
        H, W = size(u)

        # free surface: superposed components fitted to the lab's slope statistics, placed so the
        # air fraction matches the measured 18.1%
        xs = collect(0.0:(W - 1))
        surf = AIRFRAC * H .+ surface_row(xs, W)

        # Particles above the interface must be DROPPED, not merely given zero velocity. The lab's
        # air region contains no tracers at all; leaving them in would teach the model that visible
        # particles move zero, which is worse than not showing it the surface in the first place.
        inwater = falses(length(xA0))
        @inbounds for k in eachindex(xA0)
            j = clamp(round(Int, xA0[k]) + 1, 1, length(surf))
            inwater[k] = yA0[k] >= surf[j]
        end
        wet = findall(inwater)
        k_use = min(k_particles, length(wet))
        idx = wet[randperm(rng, length(wet))[1:k_use]]
        xA = xA0[idx]; yA = yA0[idx]
        @printf("  particles: %d of %d in water, using %d\n", length(wet), length(xA0), k_use)

        # impose the measured profile on the simulation's own field
        us, vs, water = apply_shear(u, v, surf, DX_MM)

        # displacement at each particle, from the SCALED field
        dxp = sample_field(us, yA .+ 1, xA .+ 1)
        dyp = sample_field(vs, yA .+ 1, xA .+ 1)

        meta = Dict{String,Any}("generator" => "ImageGenV3", "target_stat" => "top2_median",
                                "air_fraction" => 1 - mean(water), "dx_mm" => DX_MM,
                                "nframes" => nfr, "frameA_index" => iA,
                                "divergence_free" => false, "frozen_field_warp" => true)

        @printf("\n%s\n  frames %d  air %.1f%%\n", basename(infile), nfr, 100*(1 - mean(water)))
        @printf("  %-8s %8s %10s %8s %8s  %s\n", "bin", "target", "achieved", "p99", "max", "status")

        for m in SURF_TARGETS
            # solve the scale s that puts the TOP-2mm median at the target. The median is linear in
            # s, so one evaluation gives it exactly -- no iteration needed.
            base = top2_median(dxp, dyp, xA, yA, surf, 1.0)
            (isnan(base) || base <= 0) && (nskipped += 1; continue)
            s = m / base
            got = top2_median(dxp, dyp, xA, yA, surf, s)
            rel = abs(got - m) / m
            if rel > TOL
                @printf("  %-8s %8.1f %10.2f %8s %8s  OFF by %.0f%% — skipped\n",
                        BIN_NAME(m), m, got, "-", "-", 100rel)
                nskipped += 1; continue
            end

            dxs = s .* dxp; dys = s .* dyp
            xB = xA .+ dxs; yB = yA .+ dys                    # frozen-field warp
            uA = s .* us;  vA = s .* vs                       # labels: the field actually used
            mags = hypot.(dxs, dys)
            @printf("  %-8s %8.1f %10.2f %8.2f %8.2f  ok\n", BIN_NAME(m), m, got,
                    quantile(mags, 0.99), maximum(mags))

            bin = BIN_NAME(m)
            outdir = joinpath(data_root, "data", "visual_v3", bin); mkpath(outdir)
            out_file = if name !== nothing
                joinpath(outdir, formatted_time * "_image_pairs_" * name * "_" * bin * ".jld2")
            elseif vars !== nothing
                joinpath(outdir, "image_pairs_" * vars * "_" * bin * ".jld2")
            else
                error("provide --name or --vars")
            end

            jldopen(out_file, "a+") do fout
                c = get(counters, bin, 0) + 1
                bg, pk, sp, nσ = appearance_draw(rng)
                pdrop = dropout_draw(rng)
                kA, kB = keep_masks(rng, length(xA), pdrop)
                meta["$(bin)_dropout"] = pdrop
                meta["$(bin)_scale"]   = s
                meta["$(bin)_top2med"] = got
                make_image_pair(fout, xA[kA], yA[kA], xB[kB], yB[kB], uA, vA, uA, vA, c;
                                width = img_width, height = img_height,
                                xlim = img_xlim, ylim = img_ylim,
                                σₚ = sp, Δt_pair = 1.0,
                                background = bg, peak = pk, noise_σ = nσ, rng = rng)
                counters[bin] = c
            end
            nwritten += 1
        end

        metafile = joinpath(data_root, "data", "binary", "metadata_v3" * vars * ".toml")
        mkpath(dirname(metafile))
        open(metafile, "w") do io; TOML.print(io, meta) end
    end
end

@info "v3 done: $nwritten samples written, $nskipped skipped"
