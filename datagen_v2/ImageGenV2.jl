#=
ImageGenV2 — image-pair generation with displacement targeted by MEDIAN, not by an integer frame gap.

Drop-in sibling of scripts/ImageGen.jl. Same CLI (it includes src/ImageGenFunc.jl, which parses ARGS),
same renderer, same output layout — only the choice of the B frame differs.

Why it exists
-------------
The old path takes `dp = max(1, floor(pix / smax))` saved frames, so achievable displacement is
quantised in steps of smax (~5 px) and a request below smax is silently written under the wrong label
(BUG-13). More importantly, the resulting dataset is specified by its *maximum* displacement while the
model is governed by where the distribution has **mass**: the current data reaches 31 px yet the model
breaks at 22 px, because only 3.4% of its pixels exceed 22 and none exceed 30.

v2 therefore:
  * picks the B frame at a *fractional* index via FracFrame (BUG-14 fix), so any target is reachable;
  * targets the **median** displacement, which is stable across sims (max/median has 32% spread);
  * **refuses to write a mislabelled sample** — if the target cannot be reached with the frames
    available it warns and skips, rather than writing it under the requested name;
  * records the full distribution per sample, so the dataset can be reasoned about later.

Bins (median displacement, px): 3 6 9 12 16 20 26 30  ->  max ~ 5 10 15 20 27 33 43 50.
Rationale and the measurements behind them: datagen_v2/DATA_REQUIREMENTS.md.

Usage
-----
    julia --project=. datagen_v2/ImageGenV2.jl -d <combined_dir> -s <k_particles> --seed 7
    PIV_LAB_APPEARANCE=1 julia --project=. datagen_v2/ImageGenV2.jl -f <combined.jld2> -v tag

Environment
-----------
    PIV_OUT_DIR           output root (default: project dir), as ImageGen.jl
    PIV_LAB_APPEARANCE=1  lab-matched appearance, as ImageGen.jl
    PIV_V2_MEDIANS        comma-separated overrides, e.g. "3,9,20,30"
    PIV_V2_TOL            relative tolerance on the achieved median (default 0.05)
    PIV_V2_LOOSE=1        write out-of-tolerance samples (flagged) instead of skipping
=#

using DrWatson
@quickactivate "ml-training"

include(projectdir() * "/src/ImageGenFunc.jl")     # parses ARGS; defines loaders + make_image_pair
include(joinpath(@__DIR__, "FracFrame.jl"))
using .FracFrame

using JLD2, Printf, Random, Statistics, Dates, TOML

# Bin targets and solver tolerance, from params.toml [bins.v2]. PIV_V2_MEDIANS / PIV_V2_TOL /
# PIV_V2_LOOSE still override, so existing Slurm scripts keep working — see src/Params.jl.
const MEDIANS = Params.get_vector("bins.v2.medians", [3.0, 6.0, 9.0, 12.0, 16.0, 20.0, 26.0, 30.0])
const TOL   = Params.get("bins.v2.tolerance", 0.05)
const LOOSE = Params.get("bins.v2.write_out_of_tolerance", false)
const BIN_NAME = m -> @sprintf("med%02d", round(Int, m))

infiles = if input_dir !== nothing
    filter(f -> endswith(f, "_combined.jld2"), readdir(input_dir, join = true))
elseif file !== nothing
    [file]
else
    error("Must provide either --combined_file (-f) or --input_dir (-d)")
end
isempty(infiles) && error("no *_combined.jld2 files found")

# `lab_appearance` and `appearance_draw` come from src/ImageGenFunc.jl, which resolved them from
# params.toml [imaging.appearance] — one definition, shared with the v1 generator.
@info "v2 median targets (px): $(join(Int.(round.(MEDIANS)), ", "))  tol=$(TOL)  loose=$(LOOSE)"

formatted_time = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
counters = Dict{String,Int}()
nwritten = 0
nskipped = 0

for infile in infiles
    global vars, nwritten, nskipped

    if input_dir !== nothing
        vars = replace(basename(infile), "_combined.jld2" => "")
    end

    jldopen(infile, "r") do fin
        keys_ = FracFrame.frame_keys_of(fin)
        nfr = length(keys_)
        if nfr < 4
            @warn "only $nfr frames in $(basename(infile)); need ≥4 for cubic interpolation — skipping"
            return
        end

        Δt_save = FracFrame.time_at(fin, keys_, 2.0) - FracFrame.time_at(fin, keys_, 1.0)
        u1, v1 = FracFrame._field(fin, keys_[1])
        smax = maximum(sqrt.(u1 .^ 2 .+ v1 .^ 2)) * Δt_save

        # Shared particle subset, so frame A is identical across bins (standard PIV convention).
        xpool, _ = FracFrame._particles(fin, keys_[1])
        k_use = min(k_particles, length(xpool))
        idx = randperm(rng, length(xpool))[1:k_use]

        # Anchor A so the LARGEST target still fits before the final frame. dp depends on iA, and iA
        # depends on dp, so estimate once from an early anchor and then solve properly.
        dp_est, _, _, _ = solve_frac_dp(fin, keys_, 1, maximum(MEDIANS);
                                        stat = :median, idx = idx, tol = TOL)
        iA = clamp(nfr - ceil(Int, dp_est), 1, nfr - 1)

        xA, yA = particles_at(fin, keys_, float(iA); idx = idx)
        uA, vA = FracFrame._field(fin, keys_[iA])
        tA = FracFrame._time(fin, keys_[iA])

        meta = Dict{String,Any}("smax" => smax, "dt_save" => Δt_save,
                                "frameA_index" => iA, "nframes" => nfr,
                                "generator" => "ImageGenV2", "target_stat" => "median")

        @printf("\n%s\n  frames %d  smax %.2f px  anchor A = %d  (est dp_max %.2f)\n",
                basename(infile), nfr, smax, iA, dp_est)
        @printf("  %-7s %8s %8s %8s %8s %8s %7s %7s  %s\n",
                "bin", "target", "achieved", "p90", "p99", "max", ">22px", "dp", "status")

        for m in MEDIANS
            dp, st, iters, conv = solve_frac_dp(fin, keys_, iA, m;
                                                stat = :median, idx = idx, tol = TOL)
            rel = abs(st.median - m) / m
            ok  = conv && rel <= TOL

            status = ok ? "ok" : @sprintf("OFF by %.0f%%", 100rel)
            @printf("  %-7s %8.1f %8.2f %8.2f %8.2f %8.2f %6.1f%% %7.3f  %s\n",
                    BIN_NAME(m), m, st.median, st.p90, st.p99, st.max,
                    100 * st.frac_over_22, dp, status)

            if !ok && !LOOSE
                # This is exactly the BUG-13 failure mode: do NOT write it under a label it does
                # not have. Usually means the run is too short for this target.
                @warn "$(BIN_NAME(m)): achieved median $(round(st.median, digits=2)) px vs target $m — skipping (set PIV_V2_LOOSE=1 to write anyway)"
                nskipped += 1
                continue
            end

            xB, yB = particles_at(fin, keys_, iA + dp; idx = idx)
            uB, vB = field_at(fin, keys_, iA + dp)
            Δt_pair = time_at(fin, keys_, iA + dp) - tA

            bin = BIN_NAME(m)
            outdir = joinpath(data_root, "data", "visual_v2", bin)
            mkpath(outdir)
            out_file = if name !== nothing
                joinpath(outdir, formatted_time * "_image_pairs_" * name * "_" * bin * ".jld2")
            elseif vars !== nothing
                joinpath(outdir, "image_pairs_" * vars * "_" * bin * ".jld2")
            else
                error("provide --name or --vars")
            end

            jldopen(out_file, "a+") do fout
                c = get(counters, bin, 0) + 1
                bg, pk, sp, nσ = appearance_draw(rng)   # params.toml [imaging.appearance]
                make_image_pair(fout, xA, yA, xB, yB, uA, vA, uB, vB, c;
                                width = img_width, height = img_height,
                                xlim = img_xlim, ylim = img_ylim,           # BUG-15
                                σₚ = sp, Δt_pair = Δt_pair,
                                background = bg, peak = pk, noise_σ = nσ, rng = rng)
                counters[bin] = c
            end

            meta["$(bin)_dp"]        = dp
            meta["$(bin)_median"]    = st.median
            meta["$(bin)_p90"]       = st.p90
            meta["$(bin)_p99"]       = st.p99
            meta["$(bin)_max"]       = st.max
            meta["$(bin)_over20"]    = st.frac_over_20
            meta["$(bin)_over22"]    = st.frac_over_22
            meta["$(bin)_over24"]    = st.frac_over_24
            meta["$(bin)_dt_pair"]   = Δt_pair
            nwritten += 1
        end

        # sidecar next to the outputs, so a run can be audited without reopening every sample
        sdir = joinpath(data_root, "data", "visual_v2", "metadata")
        mkpath(sdir)
        open(joinpath(sdir, (vars === nothing ? "sim" : vars) * "_v2.toml"), "w") do io
            TOML.print(io, Dict("displacement_v2" => meta))
        end
    end
end

@printf("\ndone: %d samples written, %d skipped as out of tolerance\n", nwritten, nskipped)
