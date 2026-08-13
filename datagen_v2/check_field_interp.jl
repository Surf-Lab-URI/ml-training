#=
Measure the error introduced by interpolating the VELOCITY FIELD between saved frames.

This is the one gap the local self-test could not close. `FracFrame.field_at` interpolates u,v
linearly in time, and the mock file used for testing has a *steady* field — so that path was exact
by construction and therefore untested. On real turbulence the field evolves between saves, and any
error lands directly in the uB/vB labels of every v2 sample.

Run this on a REAL combined file before the production run.

Method: leave-one-out. Reconstruct saved frame i from frames i-1 and i+1 without using i, and
compare against the truth. That spans a doubled interval, so it is a conservative upper bound —
roughly 4x the error of interpolating between adjacent frames, for a smooth field.

    julia --project=. datagen_v2/check_field_interp.jl /path/to/seed7.jld2 [more.jld2 ...]

Reports the field error, and what it means for the label in px, which is the number that matters.
=#
using JLD2, Statistics, Printf

include(joinpath(@__DIR__, "FracFrame.jl"))
using .FracFrame

if isempty(ARGS)
    println("usage: julia --project=. datagen_v2/check_field_interp.jl <combined.jld2> [more...]")
    println()
    println("find the runs that actually have combined/ with:")
    println("  ls -d /project/pi_nicholas_pizzo_uri_edu/arup/piv_2dturb_dataset/run_*/combined")
    exit(1)
end

let missing = filter(p -> !isfile(p), ARGS)
    if !isempty(missing)
        println("ERROR: file(s) not found:")
        for p in missing
            println("  ", p)
        end
        println()
        if any(p -> startswith(p, "/combined"), missing)
            println("The path starts with /combined — the RUN variable was empty.")
            println("Set it to a real directory first (do not paste <stamp> literally):")
            println("  ls -d /project/pi_nicholas_pizzo_uri_edu/arup/piv_2dturb_dataset/run_*/combined")
        end
        exit(1)
    end
end

@printf("%-26s %7s %10s %10s %10s %12s\n",
        "file", "frames", "|u| rms", "lin err", "cubic err", "label err px")

tot_lin = Float64[]; tot_cub = Float64[]; tot_px = Float64[]

for path in ARGS
    jldopen(path, "r") do f
        keys_ = FracFrame.frame_keys_of(f)
        n = length(keys_)
        n < 5 && (@warn "only $n frames in $(basename(path)) — skipping"; return)

        dt = FracFrame._time(f, keys_[2]) - FracFrame._time(f, keys_[1])

        lin = Float64[]; cub = Float64[]; mag = Float64[]
        for i in 3:(n - 2)
            ut, vt = FracFrame._field(f, keys_[i])
            um, vm = FracFrame._field(f, keys_[i - 1])
            up, vp = FracFrame._field(f, keys_[i + 1])

            ul = 0.5 .* (um .+ up); vl = 0.5 .* (vm .+ vp)
            push!(lin, sqrt(mean((ul .- ut) .^ 2 .+ (vl .- vt) .^ 2)))

            umm, vmm = FracFrame._field(f, keys_[i - 2])
            upp, vpp = FracFrame._field(f, keys_[i + 2])
            # 4-point (Catmull-Rom) midpoint of the [i-1, i+1] span
            uc = @. (-0.0625) * umm + 0.5625 * um + 0.5625 * up - 0.0625 * upp
            vc = @. (-0.0625) * vmm + 0.5625 * vm + 0.5625 * vp - 0.0625 * vpp
            push!(cub, sqrt(mean((uc .- ut) .^ 2 .+ (vc .- vt) .^ 2)))

            push!(mag, sqrt(mean(ut .^ 2 .+ vt .^ 2)))
        end

        # label = (uA+uB)/2 * dt_pair, in px. An error ε in uB contributes ε/2 * dt_pair.
        # Use the largest v2 target (median 30 px) as the worst case for dt_pair.
        dt_pair_worst = 13.3 * dt          # dp ~ 13 saved frames for the median-30 bin
        px_err = mean(cub) / 2 * dt_pair_worst

        append!(tot_lin, lin); append!(tot_cub, cub); push!(tot_px, px_err)
        @printf("%-26s %7d %10.4f %10.5f %10.5f %12.4f\n",
                first(basename(path), 26), n, mean(mag), mean(lin), mean(cub), px_err)
    end
end

if !isempty(tot_cub)
    println()
    @printf("pooled: linear %.5f   cubic %.5f   (cubic is %.1fx better)\n",
            mean(tot_lin), mean(tot_cub), mean(tot_lin) / mean(tot_cub))
    @printf("worst-case label error from field interpolation: %.4f px\n", maximum(tot_px))
    println()
    if maximum(tot_px) < 0.05
        println("PASS — well below label precision. Linear field interpolation is fine.")
    elseif maximum(tot_px) < 0.2
        println("MARGINAL — consider switching field_at to the 4-point cubic before a full run.")
    else
        println("FAIL — field interpolation would corrupt the labels. Switch to cubic, or save")
        println("       simulation frames more often, before generating v2 data.")
    end
end
