using DrWatson
@quickactivate "ml-training"

using JLD2
using Statistics
using Printf

# --- Which combined file to inspect (pass a path as the first arg, or edit this default) ---
default_file = projectdir() * "/data/binary/_2026-05-26T19:28:30.383_2DT-A266.0812938162001-nmax21-mjet2_combined.jld2"
file = isempty(ARGS) ? default_file : ARGS[1]

# --- pix bins to test (mirror scripts/ImageGen.jl line 31) ---
pix_vals = [3, 5, 11]

# --- helpers replicating ImageGenFunc.jl, but self-contained (no include, no ARGS parsing) ---
get_frame_keys(f) = sort!(filter(k -> k != "serialized", collect(keys(f["fields/timeseries/u"]))),
                          by = k -> parse(Int, k))
load_time(f, k)  = f["fields/timeseries/t/$k"]
function load_uv(f, k)
    u = Array(f["fields/timeseries/u/$k"]); v = Array(f["fields/timeseries/v/$k"])
    ndims(u) == 3 && size(u,3) == 1 && (u = dropdims(u, dims=3))
    ndims(v) == 3 && size(v,3) == 1 && (v = dropdims(v, dims=3))
    return u, v
end

println("File: ", basename(file))

jldopen(file, "r") do f
    keys_ = get_frame_keys(f)
    nf = length(keys_)

    # Δt between saved frames (ImageGen uses frames 1 and 2)
    Δt = load_time(f, keys_[2]) - load_time(f, keys_[1])

    # smax exactly as ImageGen.jl computes it: frame-1 speed * Δt
    u1, v1 = load_uv(f, keys_[1])
    smax_frame1 = maximum(sqrt.(u1.^2 .+ v1.^2)) * Δt

    # worst case across ALL frames (speed grows/decays over the run)
    smax_global = 0.0
    for k in keys_
        u, v = load_uv(f, k)
        smax_global = max(smax_global, maximum(sqrt.(u.^2 .+ v.^2)) * Δt)
    end

    @printf("frames: %d   Δt(save): %.4f\n", nf, Δt)
    @printf("smax (frame 1, what ImageGen uses to set dp): %.3f px\n", smax_frame1)
    @printf("smax (max over all %d frames, true worst case): %.3f px\n", nf, smax_global)
    println("\nFloor on achievable displacement = smax. Achievable bins are multiples: smax, 2·smax, 3·smax, ...\n")

    println("Per-bin breakdown (using frame-1 smax, like the real code):")
    println(rpad("pix", 6), rpad("dp", 5), rpad("actual = dp·smax", 20), "status")
    seen = Dict{Int,Vector{Int}}()   # dp -> list of pix that collapse onto it
    for pix in pix_vals
        dp = max(1, Int(floor(pix / smax_frame1)))
        actual = dp * smax_frame1
        push!(get!(seen, dp, Int[]), pix)
        status = pix < smax_frame1 ? "BELOW FLOOR (pix<smax: clamped, mislabeled)" :
                 abs(actual - pix) > 0.2*pix ? "achievable but quantized (label≉actual)" :
                 "ok"
        @printf("%-6d%-5d%-20.2f%s\n", pix, dp, actual, status)
    end

    collisions = [(dp, ps) for (dp, ps) in seen if length(ps) > 1]
    println()
    if isempty(collisions)
        println("No collisions: every bin maps to a distinct dp. ✅")
    else
        println("COLLISIONS (these bins produce IDENTICAL data — BUG-13): ⚠️")
        for (dp, ps) in collisions
            @printf("  dp=%d shared by pix bins %s\n", dp, join(ps, ", "))
        end
    end
end
