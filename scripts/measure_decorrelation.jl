using DrWatson
@quickactivate "ml-training"

using JLD2
using Statistics
using Printf

# ---------------------------------------------------------------------------
# How many frames per simulation are actually INDEPENDENT?
#
# Adjacent saved frames are near-duplicates (the save interval dt is far finer
# than the flow's eddy-turnover time). This script measures the velocity-field
# autocorrelation as a function of frame gap, so you can read off the
# correlation time τ — the spacing at which two frames become "distinct."
#
#   frames-per-independent-sample ≈ (gap where correlation first drops below ~1/e)
#
# Usage:
#   julia --project=. scripts/measure_decorrelation.jl
#   julia --project=. scripts/measure_decorrelation.jl path/to/<name>_combined.jld2
# ---------------------------------------------------------------------------

# --- Pick a combined file: first CLI arg, else the first one in data/binary/ ---
function default_combined_file()
    bindir = projectdir() * "/data/binary/"
    isdir(bindir) || error("No data/binary/ directory found — run a simulation first.")
    files = filter(f -> endswith(f, "_combined.jld2"), readdir(bindir, join = true))
    isempty(files) && error("No *_combined.jld2 files in $bindir — run a simulation first.")
    return first(sort(files))
end
file = isempty(ARGS) ? default_combined_file() : ARGS[1]

# --- helpers (self-contained, mirror ImageGenFunc.jl) ---
get_frame_keys(f) = sort!(filter(k -> k != "serialized", collect(keys(f["fields/timeseries/u"]))),
                          by = k -> parse(Int, k))
load_time(f, k) = f["fields/timeseries/t/$k"]
function load_uv(f, k)
    u = Array(f["fields/timeseries/u/$k"]); v = Array(f["fields/timeseries/v/$k"])
    ndims(u) == 3 && size(u, 3) == 1 && (u = dropdims(u, dims = 3))
    ndims(v) == 3 && size(v, 3) == 1 && (v = dropdims(v, dims = 3))
    return u, v
end

# Pearson correlation between two flow fields, stacking u and v into one vector.
# Pearson subtracts the mean and normalizes by std, so it measures PATTERN
# similarity independent of overall amplitude — i.e. it ignores the slow energy
# decay and reports only "has the flow reorganized?".
function field_corr(uvA, uvB)
    a = vcat(vec(uvA[1]), vec(uvA[2]))
    b = vcat(vec(uvB[1]), vec(uvB[2]))
    ā = mean(a); b̄ = mean(b)
    da = a .- ā; db = b .- b̄
    denom = sqrt(sum(da .^ 2)) * sqrt(sum(db .^ 2))
    return denom == 0 ? 1.0 : sum(da .* db) / denom
end

println("File: ", basename(file))

jldopen(file, "r") do f
    keys_ = get_frame_keys(f)
    nf = length(keys_)
    nf < 2 && error("Need at least 2 frames; this file has $nf.")

    # mean save interval (frames are evenly spaced, but average to be safe)
    times = [load_time(f, k) for k in keys_]
    Δt = (times[end] - times[1]) / (nf - 1)

    # preload all velocity fields once
    fields = [load_uv(f, k) for k in keys_]

    @printf("frames: %d   Δt(save): %.4f   total span: %.3f\n\n", nf, Δt, times[end] - times[1])
    println("Autocorrelation of the velocity field vs frame gap")
    println("(averaged over every starting frame that admits that gap):\n")
    println("gap   Δtime    corr     #pairs   ")
    println("---   ------   ------   ------   ")

    corr_by_gap = Float64[]
    for gap in 1:(nf - 1)
        cs = Float64[]
        for i in 1:(nf - gap)
            push!(cs, field_corr(fields[i], fields[i + gap]))
        end
        c = mean(cs)
        push!(corr_by_gap, c)
        bar = repeat("█", clamp(round(Int, c * 20), 0, 20))
        @printf("%-3d   %6.3f   %+.3f   %-6d   %s\n", gap, gap * Δt, c, length(cs), bar)
    end

    # --- find where correlation crosses the standard thresholds ---
    function first_crossing(thresh)
        for gap in 1:length(corr_by_gap)
            if corr_by_gap[gap] < thresh
                return gap
            end
        end
        return nothing
    end

    e_gap   = first_crossing(1 / ℯ)   # ≈ 0.368 — the classic correlation-time definition
    half_gap = first_crossing(0.5)

    println("\n--- Correlation time τ ---")
    if half_gap === nothing
        @printf("corr never drops below 0.50 within %d frames — run is too short to resolve τ.\n", nf - 1)
    else
        @printf("crosses 0.50  at gap %d frames  (τ ≈ %.3f)\n", half_gap, half_gap * Δt)
    end
    if e_gap === nothing
        @printf("corr never drops below 1/e (0.368) within %d frames — flow stays correlated across the whole run.\n", nf - 1)
        println("\n⇒ Effectively only ~1 independent snapshot in this run. Use MANY sims, not a longer one.")
    else
        @printf("crosses 1/e   at gap %d frames  (τ ≈ %.3f)\n", e_gap, e_gap * Δt)
        indep = max(1, div(nf - 1, e_gap))
        println()
        @printf("⇒ Frames are ~independent every %d saved frames.\n", e_gap)
        @printf("⇒ This %d-frame run holds ~%d statistically independent snapshot(s).\n", nf, indep)
        @printf("⇒ Saving frames closer than every %d is oversampling (near-duplicate pairs).\n", e_gap)
    end
end
