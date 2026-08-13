"""
    FracFrame

Fractional-frame sampling of a `*_combined.jld2` simulation — the root-cause fix for BUG-13.

The existing generator picks the B frame as `dp = max(1, floor(pix / smax))` saved frames after A.
`dp` is an integer, so achievable displacement is quantised in steps of `smax` (~5 px), and any
request below `smax` is unachievable yet still written under the requested label. That is BUG-13;
BUG-14 is this fix.

Here a virtual frame is synthesised at any *fractional* index `s`, so the exact Δt that produces a
requested displacement can be used. Particle positions are interpolated with Catmull-Rom (cubic)
and the velocity fields linearly in time.

Two things this module gets right that a naive implementation does not:

  * **Periodicity.** The sim is `topology=(Periodic, Periodic, Flat)` on `[0, 512)` (see BUG-15).
    A particle crossing the seam jumps 512 px between saves; interpolating that directly sends it
    backwards across the whole domain. Every frame is therefore unwrapped to a common branch
    before interpolation and re-wrapped afterwards.
  * **Displacement across the seam.** A→B displacement uses the minimum-image convention, valid
    while true displacement stays below L/2 = 256 px (ours are ≤ 50 px).

Loader functions mirror `src/ImageGenFunc.jl` so this module stays dependency-free and testable
without Oceananigans; keep them in sync if the file layout changes.
"""
module FracFrame

using JLD2, Statistics, Printf

export particles_at, field_at, time_at, displacement, disp_stats, solve_frac_dp,
       loo_error, frame_keys_of, DOMAIN_L

"Periodic extent of the simulation grid, in the same units as the stored particle positions."
const DOMAIN_L = 512.0

# ---------------------------------------------------------------- loaders (mirror ImageGenFunc)
frame_keys_of(file) = sort!(filter(k -> k != "serialized",
                                   collect(keys(file["fields/timeseries/u"]))),
                            by = k -> parse(Int, k))

function _particles(file, key)
    p = file["particles/timeseries/particles/$key"]
    return Array(p.x), Array(p.y)
end

function _field(file, key)
    u = Array(file["fields/timeseries/u/$key"])
    v = Array(file["fields/timeseries/v/$key"])
    ndims(u) == 3 && size(u, 3) == 1 && (u = dropdims(u, dims = 3))
    ndims(v) == 3 && size(v, 3) == 1 && (v = dropdims(v, dims = 3))
    return u, v
end

_time(file, key) = file["fields/timeseries/t/$key"]

# ---------------------------------------------------------------- periodic helpers
"Shift `x` by whole domains so it lies on the same branch as `ref` (nearest image)."
unwrap_to(ref, x, L = DOMAIN_L) = x .- L .* round.((x .- ref) ./ L)

"Minimum-image A→B displacement. Valid while |true displacement| < L/2."
function displacement(xA, yA, xB, yB, L = DOMAIN_L)
    dx = xB .- xA
    dy = yB .- yA
    dx .-= L .* round.(dx ./ L)
    dy .-= L .* round.(dy ./ L)
    return dx, dy
end

# Catmull-Rom through p1,p2 with p0,p3 setting the tangents; t in [0,1].
@inline function _cr(p0, p1, p2, p3, t)
    return 0.5 * ((2p1) + (-p0 + p2) * t +
                  (2p0 - 5p1 + 4p2 - p3) * t^2 +
                  (-p0 + 3p1 - 3p2 + p3) * t^3)
end

# ---------------------------------------------------------------- fractional sampling
"""
    particles_at(file, keys, s; idx, order, L)

Particle positions at fractional frame index `s` (1-based, `1 ≤ s ≤ length(keys)`).
`order = :cubic` uses Catmull-Rom over four frames and falls back to `:linear` at the ends.
Returns positions re-wrapped into `[0, L)`.
"""
function particles_at(file, keys, s; idx = nothing, order = :cubic, L = DOMAIN_L)
    n = length(keys)
    (1.0 - 1e-9 <= s <= n + 1e-9) || error("s = $s outside [1, $n]")
    s = clamp(s, 1.0, float(n))
    i = clamp(floor(Int, s), 1, n - 1)
    t = s - i
    take = k -> begin
        x, y = _particles(file, keys[k])
        idx === nothing ? (x, y) : (x[idx], y[idx])
    end
    x1, y1 = take(i)
    x2, y2 = take(i + 1)
    x2 = unwrap_to(x1, x2, L); y2 = unwrap_to(y1, y2, L)

    if order === :linear || i == 1 || i + 2 > n
        xx = x1 .+ t .* (x2 .- x1)
        yy = y1 .+ t .* (y2 .- y1)
    else
        x0, y0 = take(i - 1)
        x3, y3 = take(i + 2)
        x0 = unwrap_to(x1, x0, L); y0 = unwrap_to(y1, y0, L)
        x3 = unwrap_to(x2, x3, L); y3 = unwrap_to(y2, y3, L)
        xx = _cr.(x0, x1, x2, x3, t)
        yy = _cr.(y0, y1, y2, y3, t)
    end
    return mod.(xx, L), mod.(yy, L)
end

"""
    field_at(file, keys, s)

Velocity field at fractional frame index `s`, linear in time. The fields are smooth on the
save interval, so linear is adequate; `selftest_fracframe.jl` measures the error.
"""
function field_at(file, keys, s)
    n = length(keys)
    s = clamp(s, 1.0, float(n))
    i = clamp(floor(Int, s), 1, n - 1)
    t = s - i
    u1, v1 = _field(file, keys[i])
    u2, v2 = _field(file, keys[i + 1])
    return (1 - t) .* u1 .+ t .* u2, (1 - t) .* v1 .+ t .* v2
end

function time_at(file, keys, s)
    n = length(keys)
    s = clamp(s, 1.0, float(n))
    i = clamp(floor(Int, s), 1, n - 1)
    t = s - i
    return (1 - t) * _time(file, keys[i]) + t * _time(file, keys[i + 1])
end

# ---------------------------------------------------------------- displacement statistics
"""
    disp_stats(dx, dy)

The summary the v2 dataset is specified by. `max` alone is a poor descriptor — it is a tail
statistic, and two configurations with the same max can differ several-fold in error.
"""
function disp_stats(dx, dy)
    m = sqrt.(dx .^ 2 .+ dy .^ 2)
    sm = sort(m)
    q = p -> sm[clamp(ceil(Int, p * length(sm)), 1, length(sm))]
    return (median = q(0.50), p90 = q(0.90), p99 = q(0.99), max = sm[end],
            frac_over_20 = mean(m .> 20), frac_over_22 = mean(m .> 22),
            frac_over_24 = mean(m .> 24))
end

_stat_value(st, which) = which === :median ? st.median :
                         which === :p90    ? st.p90    :
                         which === :p99    ? st.p99    :
                         which === :max    ? st.max    :
                         error("unknown statistic $which")

"""
    solve_frac_dp(file, keys, iA, target; stat, idx, tol, maxit, order, L)

Find the fractional frame gap `dp` such that `stat` of the A→B displacement equals `target` px.

Displacement grows very nearly linearly with `dp`, so a scaled fixed-point iteration converges in
a handful of steps. Returns `(dp, stats, iters, converged)`; inspect `converged` — a target beyond
what the remaining frames allow is clamped rather than silently mislabelled, which is precisely
the BUG-13 failure this replaces.
"""
function solve_frac_dp(file, keys, iA, target;
                       stat = :median, idx = nothing, tol = 0.02, maxit = 15,
                       order = :cubic, L = DOMAIN_L)
    n = length(keys)
    dp_max = float(n) - iA
    dp_max > 0 || error("frame A is the last frame; no room for B")
    xA, yA = particles_at(file, keys, float(iA); idx = idx, order = order, L = L)

    measure = dp -> begin
        xB, yB = particles_at(file, keys, iA + dp; idx = idx, order = order, L = L)
        dx, dy = displacement(xA, yA, xB, yB, L)
        disp_stats(dx, dy)
    end

    dp = clamp(0.5 * dp_max, 1e-3, dp_max)          # neutral start; growth is ~linear
    st = measure(dp)
    local converged = false
    iters = 0
    for k in 1:maxit
        iters = k
        val = _stat_value(st, stat)
        if abs(val - target) <= tol * target
            converged = true
            break
        end
        val <= 0 && break
        dp_new = clamp(dp * target / val, 1e-3, dp_max)
        if abs(dp_new - dp) < 1e-6 * dp_max         # clamped against the end of the run
            dp = dp_new
            st = measure(dp)
            break
        end
        dp = dp_new
        st = measure(dp)
    end
    return dp, st, iters, converged
end

# ---------------------------------------------------------------- validation
"""
    loo_error(file, keys, i; idx, L)

Leave-one-out check: reconstruct saved frame `i` from its neighbours *without* using it, and
compare against the truth. This is a conservative upper bound on the interpolation error, because
it spans a doubled interval — twice the gap actually used when sampling between adjacent frames.

Returns `(linear = ..., cubic = ...)` mean absolute position error in px.
"""
function loo_error(file, keys, i; idx = nothing, L = DOMAIN_L)
    n = length(keys)
    (2 <= i <= n - 1) || error("need an interior frame")
    take = k -> begin
        x, y = _particles(file, keys[k])
        idx === nothing ? (x, y) : (x[idx], y[idx])
    end
    xt, yt = take(i)
    xm, ym = take(i - 1)
    xp, yp = take(i + 1)
    xp_u = unwrap_to(xm, xp, L); yp_u = unwrap_to(ym, yp, L)
    xt_u = unwrap_to(xm, xt, L); yt_u = unwrap_to(ym, yt, L)

    xl = 0.5 .* (xm .+ xp_u); yl = 0.5 .* (ym .+ yp_u)
    lin = mean(sqrt.((xl .- xt_u) .^ 2 .+ (yl .- yt_u) .^ 2))

    cub = NaN
    if i - 2 >= 1 && i + 2 <= n
        xmm, ymm = take(i - 2)
        xpp, ypp = take(i + 2)
        xmm_u = unwrap_to(xm, xmm, L); ymm_u = unwrap_to(ym, ymm, L)
        xpp_u = unwrap_to(xp_u, xpp, L); ypp_u = unwrap_to(yp_u, ypp, L)
        xc = _cr.(xmm_u, xm, xp_u, xpp_u, 0.5)
        yc = _cr.(ymm_u, ym, yp_u, ypp_u, 0.5)
        cub = mean(sqrt.((xc .- xt_u) .^ 2 .+ (yc .- yt_u) .^ 2))
    end
    return (linear = lin, cubic = cub)
end

end # module
