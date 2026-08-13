#=
Self-test for FracFrame — runs with no simulation data.

The real `*_combined.jld2` files live on Unity, so this builds a mock one with *analytically known*
particle trajectories and checks the three things that could silently corrupt a dataset:

  1. interpolation accuracy against exact truth,
  2. periodic seam handling (a naive implementation fails here by ~L per crossing particle),
  3. that the Δt solver actually hits a requested displacement — the BUG-13 failure.

    julia --project=. datagen_v2/selftest_fracframe.jl
=#
using JLD2, Statistics, Printf, Random

include(joinpath(@__DIR__, "FracFrame.jl"))
using .FracFrame

const L = FracFrame.DOMAIN_L
const NF = 24                    # frames in the mock run
const DT = 0.5                   # save interval
const NP = 4000                  # particles

# A steady, smooth, analytic velocity field: two counter-rotating cells plus a uniform drift.
# Steady in time, so a particle's exact position is available by integrating accurately.
kx = 2π / L
vel(x, y) = ( 6.0 * sin(kx * y) + 3.0,      # u
             -6.0 * sin(kx * x)      )      # v

function exact_position(x0, y0, t; substeps = 4000)
    x, y = x0, y0
    h = t / substeps
    for _ in 1:substeps                      # RK4, effectively exact at this step size
        u1, v1 = vel(x, y)
        u2, v2 = vel(x + 0.5h * u1, y + 0.5h * v1)
        u3, v3 = vel(x + 0.5h * u2, y + 0.5h * v2)
        u4, v4 = vel(x + h * u3,    y + h * v3)
        x += h * (u1 + 2u2 + 2u3 + u4) / 6
        y += h * (v1 + 2v2 + 2v3 + v4) / 6
    end
    return mod(x, L), mod(y, L)
end

# ---------------------------------------------------------------- build the mock combined file
mktempdir() do dir
    path = joinpath(dir, "mock_combined.jld2")
    rng = MersenneTwister(0)
    x0 = rand(rng, NP) .* L
    y0 = rand(rng, NP) .* L

    jldopen(path, "w") do f
        for i in 1:NF
            t = (i - 1) * DT
            xs = similar(x0); ys = similar(y0)
            for p in 1:NP
                xs[p], ys[p] = exact_position(x0[p], y0[p], t)
            end
            f["particles/timeseries/particles/$i"] = (x = xs, y = ys)
            f["fields/timeseries/t/$i"] = t
            gx = range(0, L, length = 65)
            f["fields/timeseries/u/$i"] = [vel(x, y)[1] for x in gx, y in gx]
            f["fields/timeseries/v/$i"] = [vel(x, y)[2] for x in gx, y in gx]
        end
    end

    jldopen(path, "r") do f
        keys = FracFrame.frame_keys_of(f)
        @printf("mock run: %d frames, %d particles, save interval %.2f\n\n", length(keys), NP, DT)

        # ---- 1. interpolation accuracy at fractional frames, against exact truth
        println("1. interpolation error vs exact trajectories (px)")
        @printf("   %-8s %12s %12s\n", "s", "linear", "cubic")
        worst_cubic = 0.0
        for s in (6.25, 6.5, 6.75, 11.5, 17.5)
            xt = similar(x0); yt = similar(y0)
            t = (s - 1) * DT
            for p in 1:NP
                xt[p], yt[p] = exact_position(x0[p], y0[p], t)
            end
            errs = Float64[]
            for order in (:linear, :cubic)
                xi, yi = particles_at(f, keys, s; order = order)
                dx, dy = FracFrame.displacement(xt, yt, xi, yi, L)
                push!(errs, mean(sqrt.(dx .^ 2 .+ dy .^ 2)))
            end
            worst_cubic = max(worst_cubic, errs[2])
            @printf("   %-8.2f %12.5f %12.5f\n", s, errs[1], errs[2])
        end

        # ---- 2. periodic seam: how many particles cross, and what a naive version would do
        println("\n2. periodic seam handling")
        xa, ya = FracFrame._particles(f, keys[10])
        xb, yb = FracFrame._particles(f, keys[11])
        naive = sqrt.((xb .- xa) .^ 2 .+ (yb .- ya) .^ 2)
        dx, dy = FracFrame.displacement(xa, ya, xb, yb, L)
        proper = sqrt.(dx .^ 2 .+ dy .^ 2)
        ncross = count(naive .> L / 2)
        @printf("   particles crossing the seam in one interval : %d of %d\n", ncross, NP)
        @printf("   naive  max |displacement|                   : %8.2f px\n", maximum(naive))
        @printf("   minimum-image max |displacement|            : %8.2f px\n", maximum(proper))

        # ---- 3. the BUG-13 fix: can we hit a requested displacement?
        println("\n3. requested vs achieved displacement (BUG-13)")
        u0, v0 = FracFrame._field(f, keys[1])
        smax = maximum(sqrt.(u0 .^ 2 .+ v0 .^ 2)) * DT
        @printf("   smax = %.2f px per saved frame -> old code could only reach multiples of this\n\n",
                smax)
        @printf("   %-9s %-10s %10s %10s %9s %8s\n",
                "target", "statistic", "achieved", "old (int dp)", "dp used", "iters")
        iA = 2
        for (target, stat) in ((3.0, :median), (8.0, :median), (14.0, :median),
                               (20.0, :median), (30.0, :max), (45.0, :max))
            dp, st, iters, conv = solve_frac_dp(f, keys, iA, target; stat = stat)
            got = stat === :median ? st.median : st.max
            dp_old = max(1, Int(floor(target / smax)))
            xB, yB = particles_at(f, keys, iA + dp_old)
            xA, yA = particles_at(f, keys, float(iA))
            dxo, dyo = FracFrame.displacement(xA, yA, xB, yB, L)
            so = FracFrame.disp_stats(dxo, dyo)
            old = stat === :median ? so.median : so.max
            flag = conv ? "" : "  (clamped)"
            @printf("   %-9.1f %-10s %10.2f %10.2f %9.3f %8d%s\n",
                    target, string(stat), got, old, dp, iters, flag)
        end

        # ---- 4. leave-one-out, the check that transfers to real data
        println("\n4. leave-one-out reconstruction of a saved frame (conservative bound)")
        e = loo_error(f, keys, 12)
        @printf("   linear %.5f px   cubic %.5f px   -> cubic is %.1fx better\n",
                e.linear, e.cubic, e.linear / e.cubic)

        println()
        ok = worst_cubic < 0.05
        println(ok ? "PASS — cubic interpolation stays well under 0.05 px" :
                     "FAIL — interpolation error too large: $worst_cubic px")
    end
end
