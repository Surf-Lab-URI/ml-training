#=
Build a mock *_combined.jld2 so ImageGenV2 can be exercised end-to-end without Unity data.

Analytic, smooth, steady velocity field with tracked particles, written in the same layout the real
simulation produces. Enough frames that the largest median target (30 px) is actually reachable.

    julia --project=. datagen_v2/make_mock_combined.jl [outdir]
=#
using JLD2, Random, Printf

const L  = 512.0
const NF = 60          # frames — must be enough for the biggest target
const DT = 0.5
const NP = 16384

kx = 2π / L
vel(x, y) = (6.0 * sin(kx * y) + 3.0, -6.0 * sin(kx * x))

function step_rk4(x, y, h)
    u1, v1 = vel(x, y)
    u2, v2 = vel(x + 0.5h * u1, y + 0.5h * v1)
    u3, v3 = vel(x + 0.5h * u2, y + 0.5h * v2)
    u4, v4 = vel(x + h * u3,    y + h * v3)
    return x + h * (u1 + 2u2 + 2u3 + u4) / 6, y + h * (v1 + 2v2 + 2v3 + v4) / 6
end

outdir = length(ARGS) >= 1 ? ARGS[1] : mktempdir()
mkpath(outdir)
path = joinpath(outdir, "mock_2DT-A000-nmax21-mjet2_combined.jld2")

rng = MersenneTwister(3)
x = rand(rng, NP) .* L
y = rand(rng, NP) .* L
gx = range(0, L, length = 512)

jldopen(path, "w") do f
    for i in 1:NF
        t = (i - 1) * DT
        f["particles/timeseries/particles/$i"] = (x = mod.(x, L), y = mod.(y, L))
        f["fields/timeseries/t/$i"] = t
        f["fields/timeseries/u/$i"] = [vel(px, py)[1] for px in gx, py in gx]
        f["fields/timeseries/v/$i"] = [vel(px, py)[2] for px in gx, py in gx]
        if i < NF                                   # advance to the next save
            sub = 40
            h = DT / sub
            for _ in 1:sub, p in 1:NP
                x[p], y[p] = step_rk4(x[p], y[p], h)
            end
        end
    end
end

@printf("wrote %s\n  %d frames, %d particles, save interval %.2f\n", path, NF, NP, DT)
