# Does the imposed profile actually reproduce the measured tank profile?
include(joinpath(@__DIR__, "ShearProfile.jl"))
using .ShearProfile, Printf, Statistics, Random

DX_MM = 0.0565454946380008
# measured, 60 frames of XCPIV/ExpLCL_1_03-200 (DATA_REQUIREMENTS R6)
MEASURED = [(0,1,8.06),(1,2,7.71),(2,3,6.98),(3,4,5.84),(4,6,4.09),
            (6,8,2.56),(8,12,1.53),(12,16,0.96),(16,24,0.53),(24,40,0.38)]

println("1. shear_factor reproduces the measured profile")
@printf("   %-12s %10s %10s %8s\n", "depth (mm)", "measured", "modelled", "ratio")
base = 0.38                                   # the deep-water value the factor is normalised to
errs = Float64[]
for (lo, hi, meas) in MEASURED
    mid = (lo + hi) / 2
    modelled = base * shear_factor(mid)
    r = modelled / meas; push!(errs, abs(r - 1))
    @printf("   %-12s %10.2f %10.2f %8.2f\n", "$lo-$hi", meas, modelled, r)
end
@printf("   worst relative error: %.0f%%\n\n", 100maximum(errs))

println("2. surface geometry matches the measured slope statistics")
W = 512; x = collect(0.0:W-1)
eta = surface_row(x, W)
slope = abs.(diff(eta))
@printf("   amplitude    %.1f px peak-to-peak\n", maximum(eta) - minimum(eta))
@printf("   slope median %.3f   (lab 0.06)\n", median(slope))
@printf("   slope p99    %.3f   (lab 0.26, max 0.81)\n", quantile(slope, 0.99))

println("\n3. apply_shear masks air and scales water")
H = 512
u = ones(H, W); v = zeros(H, W)
surf = 0.18 * H .+ eta                        # 18% air fraction, measured
us, vs, water = apply_shear(u, v, surf, DX_MM)
@printf("   air fraction  %.1f%%   (lab 18.1%%)\n", 100 * (1 - mean(water)))
@printf("   air velocity  max |u| = %.3f  (must be 0)\n", maximum(abs.(us[.!water])))
sd(lo, hi) = begin
    vals = Float64[]
    for j in 1:W, i in 1:H
        d = (i - surf[j]) * DX_MM
        (d >= lo && d < hi && water[i, j]) && push!(vals, us[i, j])
    end
    isempty(vals) ? NaN : median(vals)
end
@printf("   scaled field: 0-1mm %.2f  vs  24-40mm %.2f   ratio %.0fx  (lab 21x)\n",
        sd(0,1), sd(24,40), sd(0,1)/sd(24,40))
