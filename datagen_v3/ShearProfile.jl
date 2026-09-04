"""
ShearProfile — impose the tank's measured near-surface shear on an otherwise homogeneous flow.

WHY THIS EXISTS
---------------
Measured over 60 frames of the traditional-PIV collection, the tank's median displacement falls
from 8.06 px at 0-1 mm below the surface to 0.38 px at 24-40 mm: a **21x decay over about 8 mm**.
Our simulation is 2-D periodic turbulence with no surface, so its displacement is statistically
uniform with depth. That structural mismatch, not the displacement magnitude, is the largest known
difference between the training data and the tank -- see datagen_v2/DATA_REQUIREMENTS.md R6.

WHAT IT DOES
------------
Multiplies the velocity field by a depth-dependent factor s(z) fitted to the measured profile, and
places a wavy free surface with an air region above it. Particles are advected by the scaled field,
so images and labels stay consistent.

WHAT IT IS NOT
--------------
The scaled field is **not divergence-free** and is not a solution of the free-surface equations. It
is a kinematic imposition that reproduces the measured profile. Two consequences, both real:

  * `lambda_div` must be reduced or set to 0 when training on this data -- the divergence penalty
    would otherwise fight the imposed shear.
  * Do not present this as physical free-surface turbulence. It is a targeted distribution match.

A proper free-surface simulation is the honest long-term fix; this is the version that can be built
by re-rendering existing runs rather than re-simulating everything.
"""
module ShearProfile

export shear_factor, surface_row, apply_shear

using Random

"""
    shear_factor(depth_mm; surf=8.06, deep=0.38, decay=8.0)

Multiplier reproducing the measured profile:  amp*exp(-z/decay) + floor_, normalised by `floor_`
so the DEEP water keeps the simulation's own speed -- the surface is sped up rather than the bulk
slowed down, which preserves the small-scale structure the network learns from.

Defaults are a least-squares fit **in log space** (so every depth decade weighs equally, not just
the fast bins) to the 60-frame measurement in DATA_REQUIREMENTS R6:

    0-1mm 8.06 | 2-3 6.98 | 4-6 4.09 | 8-12 1.53 | 16-24 0.53 | 24-40 0.38

Worst relative error across the ten depth bins: 16%. Surface/deep ratio 24.6x against a measured
21x. A linear-space fit was tried first and was much worse in the tail (62% error, and it put the
24-40 mm bin at 0.14 against a measured 0.38).
"""
function shear_factor(depth_mm::Real; amp::Real=9.940, decay::Real=4.882, floor_::Real=0.365)
    d = max(depth_mm, 0.0)
    return (amp * exp(-d / decay) + floor_) / floor_
end

"""
    surface_row(x, W; amp, lam, phase, comps)

Free-surface elevation in pixels for each column. A SUPERPOSITION of components, because the
measured slope p99 reaches 0.81 and a single monochromatic wave cannot exceed the Stokes limit
ak = 0.443 (DATA_REQUIREMENTS 3.2).

The three defaults were fitted to hit the measured slope statistics: they give median 0.060 and
p99 0.260 against a lab median of 0.06 and p99 of 0.26, with 9.8 px peak-to-peak. The first
hand-picked set was four times too steep (median 0.364, p99 1.104) -- the self-test caught it.
"""
function surface_row(x::AbstractVector, W::Real; comps=[(2.44, 254.2, 0.0), (1.94, 95.7, 1.1),
                                                       (0.75, 60.4, 2.3)])
    eta = zeros(length(x))
    for (a, lam, ph) in comps
        eta .+= a .* sin.(2π .* x ./ lam .+ ph)
    end
    return eta
end

"""
    apply_shear(u, v, x, y, surf_rows, dx_mm; kw...)

Scale the velocity field by depth and return (u, v, water_mask). `surf_rows` is the surface row per
column; rows above it are air. Depth is measured downward in image rows, converted to mm by
`dx_mm`.
"""
function apply_shear(u::AbstractMatrix, v::AbstractMatrix, surf_rows::AbstractVector,
                     dx_mm::Real; kw...)
    H, W = size(u)
    us = similar(u); vs = similar(v); water = falses(H, W)
    @inbounds for j in 1:W
        sj = surf_rows[min(j, length(surf_rows))]
        for i in 1:H
            depth_mm = (i - sj) * dx_mm
            if depth_mm < 0
                us[i, j] = 0; vs[i, j] = 0            # air: masked to exactly zero, as the lab is
            else
                f = shear_factor(depth_mm; kw...)
                us[i, j] = u[i, j] * f; vs[i, j] = v[i, j] * f
                water[i, j] = true
            end
        end
    end
    return us, vs, water
end

end # module
