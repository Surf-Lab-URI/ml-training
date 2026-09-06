"""
ShearProfile — impose a near-surface shear layer on an otherwise homogeneous flow.

WHY THIS EXISTS
---------------
Measured over 60 frames of the traditional-PIV collection, the tank's median displacement falls
from 8.06 px at 0-1 mm below the surface to 0.38 px at 24-40 mm: a **21x decay over about 8 mm**.
Our simulation is 2-D periodic turbulence with no surface, so its displacement is statistically
uniform with depth. That structural mismatch is a large known difference between the training data
and the tank -- see datagen_v2/DATA_REQUIREMENTS.md R6.

WHAT THE FIRST ATTEMPT GOT WRONG, AND WHY THE DEFAULT RATIO IS NOT 21
--------------------------------------------------------------------
The first version reproduced the 21x ratio faithfully and produced **unlearnable data**. With the
top-2 mm median pinned at the bin target, a 21x decay puts the deep water at target/21 -- 0.43 px
for a 9 px bin. Measured on the delivered surf02 samples: median |d| 0.265 px, with **83% of the
water below 1 px and 39% below 0.2 px**. A 2 px particle cannot resolve that, and correlation-based
methods need roughly 1 px to lock on.

Five training arms confirmed it, all landing on the same score (val 2.650-2.651, i.e. exactly the
mean target magnitude -- the model predicting zero): normalisation off, 4x learning rate, a lower
norm_floor, fast bins only, and a loader fix removing row flips and biasing crops to the surface.
None escaped, because the problem was never a hyperparameter. Predicting zero is close to CORRECT
on data that is mostly sub-pixel.

So there is a genuine tension and it cannot be tuned away:

  * reproducing the tank's ratio faithfully makes most of the frame unresolvable;
  * raising the overall scale to compensate pushes the surface past the tank's own maximum, which
    is the mistake v2 made (med30 reached ~91 px against a tank maximum of 24).

The resolution is to keep the STRUCTURE -- a fast thin layer over slower bulk, which is what the
model has never seen -- while compressing the RATIO so both ends stay measurable. At ratio 5 and a
9 px surface target the deep water sits at 1.8 px, and a 20 px target keeps the surface inside the
tank's observed top-2 mm range of 0.97-22.03. That is a deliberate, documented compromise: v3 data
teaches that free-surface flows have a sheared skin, not that the skin is exactly 21x.

WHAT IT IS NOT
--------------
The scaled field is **not divergence-free** and is not a solution of the free-surface equations.
`lambda_div` must be 0 when training on it. A proper free-surface simulation is the honest
long-term fix; this is the version that can be built by re-rendering existing runs.
"""
module ShearProfile

export shear_factor, surface_row, apply_shear, RATIO, DECAY_MM

# Surface/deep displacement ratio. The tank measures 21x; we use 5x deliberately -- see the module
# docstring. Override per run with params.toml [bins.v3].shear_ratio.
const RATIO = 5.0
# e-folding depth in mm: the log-space fit to the measured profile, kept unchanged.
const DECAY_MM = 4.882

using Random

"""
    shear_factor(depth_mm; ratio, decay)

Multiplier in [1, ratio]: `ratio` at the interface, falling exponentially with e-folding depth
`decay` mm toward 1 in the deep water. Normalised so the DEEP water keeps the simulation's own
speed -- the surface is sped up rather than the bulk slowed down, which preserves the small-scale
structure the network learns from AND keeps the bulk measurable.

`ratio` defaults to 5, not the measured 21. See the module docstring: 21 puts the deep water below
the resolution limit and produced five straight training collapses. `decay` of 4.88 mm is the
log-space fit to the measured profile and is kept.
"""
function shear_factor(depth_mm::Real; ratio::Real=RATIO, decay::Real=DECAY_MM)
    d = max(depth_mm, 0.0)
    return 1.0 + (ratio - 1.0) * exp(-d / decay)
end

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
