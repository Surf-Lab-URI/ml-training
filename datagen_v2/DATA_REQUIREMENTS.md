# Training data v2 — what we need, and why

Working notes for the next data-generation run. Scripts will live in this directory.

Everything below is measured, not assumed. Where a number is uncertain or unverified it says so.

---

## 1. Why regenerate at all

The trained model is accurate on synthetic data and in the bulk of the lab frames, but its error
in the **top 2 mm** of the lab data is 2–3× higher than elsewhere. The diagnostic campaign
(`piv-models/notebooks/andy_test_plan.html`) traced this to a **displacement-range** problem, not a
wave-geometry problem:

- Sliding a real lab patch by a known amount, the model is exact to **22 px** (error ≤ 0.13 px),
  degrades at 24 px, and returns the **wrong sign** at 26 px. No waves involved.
- Error splits cleanly into two populations: pixels moving **less** than 22 px score 1.2–1.6 px in
  every configuration tested; pixels moving **more** score 12–32 px. The reported error is just the
  weighted average of the two.
- **22 px is where the training data ends.** The pix20 bin's largest displacement is 22.02 px, and
  the most extreme bin (pix30) has only 10.3% of its pixels beyond it.
- Applying that split blind to the lab reproduces the observed near-surface error to within 10%
  across frames spanning a 50× range of surface steepness (§3).

So the fix is to train on displacements the lab actually contains. That is the whole point of v2.

---

## 2. What the current training data contains

Measured from `ml-training/data/visual/*/*.jld2`, label = (uA+uB)/2, 20 pairs per bin:

| bin | median | p90 | max | % past 22 px |
|---|---|---|---|---|
| pix10 | 3.12 | 6.68 | 11.40 | 0.00% |
| pix20 | 8.43 | 13.83 | 22.02 | 0.00% |
| pix30 | 13.58 | 22.07 | 31.44 | 10.33% |

Two things to note:

- **The bin names are nominal.** `dp = floor(pix/smax)` with smax ≈ 5 px, so `pixN` fixes roughly the
  *maximum* displacement, not the typical one. "pix30" really means median 13.6 px.
- **There is no free surface anywhere in the training data.** It is 2-D periodic turbulence: no
  air/water interface, no masked air region, no surface-parallel boundary condition. The lab fails in
  the top 2 mm and the training data has no "top".

> Caveat: these files are the **2026-06-09** run. The checkpoint actually trained on
> `run_labapp_2026-07-06_20-28-55` (Unity), which applied lab appearance. The displacement statistics
> should carry over since the bin construction is unchanged, but **the appearance numbers here do not
> describe what the model saw** — the local files have background 0, the lab has ~45. Re-measure on
> Unity before relying on §5.

---

## 3. What the lab actually contains — the target

### 3.1 Displacement (the number that matters)

From Andy's hand-tracked points, restricted to the **top 2 mm**:

| pair | surface slope p99 | n | median | p90 | max | % past 22 px | model EPE |
|---|---|---|---|---|---|---|---|
| 80 | 0.009 (quiet) | 85 | 8.81 | 11.22 | 14.52 | 0.0% | 1.432 |
| 100 | 0.044 (quiet) | 77 | 12.00 | 18.25 | 23.90 | 3.9% | 1.988 |
| 123 | **0.480 (wavy)** | 36 | 13.48 | 20.53 | 24.09 | 5.6% | 2.627 |

**Target: real mass — not just a tail — out to ~45 px**, so the ceiling ends up comfortably past
the lab's maximum rather than sitting inside it. See R1 for the bin set that achieves this.

Note the lab's displacement is strongly depth-dependent: whole-frame median 3.3–4.7 px against
top-2 mm median 8.8–13.5 px. Training crops must span both regimes, which is why the bin set starts
at median 3 rather than higher.

> Two caveats. This is **36 points on one wave-active frame** — thin. And hand-tracked points are
> placed where particles are *trackable*, which may systematically avoid steep crest faces and glare.
> If so the true lab fraction past the ceiling is **higher** than 5.6%, which strengthens the case for
> a larger range rather than weakening it.

### 3.2 Surface geometry

| | slope p99 | \|Δη\| mean | \|Δη\| max |
|---|---|---|---|
| quiet frames (80, 100) | 0.009 – 0.044 | 1.1 – 1.5 px | 2.6 – 2.9 px |
| wave-active (122–126, 143–145) | **0.33 – 0.69** | 5.1 – 7.0 px | 18.8 – **34.6 px** |

- Dominant wave: phase speed 0.333–0.358 m/s → λ ≈ **7.1–8.2 cm** by the deep-water relation.
- Surface **spectrum** peaks near λ = **2.3 cm** — much shorter than the phase speed implies.
  The two reconcile as short **gravity-capillary** waves riding on a **0.090–0.101 m/s** wind drift
  (measured consistently across four pairs).
- Slope p99 of 0.48–0.69 **exceeds the Stokes limit** (ak = 0.443), so no single monochromatic wave
  can reproduce it. It needs a superposition of components.
- Δt **within** a pair is 10.0 ms. Consecutive pairs are 138.9 ms apart (7.2 Hz) — about 0.6 of a wave
  period, so neighbouring pairs are near-independent samples, not a time series.

### 3.3 Appearance

| property | lab | our synthetic test images | verdict |
|---|---|---|---|
| particle diameter (autocorrelation FWHM) | 1.98–2.07 px | 1.92 px | matches |
| particle density (4σ local maxima) | 57–61 /1000 px² | 42.5 /1000 px² | lab **1.4× denser** |
| background level | 41–46 | 50.7 | close |
| sensor noise σ | **not measurable** | 5.2 by construction | see below |

**Do not try to match a lab noise σ.** It cannot be measured from these files: the air above the
waterline is masked to *exactly* zero, so there is no particle-free region, and every in-water
estimator is contaminated by the 2 px particle texture. Validated against synthetic data where the
truth is known: three standard estimators read −40%, +33% and +71%. Density is likewise only
meaningful as a **ratio** (the estimator reads 42.5 against a nominal 28.6).

---

## 4. Requirements, ordered by how well the evidence supports them

### R1 — Displacement bins by MEDIAN: 3, 6, 9, 12, 16, 20, 26, 30 px *(strong evidence)*

**The ceiling is set by where the training mass runs out, not by the training maximum.** This is the
single most important design fact and it was not obvious. The current data reaches 31.4 px, yet the
model breaks at 22 px — because only **3.4%** of training pixels exceed 22 px, 0.5% exceed 26, and
**0% exceed 30**. A tail that merely touches a displacement does not teach the model to handle it.

Measured within-bin shape (pooled over 60 samples), where *M* is the bin median:

| fraction of a bin above | 1.0·M | 1.2·M | 1.4·M | 1.5·M | 1.6·M |
|---|---|---|---|---|---|
| % of pixels | 50% | 35% | 15% | 7% | 4% |

with **max/median = 1.67**, near-constant across a 5x range of displacement (1.71 -> 1.68 -> 1.66,
measured within the same sims at dp = 2, 4, 6), so it extrapolates safely.

> **Re-measured on the delivered v2 data, 2026-09-01** (`run_v2_2026-08-14_05-26-22`, 40 samples per
> bin, all eight bins). The near-constancy claim is **confirmed exactly** — the per-sample ratio is
> identical to two decimals in every bin (mean 1.89, sd 0.33, median 1.78, worst 3.03). That is
> structural rather than lucky: within one simulation every bin shares frame A and the same particle
> subset, so a bin's displacement field is the same velocity field scaled by its time gap, and
> scaling leaves max/median unchanged.
>
> **The value, however, is 1.78 (typical sample) to 1.89 (mean), not 1.67** — about 12% higher. The
> 1.67 was measured on the older v1 data. Two consequences for the numbers below:
>
> - The bin set does not span "max ~ 5 -> 50 px". A typical `med30` sample reaches **53 px**, the
>   average sample **57 px**, and the widest of 40 samples **91 px**; pooled, the bin's maximum is
>   **91.5 px**. Coverage at the high end is therefore *better* than this section claims.
> - That is not automatically good news. `med30` contains displacements roughly twice the intended
>   maximum, which makes it a harder learning problem than budgeted for — a candidate explanation
>   for why the wide-range v2 model underfit (train loss 0.502 vs 0.224, plateaued at epoch 33)
>   rather than simply needing more epochs. Untested; see the group report's §5.
>
> Reproduce: the per-bin median/p90/p99/max table on any run is printed by
> `ml-training/scripts/make_report.py --root <RUN_DIR>`.

Pooled coverage of the candidate bin sets, and the ceiling each implies under the
*mass-falls-below-3.4%* rule calibrated on the current dataset:

| threshold | current (3.6/8.9/14.3) | v2 medians 3-20 | **v2 medians 3-30** |
|---|---|---|---|
| 22 px | 1.9% | 10.3% | **23.6%** |
| 30 px | 0.2% | 1.4% | **12.1%** |
| 45 px | 0.0% | 0.1% | 1.2% |
| **implied ceiling** | **22 px** (observed) | ~28 px | **~41 px** |

Stopping at median 20 would put the ceiling at ~28 px — barely above the lab's observed 24 px max,
with no margin for the steep regions hand-tracking may be missing. **Going to median 30 puts it at
~41 px**, which is the headroom we want. Hence the eight bins above, spanning max ~ 5 -> 50 px.

> The "ceiling ~ where mass falls to 3.4%" rule is calibrated on a **single** data point — the
> current dataset. It is a plausible heuristic, not an established law. The pilot in section 5 is
> what tests it, and it should be run before committing to a full-scale generation run.

### R2 — Fix the displacement quantization, BUG-13 / BUG-14 *(prerequisite for R1)*

`ImageGen.jl:68` does `dp = max(1, Int(floor(pix / smax)))`, an integer count of saved frames, so
achievable displacement is quantized in steps of smax ≈ 5 px and anything below smax is unachievable
but still written under the requested label (**BUG-13**).

The root-cause fix is already designed as **BUG-14**: interpolate particle positions *between* saved
frames to synthesise a virtual frame at any t, then pick the exact Δt that hits the target
displacement. Without this we cannot hit a target distribution, only a coarse grid of maxima.
Requires interpolating the velocity field too, for the uA/vA/uB/vB labels.

### R3 — Match or exceed the lab's seeding *(strong evidence, cheap)*

The lab is ~1.4× denser than our current synthetic images. Separately, on a **wavy** surface halving
the seeding removes the surface-flattening benefit entirely (gain 0.91×) while doubling still improves
it (2.43×) — the curve has not saturated. Generate at the lab's density or above; do not under-seed.

### R4 — Include a free surface with wave geometry *(weaker evidence, but a real gap)*

The training data currently has no surface at all, and the failure is at the surface. Include:
- an air region masked to exactly zero, matching the lab convention;
- a range of surface slope p99 from ~0.01 to ~0.7, which needs a **superposition** of components
  (single waves cannot exceed 0.443);
- **gravity-capillary** dispersion ω² = gk + σk³/ρ — at λ = 1.6 cm, ignoring surface tension makes the
  phase speed 46% too slow;
- **Doppler shift by the drift current** — waves on a current propagate at c + U, and without it the
  synthetic surface pattern crawls while the lab's sweeps past.

> Honest caveat: geometry is **not** the main failure mode. With displacement controlled, steepness
> does not worsen the baseline error, and an earlier test showed that filling the masked air region
> changes the prediction by +0.03 px [−0.24, +0.35] — i.e. nothing. Treat R4 as coverage against a
> known blind spot, not as the fix.

### R5 — Log the statistics that actually predict performance *(free, do it)*

Per sample, record over the **top 2 mm band specifically**, not the whole frame:
median / p90 / p99 / max displacement; % past 20, 22 and 24 px (the ceiling is bounded, not exact);
surface slope p99; particle density; |Δη| between frames. This is what lets us filter and reason about
the dataset later — the current manifest records max displacement, which we now know is the wrong
summary statistic.

---

## 4b. Status — what is built and verified

In this directory:

| file | what it does |
|---|---|
| `FracFrame.jl` | fractional-frame sampling of a combined file — the BUG-14 fix |
| `ImageGenV2.jl` | the generator; drop-in sibling of `scripts/ImageGen.jl`, same CLI and renderer |
| `make_mock_combined.jl` | builds a mock combined file so the above can be run without Unity data |
| `selftest_fracframe.jl` | validates interpolation, periodic seam, and displacement targeting |

Verified locally, end to end on mock data:

- cubic interpolation error against exact analytic trajectories: **0.00005 px** (linear: 0.010 px);
- periodic seam handled — 28 of 4000 particles cross per interval, and a naive implementation
  reports a 511 px displacement where the truth is 5.4 px;
- all eight bins hit their target median within tolerance, in 2 solver iterations each;
- the **written label fields** reproduce the claimed medians to within 0.13 px (the residual is the
  expected Lagrangian-vs-Eulerian-midpoint difference, not an error);
- out-of-tolerance samples are **skipped with a warning** rather than written under a label they do
  not have — the BUG-13 failure mode is now impossible by construction.

### Known gap

`field_at` interpolates the velocity field **linearly in time**, and the mock field is *steady*, so
that path is exact by construction in the self-test and therefore **untested**. On real turbulence the
field evolves between saves and linear interpolation will carry some error into the `uB/vB` labels.
**Measure this on a real combined file before the production run** — it is the one place a silent
label error could still enter. A leave-one-out check on the fields, mirroring `loo_error` for
particles, is the natural test.

### Consequence: frame A moves earlier

The generator anchors A so the largest target still fits before the final frame. Wider targets push
that anchor back. On a typical 40-frame run, with a median displacement of 2.26 px per saved frame:

| target median | dp needed | anchor frame A |
|---|---|---|
| 13.6 px (v1 pix30) | 6.0 | 34 |
| 20 px | 8.8 | 31 |
| 30 px | 13.3 | **27** |

The runbook notes that later frames are *more developed* turbulence — the initial condition is
artificially smooth and the enstrophy cascade fills small scales over time. So the widest bins sample
a slightly less developed field. Either lengthen the runs (`nt`) or accept it; worth deciding
deliberately rather than discovering later.

---

## 5. Open questions to settle before the run

1. **Re-measure the labapp run on Unity.** §2's appearance numbers come from the older local files.
   Confirm background, noise and density for `run_labapp_2026-07-06_20-28-55`.
2. **Does the ceiling move with training range?** The premise of the whole exercise. Worth a small
   pilot — one bin at ~40 px — retrained briefly, then re-run the pure-translation test to see where
   the new ceiling sits, before committing to a full-scale run.
3. **Free surface or not?** Adding one is a substantial change to `2DTurbulence.jl`. Given R4's weak
   evidence, decide whether v2 is "same physics, wider displacement range" (cheap, addresses the
   demonstrated cause) or "add a free surface" (expensive, addresses a suspected blind spot).
4. **How much data?** The current run is one pair per sim. Extending the range means either more sims
   or more pairs per sim; the latter is cheaper but correlates the samples.

---

## 6. What success looks like

The pure-translation test on the retrained model should stay exact past **45 px**, and the top-2 mm
error on lab pairs 100 and 123 should drop toward the ~1.4 px that below-ceiling pixels achieve today.

**We cannot currently verify that.** Only 36 hand-tracked points measure the top 2 mm on a wave-active
frame, and 172 of our 208 near-surface points come from near-flat frames. More hand-tracked ground
truth on steep frames is on the critical path for v2, not optional.
