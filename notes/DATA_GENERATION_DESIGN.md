# Data-Generation Design Notes

> Synthetic PIV training-data pipeline (2D turbulence → image pairs).
> Captures the design discussion with the collaborating Andy and Colton (June 2026) so it
> can be presented and revisited. Each section answers one open question.
>
> **Status:** discussion / proposal. Nothing here is implemented yet except where
> noted as "current code".

---

## Current code, for reference (the baseline we are designing away from)

So the proposals below are concrete, here is what `2DTurbulence.jl` actually does today
(grid units = pixels, since the 512×512 grid has extent 512×512):

- **Grid:** 512 × 512, doubly periodic (`topology=(Periodic, Periodic, Flat)`), `dx = dy = 1 px`.
- **Physics:** `NonhydrostaticModel`, `WENO(order=5)` advection, `ScalarDiffusivity(ν = 1e-5)`
  — effectively inviscid over the short sampling window.
- **Initial streamfunction** (random-phase modes + one large-scale jet):

  ```
  ψ(x,y) = A · cos( l(round(m_jet·sin φⱼ))·y + k(round(m_jet·cos φⱼ))·x − φ[1,2] )      ← jet
         + Σ_{m=1}^{m_max} Σ_{n=1}^{n_max} a[m,n] · cos( k(n−c)·x + l(m−c)·y − φ[m,n] )   ← random field

  k(n) = 2π(n−1)/N,   l(m) = 2π(m−1)/M,   c = floor(n_max/2 + 1)
  a[m,n] = rand() · (21² / n_max²)          # smaller n_max ⇒ larger per-mode amplitude
  φ[m,n], φⱼ ~ Uniform(0, 2π)
  A = jet_amp · (1.5 − rand())              # so A ∈ [0.5·jet_amp, 1.5·jet_amp] = [150, 450] at default
  ```

  Velocities: `u = ∂ψ/∂y`, `v = −∂ψ/∂x`.
- **Current defaults** (`src/args.jl`): `jet_amp = 300.0`, `n_max = m_max = 21`, `m_jet = 2`,
  `seed = 1234`, `nt = 20`.
- **Particles — TWO separate knobs** (see §8):
  - *Simulation tracer pool:* `Nparticles = M·N/16 = 16384`, hard-coded in `2DTurbulence.jl`
    (the particles physically advected by the flow).
  - *Rendered per image:* `--sample`/`-k`, **default 5000**, randomly subsampled from the pool in
    `ImageGenFunc.jl` (`randperm(rng,n)[1:min(k,n)]`). **This is the PIV image density: 5000/512² =
    0.019 ppp.** The pool caps it: you can never render more than 16384 per image.
- **Six random draws per sim** seeded by one master `--seed`: jet amp `A`, mode amplitudes `a`,
  mode phases `φ`, jet phase `φⱼ`, particle `x₀`, particle `y₀`.

The six random draws are why **one seed = one reproducible flow** — this underpins §2 and §4.

---

## 1. Batch driver: "run N simulations" from the command line

**Goal:** `... --n_sims 10000` runs 10 000 independent simulations, each producing one sample.

**Proposal:** add an outer loop driven by a new arg (e.g. `--n_sims`), where each iteration
**re-randomizes via a distinct seed** so the flows are independent:

```
for i in 1:n_sims
    seed_i = base_seed + i          # deterministic, reproducible, but distinct per sim
    run one simulation with seed_i  → produce one sample + metadata
end
```

- Keep `--seed` as the **base seed** so the whole 10 000-run campaign is reproducible from a
  single number (`seed_i = base_seed + i`), yet every sim is a different flow.
- This is **embarrassingly parallel**: on Unity/SLURM, run it as an array job (one task per sim,
  `seed_i` from `SLURM_ARRAY_TASK_ID`) rather than a serial Julia loop. That also sidesteps any
  memory accumulation across runs.
- Caveat to verify: `base_seed + i` gives well-separated RNG streams for Julia's default RNG; if
  we ever see correlation between consecutive sims, switch to hashing the index
  (`seed_i = hash((base_seed, i))`) or to independent `Xoshiro`/`MersenneTwister` substreams.

---

## 2. One sample (image pair) per simulation

**Decision: one image pair per simulation.** This is the right call and our own measurements
back it up.

**Why one, not many:** the velocity-field **decorrelation time** of these sims is
**τ ≈ 110 saved frames ≈ 30 time units** (measured; autocorrelation of `[u;v]` stays > 0.99 for
the first ~5 time units — the regime image pairs are taken in). So extra frames from the *same*
sim are **near-duplicate flows**, not new labels. They give only mild particle-position
augmentation, not label diversity. Therefore:

> **One simulation ≈ one independent flow label.** All diversity must come from running **many
> sims** (§1) with varied seeds (§2's six random draws) and varied IC parameters (§7), not from
> many frames per sim.

This is also what makes §4's "don't store full sims" strategy viable: if a sim only yields one
useful sample, there is no reason to keep all its frames.

**Implementation note — one pair vs. `nt` (discussed 2026-06-08).** Today `ImageGen.jl` loops
`p = 1 … (nframes − dp)` and emits a pair starting at *every* frame, so each sim yields ~12
pairs/bin (the pilot: pix10/20/30 ≈ 14.7/12.7/10.7 pairs each). To get **one pair per sim**, emit
just the single pair `[A, A+dp]`. That mainly cuts **storage + correlated redundancy** (~12×), and
is what one-sim-one-flow calls for. It does *not* by itself shrink `nt`:

- Saved frames `= nt + 1`; a pair needs frames `dp` apart, `dp = max(1, floor(pix/smax))`. The
  binding bin is the largest displacement (pix30, biggest `dp`). So `nt ≥ dp_max`.
- If we keep pairing **from frame 1** (the raw IC), `nt ≈ 10` covers pix30 even for slow sims.
- But frame 1 is the *undeveloped* IC — §3 says sample from a **developed** flow at `t = C/(U·k_p)`.
  Then `nt = (frames to reach t_sample) + dp`, possibly *larger*. So `nt` is governed by the
  sampling time, not by the one-pair choice.
- Cleanest endgame (frozen-field warp, deferred BUG-14): one developed frame warped to any
  displacement → need essentially **one saved field**; `nt` only has to reach the developed state.

**Decision order:** choose the sampling time `C` (§3) → that fixes which frame is `A` → then `nt`.

---

## 3. *When* to sample — the eddy-turnover-time criterion (Andy's idea)

**The question:** after starting from the random IC, how long do we evolve before grabbing the
pair? Too early → the field is still the raw random IC (unphysical, no developed structures);
too late → coherent vortices have merged and the flow is no longer representative.

**Andy's proposal (sound, and we should adopt it):** on purely dimensional grounds, in an
(effectively) inviscid system the only timescale is set by a velocity and a wavenumber:

```
                    ┌ ∫ k² E(k) dk ┐ ½
sample at  t = C / (U · k_p) ,   k_p = │ ──────────── │      U = max speed at the IC
                    └  ∫ E(k) dk   ┘
```

- `k_p` is the **energy-weighted centroid wavenumber** (RMS wavenumber of the energy spectrum).
  Note `∫k²E dk ∝ enstrophy` and `∫E dk = energy`, so `k_p = √(enstrophy / energy)` — a clean,
  parameter-free measure of "the dominant scale of the flow."
- `1/(U·k_p)` is the **eddy-turnover time** — the natural clock of the turbulence.
- `C` is a dimensionless O(1) constant we **calibrate experimentally**: sweep `C` over a range
  (e.g. 0.5, 1, 2, 4, 8) and inspect the resulting pairs / spectra to find where the flow looks
  "developed but not yet collapsed." Then fix a `C` (or a small random range of `C`) for the campaign.

**Why this is better than a fixed `t_end`:** different ICs (different `jet_amp`, `n_max`) have
different speeds and dominant scales, so a fixed wall-clock time samples them at *different*
physical stages. Normalizing by `1/(U·k_p)` samples every sim at the **same dynamical age**,
adaptively. This directly avoids "too short / too long."

**Connection to prior measurement:** our decorrelation result (τ ≈ 30 time units) is the
empirical version of this clock; the `C/(U·k_p)` form makes it *per-sim adaptive* instead of a
single global number.

**Alternative `k_p` worth noting:** some decaying-2D-turbulence papers define the eddy-turnover
time from the **RMS vorticity**, `t* = 1/ω_rms`. Equivalent in spirit (ω_rms ~ U·k_p); we can
report both in the metadata and pick whichever is more stable in practice.

**To implement:** compute `E(k)` (2D FFT of velocity → shell-average), then `k_p`, `U = max|speed|`,
at the IC; set `t_end = C/(U·k_p)`; run; sample.

> **CALIBRATION (2026-06-08/09, `scripts/measure_kp.py` + spectrum analysis).** Two earlier wrong
> turns, then the correct answer — recorded so the reasoning is traceable:
>
> - ❌ *First read:* `k_p` flat to <0.1% over ~5 τ₀ and vorticity kurtosis flat & sub-Gaussian (1.5–2.7)
>   → I concluded "flow is stationary / jet-dominated / not turbulent → run short, nt≈10." **WRONG —
>   wrong metrics.** `k_p` is energy-weighted and kurtosis is dominated by the energy-containing
>   large scales; both are blind to the small-scale cascade.
> - ✅ *Correct read (energy spectrum `E(k)`, the right diagnostic):* the IC is **spectrally
>   truncated** — energy only up to k≈16 (the prescribed modes), a sharp cliff to zero beyond, **no
>   small scales** (an artificial start). Over the run the **2D enstrophy cascade fills the spectrum
>   out to k≈400** — i.e. **turbulence DOES develop.** Energy stays large-scale (97% at k≤2, normal
>   for 2D), while *enstrophy* cascades down — which is why the energy-weighted metrics looked flat.
> - **Run length DOES matter and is the cascade-development time.** Small-scale energy (k>16) grows
>   ~0 → ~0.0024 over ~21 time units and is only *beginning* to saturate at the end. The nt=15 pilot
>   (~8 units) sampled the flow **under-developed** (~half the eventual small-scale content).
> - **Andy's eddy-turnover criterion is vindicated:** sample only after the cascade has developed,
>   not at t≈0. Correct development metric = small-scale spectral fill-in (saturation of energy in
>   k>16), NOT `k_p`/kurtosis. A long run (nt=100, ~50 units) is running to find the saturation time
>   → that sets the sampling time / `nt` for the pilot (expected **nt ≳ 40**, well above 15).
> - Corollary still holds: `dt = 5/U_max` ⇒ `smax ≈ 5 px` by construction ⇒ `dp = 2,4,6` for 10/20/30.

**How long does a run take *today* (for reference).** The save interval is CFL-derived:
`dt = 10·tcfl`, `tcfl = 0.5·Δx/U_max`, and `Δx = 1` (512 grid over 512 extent), so

$$\text{dt} = 5/U_{\max}, \qquad \text{simulated time} = nt \cdot \text{dt} = 5\,nt/U_{\max}.$$

For `nt = 15`, `U_max ≈ 9`: **≈ 8 simulated time units, ~150 solver steps, ~43 s wall** on the Mac
CPU (≈130 s/run including Julia startup; the 100-sim pilot ran ~2.2 min/sim). It **auto-scales with
speed** (faster flow → smaller `dt`). Crucially this length is set *arbitrarily by `nt`*, not by
physics — replacing it with the `t = C/(U·k_p)` criterion above is the whole point of §3.

---

## 4. Storage: keep the samples, regenerate the rest

**Recommendation: do NOT store full simulations. Store only (a) the final sample and (b) enough
metadata to regenerate the sim bit-for-bit.**

**Rough storage budget** (512×512, per sim):

| What | Size (Float64) | Size (Float32 / compressed) |
|---|---|---|
| One velocity field component (512²) | 2.0 MB | 1.0 MB |
| Label field `uA,vA,uB,vB` (4 comps) | 8.0 MB | ~4 MB |
| Two particle images (512²) | 4.0 MB | ~0.2–0.5 MB (PNG/uint8) |
| **One stored sample** | **~12 MB** | **~2–5 MB** |
| *Full sim output today* (≈16 frames of fields + particles) | **~100+ MB** | — |

- **Store full sims:** 10 000 × ~100 MB ≈ **1 TB+**. Wasteful, since §2 says only one frame is useful.
- **Store samples only:** 10 000 × ~3–12 MB ≈ **30–120 GB**. Tractable. With Float32 labels +
  uint8 images, comfortably toward the low end.
- **Regeneration cost:** one nt=15 sim ≈ **~50 s single-core** (measured on the Mac smoke test).
  10 000 sims ≈ ~140 core-hours — trivial as a SLURM array job (a few hundred cores → minutes/hours).

> **Measured (100-sim pilot, 2026-06-08):** the *current* pipeline writes **~17 GB / 100 sims**
> (→ ~170 GB at 1000, ~1.7 TB at 10 000). That is far above the per-sample estimate above because
> today every bin file keeps **all ~12 correlated pairs per sim** with **Float64** label fields
> `uA,vA,uB,vB`. Confirms the two levers that bring it back in line: **Float32 labels (~2×)** and
> **one pair per sim** (vs ~12). Both are still pending.

**Conclusion:** the seed + metadata (§6) make every sim fully reproducible, so the simulation
output is *derived data*. Keep the labels+images (the expensive-to-relabel part) and the metadata;
drop the raw fields/particles. If a downstream need appears (e.g. a new displacement bin), rerun
from the stored seed + params. This is exactly why §6's metadata must be complete.

**Within what we keep, store only `u, v` (DONE 2026-06-07).** The fields writer used to save five
fields per frame — `ω` (vorticity), `s` (speed), `div` (divergence), `u`, `v`. But `ω, s, div` are
all **derived from `u, v`** (`ω = ∂v/∂x − ∂u/∂y`, `s = √(u²+v²)`, `div = ∂u/∂x + ∂v/∂y`), so storing
them is pure redundancy. The writer now saves `(; u, v)` only:

- **~60 % smaller fields file** (5 → 2 fields), stacking on top of the "don't store full sims" win above.
- **Faster** — the derived operations are no longer evaluated at every save interval.
- **Lossless** — recompute `ω`/`s`/`div`/shear from `u, v` whenever needed (e.g. the §6 metadata).
- Implemented alongside deleting the dead Makie visualization block (closed **BUG-5**), which was the
  only consumer of the stored `ω`/`s`.
- *Optional further 2×:* write `u, v` as **Float32** (plenty for PIV labels). Not done yet — needs a
  precision cast in the writer rather than a one-liner, so left as a follow-up to avoid risking the run.

---

## 5. Configurable displacement ("push") values on the command line

**Yes — make the displacement bins a command-line list.** Currently `pix_vals = [10, 20, 30]` is
hard-coded in `scripts/ImageGen.jl`. Proposal: accept e.g. `--pix 10,20,30` (or `--pix 20,50`),
parse to an `Int`/`Float` vector, and drive the existing loop with it.

**Important caveat (carry over from BUG-13/14):** achievable displacements are quantized to integer
multiples of this sim's `smax = max(speed)·Δt_save`. Requesting a `pix` below `smax` silently
floors to `dp = 1` and mislabels. The existing `@warn` (added this session) already flags any bin
whose `dp·smax` is off from the request by > 20%. So arbitrary `--pix` lists are fine **as long as
each value is ≳ 2·smax** and the user heeds the warning. The exact-label fix (frozen-field warp)
remains deferred — see `bugs.md` BUG-13/14.

---

## 6. Per-sample metadata file

**Decision: every sample gets a sidecar metadata file** (suggest JSON or TOML — human-readable,
diffable, language-agnostic; the combined JLD2 can also embed it). Fields, grouped:

**A. Reproducibility (must regenerate the exact dataset)**
- Master/base seed and the per-sim `seed_i`
- All CLI args (`jet_amp`, `n_max`, `m_jet`, `pix` list, particle count, `C`, …)
- **Git commit hashes** of *both* `2DTurbulence.jl` and the image-gen script (and ideally a
  "dirty" flag — DrWatson already appends `-dirty`; we should record the actual SHA)
- Julia version + key package versions (`Project.toml`/`Manifest.toml` hash), so the RNG and
  solver are pinned
- Grid size, domain extent, viscosity `ν`, advection scheme

**B. Initial-condition specification**
- The **functional form of the streamfunction** used (store the formula string / a version tag,
  so a future change to the IC is detectable)
- **All mode parameters:** amplitudes `a[m,n]`, phases `φ[m,n]`, jet amp `A`, jet phase `φⱼ`,
  jet wavenumber `m_jet`. *(If the full `a`/`φ` arrays are too large, the seed + code version
  regenerates them exactly — but store at least the reduced spectrum and the jet params explicitly.)*

**C. Physical characterization at the IC**
- Initial peak wavenumber `k_p` (§3 definition)
- Initial max speed `U_max`
- `(t_end)·(k_p · U_max)` — the achieved dimensionless sampling age `C`

**D. Physical characterization at the sampling time**
- Simulation time at which the sample is taken
- Peak wavenumber `k_p` at sampling time
- Energy in the peak wavenumber at sampling time — **or store the entire `E(k)` spectrum** (cheap,
  ~a few hundred floats; recommended — it subsumes the two scalars above)
- **Max shear** at sampling, in PIV units: length in **pixels**, time in **pairs** ⇒ units of `1/pairs`
- **Max vorticity** at sampling, also in `1/pairs`
  *(NB: `ω`/`s`/shear are no longer stored as fields — §4 keeps only `u, v`. Compute these scalars
  from `u, v` at the sampling frame when writing the metadata.)*

**E. Sample bookkeeping**
- Displacement bin(s) `pix`, the `dp` used, the measured `smax`, the BUG-13 warn status
- Image resolution, particle diameter / rendering params

**F. Particle counts & pool decision (§8)**
- **Rendered count** = particles actually drawn into this image (`--sample`) — the PIV density
- **Particles-per-pixel (ppp)** = rendered count / (Nx·Ny) — the resolution-independent density
- **Simulation tracer pool** `Nparticles` (the advected count) — and the **`pool = render` flag**
  (whether the 1:1 collapse from §8 is in effect, or a larger pool was kept on purpose)
- **Particle-realization augmentation:** number of distinct subsamples drawn from the same frame
  pair (1 if no augmentation), and the **subsample seed** used by `ImageGenFunc.jl` for each
  *(needed to reproduce the exact particle placement even when pool > render)*

> Rule of thumb: if in doubt, record it. The metadata is tiny next to the images, and a missing
> field can force a full re-run campaign. Anything needed to "regenerate the exact same dataset"
> (§4) **must** be here.

---

## 7. IC parameter values — why these defaults, and how to choose ranges scientifically

**The problem we discussed:** `jet_amp = 300`, `n_max = 21`, `m_jet = 2`, `ν = 1e-5` have
**no literature reference** — they were hand-picked. We agreed (a) not to always use the same
values, but (b) a naive grid over all knobs gives **millions of combinations**. How to handle this
*scientifically*?

**First, the key distinction: realization diversity ≠ regime diversity.** The current code keeps
these four parameters **fixed** and varies only the `--seed`. That is *not* the same as varying
nothing — the six random draws still fire per seed, so you get genuinely different flows:

- `A = jet_amp·(1.5 − rand())` → jet strength still varies over **[150, 450]** even with `jet_amp`
  fixed; mode amplitudes `a[m,n]`, phases `φ`, jet phase `φⱼ`, and particle positions all re-randomize.

But every such flow is drawn from the **same statistical regime**. Fixing the four parameters pins
down: `n_max` → the spectral bandwidth / range of dominant scales `k_p`; `m_jet` → the jet length
scale; `ν` → the Reynolds number; `jet_amp` → the base jet energy. So:

> **Varying the seed gives many *realizations* of one regime; varying the parameters gives different
> *regimes*** (coarse↔fine turbulence, weak↔dominant jet, more↔less viscous). Since §2 makes
> cross-sim the *only* source of diversity, a seed-only campaign yields 10 000 snapshots of a
> **single point in parameter space** — the model would generalize within that regime and be
> untested outside it. Real PIV data spans regimes, so we need regime diversity, not just realizations.

**DECISION (current): keep the four parameters FIXED at their defaults** (`jet_amp = 300`,
`n_max = 21`, `m_jet = 2`, `ν = 1e-5`) for the first dataset, with diversity coming from the
`--seed` (realization variety) only. Rationale: a **clean, controlled v1** validates the full
pipeline + training first, and lets us *see* what a representative flow looks like before committing
to any ranges (recall: "no reference for proper values"). The realization-vs-regime limitation above
is accepted for now — the v1 model is expected to generalize *within* this regime, not across regimes.

---

### Future plan (v2): vary the parameters for regime diversity

> Everything below is **not implemented and not part of v1.** It is the plan for when we move from
> a single-regime pilot to a multi-regime production campaign. Captured here so the design is ready.

**Principle: randomize the *physics*, not the raw code knobs.** Map each code parameter to the
dimensionless physical quantity it controls, choose a defensible **range** for that quantity, and
sample. The quantities that actually matter for 2D-turbulence PIV:

| Code knob | Physical meaning | Suggested range & rationale |
|---|---|---|
| `n_max` | spectral bandwidth → sets initial `k_p` (dominant scale) | vary over odd values, e.g. **11–41**. Larger ⇒ finer scales / higher `k_p`. Span "few large eddies" → "many small eddies". |
| `jet_amp` (→ `A`) | jet-to-turbulence **energy ratio** | parameterize as a **fraction of total energy** rather than a raw amplitude; sweep from **no jet (0)** to **jet-dominated**. Reframing avoids the arbitrary "300". |
| `m_jet` | jet length scale (number of jet wavelengths) | **1–4** (small integers) for different jet scales/orientations. (Note BUG-6: rounding can give wavenumber 0/neg — decide if that's intended variety.) |
| `ν` | **Reynolds number** `Re ~ U·L/ν` | vary `ν` (or `Re`) over a decade or two to span the inviscid→mildly-viscous range we sample in. |
| particle count | seeding density (ppp) | see §8 |

**Sampling strategy (the "millions of combinations" answer):** don't grid. Use **quasi-random /
space-filling sampling** of the parameter ranges:

- **Latin Hypercube Sampling (LHS)** or **Sobol sequences** give near-uniform coverage of the
  high-dimensional parameter box with *far fewer* points than a grid, and no axis-aligned gaps.
- Draw one parameter vector per sim (tie it to `seed_i` so it's reproducible). With 10 000 sims,
  LHS/Sobol over ~4–5 parameters covers the space densely.
- This is standard **design-of-experiments** practice and is the defensible, citable answer to
  "how did you choose parameters": *we sampled physically-motivated ranges with a space-filling
  design*, not "we guessed."

**What the literature anchors (for the ranges, not exact values):**
- Decaying-2D-turbulence DNS studies start from **random superpositions of harmonic modes between
  specified wavenumbers, with the spectrum peaking at a chosen `k_p`** — exactly our IC family.
  This justifies parameterizing by `k_p` / bandwidth rather than raw `n_max`.
- The canonical deep-learning PIV training set (Cai et al.) deliberately **randomizes** particle
  density, diameter, and intensity and mixes flow types (DNS turbulence, SQG, JHTDB) — i.e. the
  field's best practice is *diversity by randomized ranges*, which is what we are proposing.

> **Recommendation:** put each of the above ranges in the metadata (§6) and sample them per sim
> via LHS/Sobol keyed to `seed_i`. Calibrate the *ranges* with a small pilot sweep (a few hundred
> sims) before the full 10 000-run campaign.

*(A deeper, citation-by-citation literature review of specific `k_p`/Re ranges for 2D-turbulence
PIV would be a good next step if we want named references in a paper — flag if you want me to run it.)*

---

## 8. Particle count — make it a variable, with a literature-backed range

**Two knobs, don't confuse them:**
- **Image density** = `--sample`/`-k` in ImageGen, **default 5000** = the particles drawn into each
  PIV image. **This is the PIV-relevant number** (already a CLI variable). 5000 / 512² = **0.019 ppp**.
- **Simulation tracer pool** = `Nparticles = M·N/16 = 16384`, hard-coded in `2DTurbulence.jl`. It
  only sets the *maximum* renderable (render = `min(--sample, pool)`); it is not the image density.

**Do we even need the 16384 pool? (consequence of one-sample-per-sim)**

No — with **one sample per sim (§2), the tracer pool only needs to equal the rendered count.**
For a single image you need exactly `--sample` particle positions, and a random 5000-of-16384
(uniform) is *statistically identical* to 5000 simulated directly. So the extra ~11 000 advected
particles contribute **nothing** to that one image — they are wasted advection work, a vestige of
the old multi-image-per-sim approach.

The *only* thing a pool larger than the render count buys is **particle-realization augmentation**:
drawing several different subsets of the *same* frame pair → multiple images sharing one velocity
label but differing in particle placement. (Note this is **not** ruled out by the decorrelation
finding, which was about *time-separated* frames; this is *same-time, different particles*.) But its
marginal value is **low here**, because diversity already comes from **10 000 independent flows**,
each with randomly-placed particles — the dataset already has huge particle-arrangement variety
*across* sims.

**Impact of collapsing pool → render count:**
- *Data quality:* none (statistically identical for one image).
- *Compute:* saves advecting the extra particles, but the **512² WENO field solve almost certainly
  dominates runtime**, so the wall-clock saving is modest (worth a quick timing check, not assumed).
- *Design:* clean win — `pool = --sample` eliminates the `min(--sample, pool)` cap and ties one knob
  (density/ppp) directly to cost; no separate pool arg to manage.

> **Recommendation:** set the simulation tracer count **equal to that sim's rendered count** (1:1),
> driven by the chosen density below. If per-sim density varies (0.02–0.1 ppp → 5000–26000),
> simulate exactly that many. Keep a pool *above* the render count **only** if you later decide you
> want same-flow particle-realization augmentation — marginal given 10 000 flows.

**What the literature says about seeding density** (express as **particles-per-pixel, ppp**, so it
is resolution-independent):

- **Classic cross-correlation PIV:** aim for **~8–25 particles per 32×32 interrogation window**
  (minimum ~10 to beat the random-correlation peak). On a 32² = 1024 px window that is
  ≈ **0.008–0.024 ppp**.
- **Deep-learning PIV training data (Cai et al. canonical set):** seeding **0.05–0.1 ppp**,
  particle diameter **1–4 px**, peak intensity 200–255. Benchmarks also probe **0.01 / 0.05 / 0.1
  ppp** and sparse↔dense splits.
- **Current image density (the default 5000)** = 5000 / 512² = **0.019 ppp** — the *sparse/classic*
  end, **below** the Cai DL-standard. The 16384 pool (0.0625 ppp) is only the reservoir, never the
  rendered density.

**Recommendation:** sample image density per sim over a range that spans **classic-sparse →
DL-dense**, e.g. **0.02 – 0.1 ppp** (≈ **5 000 – 26 000 particles** on 512²). Training across a
*range* of densities makes the model robust to seeding density at inference — a known best practice
from the Cai-style randomized datasets. Record the exact rendered count and ppp in the metadata (§6).

**Action items:** (a) drive both the render count and the tracer count from one density (ppp) knob,
with `Nparticles = rendered count` (1:1, per the pool discussion above) — this also makes the dense
end (0.1 ppp ≈ 26 000) renderable without a separate cap; (b) if a single value is wanted,
**0.05 ppp ≈ 13 000** sits in the literature sweet spot; the current default 5000 (0.019 ppp) is
fine as the sparse stress-test end.

---

## Open questions / next steps

1. ~~Reconcile particle count~~ **RESOLVED:** 5000 = particles *rendered per image* (`--sample`
   default, the PIV density = 0.019 ppp); 16384 = the *simulation tracer pool* it's drawn from.
   Both should become CLI vars; raise the pool above 16384 to reach the 0.1 ppp dense end (§8).
2. **Calibrate `C`** (§3) with a small pilot sweep before the full run.
3. **Implement (v1):** `--n_sims`, `--pix` list, `--n_particles`/`--ppp`, `k_p`/`E(k)` computation,
   eddy-time `t_end`, and the metadata writer. *(IC params stay FIXED for v1 — §7 decision.)*
4. **Decide label precision:** approximate labels (current, [10,20,30]) vs. exact (frozen-field
   warp, deferred BUG-13/14) — revisit once a first dataset has trained a model.
5. **Optional:** a companion notebook that computes & plots `E(k)`, `k_p`, the eddy-turnover time,
   and a `C`-sweep on a real sim, for presentation.
6. **Future (v2):** parameter variation via LHS/Sobol over physical ranges (§7 future-plan
   subsection) + the deeper literature review of `k_p`/Re ranges — only when moving beyond the
   single-regime pilot.

---

### Sources (literature touchpoints)

- Cai et al., *Particle Image Velocimetry Based on a Deep Learning Motion Estimator* / PIV dataset —
  https://github.com/shengzesnail/PIV_dataset (0.05–0.1 ppp, diameter 1–4 px, DNS/SQG/JHTDB flows)
- MCFormer benchmark for PIV — https://arxiv.org/abs/2507.04750 (density/displacement splits)
- Cross-correlation PIV review (Adrian/Westerweel-style guidance), Wikipedia PIV — particles-per-window rules of thumb
  https://www.aa.washington.edu/sites/aa/files/faculty/dabiri/pubs/piV.Review.Paper.final.pdf ,
  https://en.wikipedia.org/wiki/Particle_image_velocimetry
- Decaying 2D turbulence ICs & eddy-turnover time (random-mode IC, `t* = 1/ω_rms`) —
  https://arxiv.org/pdf/2108.01137 , https://arxiv.org/abs/astro-ph/0312505
