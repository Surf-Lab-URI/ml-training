# 🐛 Bug Tracker — ML Training Data Pipeline

> 2D Turbulence → Combine and Conquer → Image Gen
>
> Tracks known issues across `args.jl`, `2DTurbulence.jl`, `CombineAndConquer.jl`, and `ImageGenFunc.jl`.
> Update the **Status Checklist** at the bottom as you fix each one.

---

## Bug Details

### BUG-1 — `dt` used before it is defined
- **File:** `src/args.jl`
- **Severity:** 🔴 High — crashes on the default code path
- **Crashes?** Yes (`UndefVarError: dt not defined`) whenever `-t` is **not** passed
- **Description:** `args.jl` is `include`d near the top of `2DTurbulence.jl`, but it computes
  `st = parsed_args["nt"]*dt`. The variable `dt` is not defined until later in
  `2DTurbulence.jl` (`dt = tcfl*10`). So with `automate=true` and no `-t`, this line throws.
  The README example `julia 2DTurbulence.jl -t 100 -m false` dodges it by passing `-t`.
- **Suggested fix:** Don't compute `st` in `args.jl`. Keep the raw values:
  ```julia
  # args.jl
  t_end = parsed_args["t_end"]
  nt    = parsed_args["nt"]
  ```
  Then compute `st` in `2DTurbulence.jl` right after `dt` exists:
  ```julia
  dt = tcfl*10
  st = isnothing(t_end) ? nt*dt : t_end
  ```

---

### BUG-2 — Integer default on a `Float64` argument
- **File:** `src/args.jl`
- **Severity:** 🟢 Low — usually auto-converts, but fragile / version-dependent
- **Crashes?** Maybe (depends on ArgParse version)
- **Description:** `--jet_amp` has `arg_type = Float64` but `default = 300` (an `Int` literal).
- **Suggested fix:** Use a float literal: `default = 300.0`.
- **Note:** Leave `--t_end` (`default = nothing`) as-is — that's intentional and tested for.

---

### BUG-3 — `Bool` argument parsing is unreliable
- **File:** `src/args.jl`
- **Severity:** 🟡 Medium — `-m false` may silently misbehave
- **Crashes?** No, but may parse to the wrong value
- **Description:** `--automate` uses `arg_type = Bool, default = true`. ArgParse's `Bool`
  handling is finicky; `-m false` is not guaranteed to parse to `false` across versions.
  The README depends on `-m false` working exactly.
- **Suggested fix (preferred):** Switch to a store-true flag (matches `--save_pngs` style):
  ```julia
  "--no_image_gen"
      help = "skip image pair generation after combining"
      action = :store_true
  ```
  Then in the main script: `automate = !parsed_args["no_image_gen"]`.
  (Update README, since the CLI changes.)
- **Suggested fix (minimal):** Keep `-m false` but verify it parses correctly on your
  installed ArgParse version before any large run.

---

### BUG-4 — `progress_message` defined but never registered
- **File:** `scripts/2DTurbulence.jl`
- **Severity:** 🟢 Low — no progress output during runs
- **Crashes?** No
- **Description:** `progress_message(sim)` is defined but never attached as a callback
  (unlike the `TimeStepWizard`). On HPC you get no liveness signal in the SLURM log.
- **Suggested fix:** Register it:
  ```julia
  simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(10))
  ```
- **Style note:** `return @info @sprintf(...)` is odd — `@info` returns `nothing`.
  Just `@info @sprintf(...)` (no `return`) is cleaner.

---

### BUG-5 — Commented-out `Makie.record` block is incomplete
- **File:** `scripts/2DTurbulence.jl`
- **Severity:** 🟢 Low — harmless while commented
- **Crashes?** Only if uncommented (missing `end`; references undefined `filename`)
- **Description:** The commented movie block is missing its closing `end` and uses
  `filename`, which is never defined (the script uses `vars`). Also: the **entire**
  Makie visualization section after `run!(simulation)` builds a figure that is never
  `display`ed, `save`d, or `record`ed — pure dead weight on a headless node.
- **Suggested fix:** Either finish the `record` call properly (define `filename`, add `end`),
  or delete the whole visualization block from `set_theme!` through the commented section.
- **🟢 FIXED (2026-06-07):** Deleted the entire post-`run!` visualization block (the Makie figure,
  the particle-reading helpers, and the incomplete commented `Makie.record`). It produced no output
  on a headless node. Removed as part of the "store only `u, v`" change — it was the sole consumer
  of the stored `ω`/`s` fields (see also the §4 storage decision in `notes/DATA_GENERATION_DESIGN.md`).

---

### BUG-6 — Jet wavenumber can be zero or negative
- **File:** `scripts/2DTurbulence.jl`
- **Severity:** 🟢 Low — produces bad/unexpected data, not a crash
- **Crashes?** No
- **Description:** In the stream function, `round(mjet*sin(ϕⱼ))` and `round(mjet*cos(ϕⱼ))`
  feed into `k(n)=2π(n-1)/N`. With `mjet=2`, the rounded value ranges over `-2..2`
  depending on the random phase. `n=1` → wavenumber 0 (constant); `n<1` → negative
  wavenumber. The intended jet structure may sometimes be near-constant or reversed.
- **Suggested fix:** Decide if this is intended variety. If not, clamp or offset the index
  so the jet always has a sensible positive wavenumber.

---

### BUG-7 — `CombineAndConquer.jl` loads everything into RAM
- **File:** `src/CombineAndConquer.jl`
- **Severity:** 🟡 Medium — OOM risk at scale on SLURM
- **Crashes?** Only at scale (out-of-memory kill)
- **Description:** Uses `load(fields_file)` and `load(particles_file)` to pull both full
  files into memory, then holds a merged third copy before writing. For 512×512 fields ×
  many timesteps + 16384 particles × timesteps, peak memory can spike and exceed a tight
  SLURM memory request.
- **Suggested fix:** Only if you actually hit memory limits — open files lazily with
  `jldopen` and copy groups across without fully materializing everything at once.

---

### BUG-8 — No master `--seed` for reproducibility
- **Files:** `src/args.jl`, `scripts/2DTurbulence.jl`, ImageGen launch line
- **Severity:** 🟡 Medium — runs are not reproducible
- **Crashes?** No
- **Description:** All simulation `rand()` calls (`x₀`, `y₀`, mode amplitudes `a`, phases
  `ϕ`, `ϕⱼ`, and `A` in `args.jl`) draw from Julia's global RNG with no seed. Only the
  image-gen subsampling (`MersenneTwister(seed)` in `ImageGenFunc.jl`) is seeded.
  Identical CLI args still produce different flows.
- **Suggested fix — 4 moves to thread one master `--seed`:**
  1. **Add `--seed` to `args.jl`** (no short flag — `-s` already clashes with `sim_vars`
     in `CombineAndConquer.jl` and `seed` in `ImageGenFunc.jl`):
     ```julia
     "--seed"
         help = "master random seed for reproducibility"
         arg_type = Int
         default = 1234
     ```
  2. **Remove** `A = jet_amp*(1.5-rand())` **from `args.jl`** — it fires before any seed
     could possibly be set, because `args.jl` runs on `include`.
  3. **In `2DTurbulence.jl`, immediately after the `include`:**
     ```julia
     using Random
     Random.seed!(parsed_args["seed"])
     A = parsed_args["jet_amp"]*(1.5-rand())   # moved from args.jl
     ```
  4. **Pass the seed to ImageGen on launch** (only on the ImageGen call, not the Combine
     call — `-s` collision with `sim_vars`):
     ```julia
     run(`$(Base.julia_cmd()) $(projectdir() * "/scripts/ImageGen.jl") \
          -f $(out_dir * "combined" * vars * ".jld2") -v $(vars) -s $(parsed_args["seed"])`)
     ```
  Because all sim `rand()` calls use the global RNG, one `Random.seed!` covers `x₀`, `y₀`,
  `a`, `ϕ`, `ϕⱼ` automatically. DrWatson's `tag!(parsed_args)` then stamps the seed into
  output metadata for free.
- **Related cleanup:** The `-s` flag means two different things across the three scripts
  (sim_vars vs seed). Consider standardizing on `--seed` with no short form.

---

### BUG-9 — Undefined `field_file` / `particle_file` in single-file mode
- **File:** `src/CombineAndConquer.jl` (lines 62–65)
- **Severity:** 🔴 High — crashes the auto-pipeline every time
- **Crashes?** Yes — `UndefVarError: field_file not defined`
- **Description:** Refactor that added `--input_dir` (batch) mode introduced this bug in
  the single-file `else` branch:
  ```julia
  else
      field_files = [field_file]      # undefined — parsed name is `fields_file`
      particle_files = [particle_file] # undefined — parsed name is `particles_file`
  end
  ```
  `2DTurbulence.jl` always calls this script with `-f` and `-p` (no `-i`), so the auto-
  chain hits this branch immediately and dies.
- **Suggested fix:** Match the names assigned from `parsed_args` (lines 43–44):
  ```julia
  else
      field_files = [fields_file]
      particle_files = [particles_file]
  end
  ```

---

### BUG-15 — Particle render `xlim`/`ylim` mismatch causes pixel-position teleport
- **File:** `scripts/ImageGen.jl` (the `make_image_pair` call, lines ~97–98)
- **Severity:** 🔴 Critical — every generated training pair is unusable
- **Crashes?** No — silently produces nonsense data
- **Description:** `scripts/2DTurbulence.jl` builds the simulation grid with
  `extent=(N, M) = (512, 512)` and seeds particles with `rand(Nparticles)*M` and
  `rand(Nparticles)*N`, so particle positions live in `[0, 512)`. But ImageGen calls
  `make_image_pair` with `xlim = (0.0, 2π)` and `ylim = (0.0, 2π)`. Inside
  `render_particles`, the position-to-pixel map is:
  ```julia
  x_wrapped = xmin .+ mod.(x .- xmin, Lx)   # Lx = 2π ≈ 6.28
  u = (x_wrapped .- xmin) ./ Lx .* (width - 1)
  ```
  So a particle at physical x=300 maps to `(300 mod 6.28) / 6.28 * 511 ≈ pixel 393`,
  and the same particle one moment later at x=305 maps to pixel ~266 — **a 5-unit
  physical drift becomes a 127-pixel jump on screen**. Every particle effectively
  teleports to a pseudo-random pixel each frame.
- **Symptom:** Images A and B look unrelated even though the sim's physical motion
  between them is only a few units. The image data carries no signal about the saved
  flow field. Any ML model trained on this would learn pure noise.
- **Verification:** before the fix, the Section 5 patch cross-correlation in
  `visualize_pair.ipynb` returns large random shifts (5–30 px) with `RMSE >> 1 px`.
  After the fix, shifts should be ≤ smax and `RMSE < 1 px`.
- **Suggested fix:** Match `xlim`/`ylim` to the sim grid extent. Quick fix (consistent
  with the existing `width = 512` / `height = 512` style):
  ```julia
  xlim = (0.0, 512.0),
  ylim = (0.0, 512.0),
  ```
  Better long-term fix: read the actual extent from the combined file's grid
  serialization (`fields/serialized/grid`) so `xlim`/`ylim` stay synchronized with
  whatever the sim used. The function defaults in `src/ImageGenFunc.jl`'s
  `render_particles` also have `xlim = (0.0, 2π)` — should be updated for consistency
  (currently overridden by the call site).
- **Impact on past work:** every `data/visual/pix*/` dataset generated before this fix
  is invalid. Re-run ImageGen only (no sim re-run needed) with the corrected `xlim`.

---

### BUG-14 — No sub-frame interpolation; small-`pix` bins fundamentally limited by `dt`
> **FIXED 2026-08-13 in `datagen_v2/FracFrame.jl`** (commit `fc211e7`; velocity fields followed in
> `4277a0a`). Particle positions are interpolated between saved frames to synthesise a virtual
> frame at any time, so the frame gap is continuous rather than an integer. Self-test: cubic
> interpolation error 0.00005 px against exact analytic trajectories, periodic seam handled, all
> eight bins hit their target median in two solver iterations. Confirmed on the delivered dataset —
> every v2 bin's measured median matches its name to ~0.1 px. See
> `datagen_v2/DATA_REQUIREMENTS.md` R2 and §4b.
- **File:** `scripts/ImageGen.jl` (the pair-construction loop), `src/ImageGenFunc.jl`
- **Severity:** 🟡 Medium — design limitation, not a code defect
- **Crashes?** No
- **Description:** ImageGen can only pair frames that *exist* in the saved simulation
  output. A and B are always chosen as `frame_keys[p]` and `frame_keys[p + dp]` where
  `dp ≥ 1` is an integer. The smallest achievable displacement per pair is therefore
  `smax = max(speed) * Δt_save` (the motion in one saved interval). Any requested
  `pix < smax` is unachievable — that's the root cause of [[BUG-13]].
  The only current workarounds are to (a) shorten `Δt_save` in `2DTurbulence.jl`
  (line ~79: `dt = tcfl*10`) and re-run the whole sim, or (b) slow the flow
  (`--jet_amp`). Both change the dataset rather than expanding what ImageGen can do
  with a given combined file.
- **Suggested fix (design):** Pair *interpolated* particle positions, not raw saved
  frames. Linearly interpolate particle (x, y) between two consecutive saves to produce
  a "virtual" frame at any time `t ∈ [t_a, t_b]`. Then for a target `pix`, compute the
  exact Δt that produces ~pix max displacement and render A/B at `(t, t + Δt)`.
  - Pros: any `pix` value becomes achievable from any combined file; no re-simulation
    needed when changing pix bins; smooths the dataset.
  - Cons: linear interpolation introduces small position error for curving particles
    (cubic spline mitigates); also need to interpolate the velocity field for the
    saved `uA/vA/uB/vB` ground truth.
- **Relationship to [[BUG-13]]:** BUG-13 is the *symptom* (silent mislabel when the
  request is unachievable). BUG-14 is the *root cause* fix (make every request
  achievable). Either fix alone is acceptable; doing both gives the best UX.

---

### BUG-13 — `dp` floor at 1 silently mislabels low-`pix` bins
> **FIXED 2026-08-13 in `datagen_v2/ImageGenV2.jl`** (commit `fc211e7`), together with
> [[BUG-14]], whose fix this depended on. The v2 generator solves for a *fractional* frame index,
> so any target displacement is reachable, and it bins by **median** rather than by a nominal
> maximum. A sample whose achievable median misses its target by more than `[bins.v2].tolerance`
> is **skipped rather than written under a label it does not have** — the failure mode below is
> now impossible by construction. `scripts/ImageGen.jl` is unfixed and is kept only to reproduce
> the 2026-06/07 datasets; it still emits the warning quoted below.
- **File:** `scripts/ImageGen.jl` (line ~47)
- **Severity:** 🟡 Medium — produces mislabeled training data
- **Crashes?** No — silently produces wrong data
- **Description:** `dp = max(1, Int(floor(pix / smax)))` clamps the inter-frame gap to a
  minimum of 1. When the saved-frame interval already produces motion bigger than the
  target `pix` (i.e. `smax > pix`), the requested `pix` is unachievable but the file is
  still written under that label. For the `--nt 5` smoke run: smax ≈ 5 px/frame, so
  pix=3 and pix=5 both clamp to `dp=1` and produce **bit-for-bit identical flow fields**
  (only the image subsamples differ due to advancing RNG state).
- **Verification:** loading `data/visual/pix3/...jld2` vs `data/visual/pix5/...jld2` and
  comparing `pairs/000001/fields/uA` shows `np.array_equal == True` for every pair.
  Max displacement in pix3 is ~5 px, not ~3 px as the label implies.
- **Suggested fix (one of):**
  - **Warn at runtime** when a pix target can't be achieved:
    ```julia
    actual_max = dp * smax
    if actual_max > pix * 1.2
        @warn "pix=$pix unachievable; minimum is $(round(actual_max, digits=2)) px (Δt too coarse)"
    end
    ```
  - **Skip mislabeled bins** instead of silently writing duplicate data:
    ```julia
    if dp * smax > pix * 1.2
        @info "Skipping pix=$pix: would duplicate larger bin (smax=$(smax))"
        continue
    end
    ```
  - **Increase save resolution** for small-pix runs: lower `dt = tcfl*10` (in
    `scripts/2DTurbulence.jl` line ~79) to e.g. `dt = tcfl*5`, which halves `smax`.
    Tradeoff: more saved frames, bigger files.
- **Current decision (2026-06-07):** DEFER the real fix (frozen-field warp, BUG-14).
  Moved `pix_vals` to `[10, 20, 30]` (all ≥ 2·smax with smax≈5, so distinct dp ≥ 2 →
  no bit-identical bins) and added a runtime `@warn` in `ImageGen.jl` that flags any sim
  whose `dp*smax` deviates from the requested `pix` by >20% (catches occasional hot sims
  that floor to dp=1). Labels remain approximate (biased high by floor + per-sim smax
  drift), accepted as good-enough for the first dataset. Status stays 🔴 until the warp lands.
- **Confirmed in the wild (2026-06-09, 20-sim nt=40 run):** the assumption in the
  2026-06-07 note ("smax≈5 with smax<5 → clean dp=2,4,6") is FALSE in practice. By
  construction `dt = 5/U_max` ⇒ `smax ≈ 5.0` *exactly*, sitting on a knife-edge. Across the
  20 sims smax landed mostly just **above** 5.0 (5.00–5.03), so `floor(10/5.00x)=1`,
  `floor(20/..)=3`, `floor(30/..)=5` → actual displacements **{5, 15, 25}**, not {10,20,30}.
  13/20 sims tripped the @warn on pix10 & pix20; pix30 is off by ~17% (just under the 20%
  warn threshold, so silent). Worse, the ~7 sims with smax just *below* 5.0 floored cleanly to
  {10,20,30}, so **each `pixN/` folder is an inconsistent mix** of two displacement values.
  KEY CLARIFICATION: only the *bin name* is wrong — the stored velocity-field labels
  (`velocity × Δt_pair`, `Δt_pair = dp·dt`) exactly match each pair's real particle motion, so
  (image-pair → displacement-field) training data is correct & self-consistent. **User decision
  (2026-06-09): ACCEPT AS-IS** for the 20-sim dataset — treat the velocity fields as the only
  labels, ignore the folder names. Real fix (frozen-field warp / off-knife-edge bins) still
  deferred. Status stays 🔴.

---

### BUG-12 — Combined-file naming mismatch between writer and consumer
- **Files:** `scripts/2DTurbulence.jl` (the `@info` line and the ImageGen launch),
  `src/CombineAndConquer.jl` (the `jldsave` line)
- **Severity:** 🔴 High — crashes the auto-pipeline at ImageGen
- **Crashes?** Yes — `SystemError: opening file ... No such file or directory`
- **Description:** `CombineAndConquer.jl` writes the combined file as
  `out_dir * vars * "_combined.jld2"` (e.g. `_<timestamp>...nmax21-mjet2_combined.jld2`).
  But `2DTurbulence.jl` reports the file as, and passes to ImageGen, the path
  `out_dir * "combined" * vars * ".jld2"` (e.g. `combined_<timestamp>...nmax21-mjet2.jld2`).
  Different prefix/suffix arrangement → file not found.
- **Suggested fix:** Match the writer. In `2DTurbulence.jl`, both the `@info` message and
  the ImageGen `-f` arg should use:
  ```julia
  $(out_dir * vars * "_combined.jld2")
  ```
  This also keeps the canonical "ends with `_combined.jld2`" convention that
  `ImageGen.jl`'s `--input_dir` mode relies on for filtering.

---

### BUG-11 — Child Julia processes launched without `--project`
- **File:** `scripts/2DTurbulence.jl` (the two `run(...)` lines that launch child scripts)
- **Severity:** 🔴 High — crashes the auto-pipeline every time
- **Crashes?** Yes — `ArgumentError: Package DrWatson not found in current path`
- **Description:** Both child-process launches use `$(Base.julia_cmd())` without a
  `--project=$(projectdir())` flag. `Base.julia_cmd()` returns the bare julia executable
  command, so the child process starts in Julia's default (global) environment, where
  the `ml-training` project's packages (DrWatson, JLD2, Images, …) are not installed.
  The child script then dies on its first `using DrWatson`.
- **Suggested fix:** Pass the project explicitly to each child process:
  ```julia
  run(`$(Base.julia_cmd()) --project=$(projectdir()) $(...CombineAndConquer.jl) -f ... -p ... -s ...`)
  run(`$(Base.julia_cmd()) --project=$(projectdir()) $(...ImageGen.jl) -f ... -v ...`)
  ```

---

### BUG-10 — `ImageGen.jl` ignores `-f`, only uses `input_dir`
- **File:** `scripts/ImageGen.jl` (line 21)
- **Severity:** 🔴 High — crashes the auto-pipeline every time
- **Crashes?** Yes — `MethodError: no method matching readdir(::Nothing, ...)`
- **Description:** The script parses both `-f / --combined_file` and `-d / --input_dir`,
  but the main loop only uses `input_dir`:
  ```julia
  infiles = filter(f -> endswith(f, "_combined.jld2"), readdir(input_dir, join = true))
  ```
  `input_dir` defaults to `nothing`. `2DTurbulence.jl` invokes with `-f <combined_file>`
  (no `-i`), so this fails before any pair is generated.
- **Suggested fix:** Branch on which input form was provided, fall back to a clear error
  if neither:
  ```julia
  infiles = if input_dir !== nothing
      filter(f -> endswith(f, "_combined.jld2"), readdir(input_dir, join = true))
  elseif file !== nothing
      [file]
  else
      error("Must provide either --combined_file (-f) or --input_dir (-d)")
  end
  ```

---

## Status Legend

| Badge | Meaning |
|-------|---------|
| 🔴 `PENDING` | Not started |
| 🟡 `IN PROGRESS` | Being worked on |
| 🟢 `FIXED` | Done & verified |
| ⚪ `WONT FIX` | Intentional / not worth it |


---

## ✅ Status Checklist

> Edit the **Status** column as you progress. Colors come from the emoji badges.

| ID | Bug | File | Status |
|----|-----|------|--------|
| BUG-1 | `dt` used before defined | `args.jl` | 🟢 FIXED |
| BUG-2 | Int default on Float64 arg | `args.jl` | 🟢 FIXED |
| BUG-3 | `Bool` arg parsing unreliable | `args.jl` | 🟢 FIXED |
| BUG-4 | `progress_message` not registered | `2DTurbulence.jl` | 🟢 FIXED |
| BUG-5 | Incomplete commented movie block | `2DTurbulence.jl` | 🟢 FIXED |
| BUG-6 | Jet wavenumber 0/negative | `2DTurbulence.jl` | 🔴 PENDING |
| BUG-7 | Combine loads all into RAM | `CombineAndConquer.jl` | 🔴 PENDING |
| BUG-8 | No master `--seed` for reproducibility | `args.jl` / `2DTurbulence.jl` | 🟢 FIXED |
| BUG-9 | Undefined `field_file`/`particle_file` in single-file mode | `CombineAndConquer.jl` | 🟢 FIXED |
| BUG-10 | `ImageGen.jl` ignores `-f`, only uses `input_dir` | `ImageGen.jl` | 🟢 FIXED |
| BUG-11 | Child Julia processes launched without `--project` | `2DTurbulence.jl` | 🟢 FIXED |
| BUG-12 | Combined-file naming mismatch between writer and consumer | `2DTurbulence.jl` | 🟢 FIXED |
| BUG-13 | `dp` floor at 1 silently mislabels low-`pix` bins | `ImageGen.jl` | 🟢 FIXED in `datagen_v2/` |
| BUG-14 | No sub-frame interpolation; small-pix bins limited by `dt` | `ImageGen.jl` | 🟢 FIXED in `datagen_v2/` |
| BUG-15 | Particle render xlim/ylim mismatch (2π vs 512) → teleport | `ImageGen.jl` | 🟢 FIXED |




<!-- --- -->

<!-- ### Progress

- 🔴 **Pending:** 7
- 🟡 **In progress:** 0
- 🟢 **Fixed:** 0
- ⚪ **Won't fix:** 0

---

## Priority Order (suggested)

1. **BUG-1** — blocks the default run path.
2. **BUG-3** — `-m false` may silently fail.
3. **BUG-7** — OOM risk on HPC at scale.
4. Then cleanup: **BUG-2**, **BUG-4**, **BUG-5**, **BUG-6**.

--- -->

<!-- _Last updated: 2026-05-26_ -->


---
 
---

## Issues Details:
### Why there are two different knobs for number of particles? 
- `k_particles` is in `src/ImageGenFunc.jl`
- `subset_particles` is in `scripts/ImageGen.jl`
