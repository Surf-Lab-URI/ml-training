# Data generation — complete reference

Everything established about how this data is made, why, and what actually exists. Written
2026-09-01 as a single place to check things later.

**How to read the confidence markers.** Every quantitative claim below is marked:

- **[measured]** — computed from files or run output, with the command given so it can be redone.
- **[source]** — read directly out of the code or a `RUN_INFO.txt`.
- **[design]** — a decision with reasoning, not a measurement.
- **[unverified]** — believed but not checked. Treat with suspicion.

Companion documents, none of which this replaces:
`README.md` (how to run), `params.toml` (every setting, commented),
`datagen_v2/DATA_REQUIREMENTS.md` (why the v2 bins are what they are),
`notes/DATA_GENERATION_DESIGN.md` (the original campaign design), `bugs.md`.

---

## 1. What the pipeline produces

A **sample** is one pair of 512×512 particle images plus the velocity field that labels them:

```
pairs/000001/A              512x512 uint8    frame A
pairs/000001/B              512x512 uint8    frame B
pairs/000001/fields/uA,vA   512x512 float32  displacement at A, IN PIXELS
pairs/000001/fields/uB,vB   512x512 float32  displacement at B, in pixels
```

The fields are already displacement in pixels — velocity multiplied by the pair's time gap — so
`sqrt(uA² + vA²)` is what a PIV algorithm should recover. No unit conversion anywhere. **[source]**
`src/ImageGenFunc.jl`, `save_field_pairs`.

### 1.1 The images are Lagrangian, the labels are Eulerian

This distinction is easy to miss and it has a consequence.

| | kind | what it is |
|---|---|---|
| images `A`, `B` | **Lagrangian** | rendered from the positions of tracked material particles at two times |
| labels `uA,vA,uB,vB` | **Eulerian** | the velocity field on the fixed 512×512 grid, times Δt |

A particle's true path from A to B is not exactly the midpoint Eulerian velocity times Δt, because
the particle moves through a changing field on the way. The self-test measures that residual at
**0.13 px** **[measured, on a mock steady field]** — `datagen_v2/DATA_REQUIREMENTS.md` §4b.

**Consequence:** there is a small irreducible floor between the labels and what a perfect algorithm
could recover from the images. At 0.13 px it is far below the ~1.4 px near-surface error being
chased in `piv-models`, so it does not explain that. **It has never been measured on real
production data, only on the mock field, and it is expected to grow with displacement and flow
curvature** — so it is probably larger in `med26`/`med30`. **[unverified]**

---

## 2. Three stages

```
scripts/2DTurbulence.jl      1. SIMULATE  2-D turbulence + particles, one run per seed
                                          -> combined/seed<N>.jld2
datagen_v2/ImageGenV2.jl     2. RENDER    saved frames -> image pairs at chosen displacements
scripts/make_report.py       3. CHECK     a PDF describing what came out
```

**Stage 2 runs no physics.** It reads frames stage 1 already saved. This is the single most
important operational fact about the project: **both datasets currently in use are re-renders of
one June 2026 campaign, and no new physics has been generated since 2026-06-12.** **[source]** —
the `source_run` line in each `RUN_INFO.txt`.

That is why `[run].keep_combined` matters. Keeping `combined/` costs ~900 MB per simulation and
allows a completely new dataset in minutes; discarding it permanently forecloses that for that
campaign. **A one-way door.**

`scripts/ImageGen.jl` is the legacy v1 renderer, kept only so the 2026-06 and 2026-07 datasets stay
reproducible. Its bin labels do not mean what they say — see §6.

---

## 3. What exists on disk

`/project/pi_nicholas_pizzo_uri_edu/arup/piv_2dturb_dataset`, **1.5 TB total** **[measured]**:

<table style="width: 100%; table-layout: fixed;">
<colgroup>
<col style="width: 30%;">
<col style="width: 10%;">
<col style="width: 60%;">
</colgroup>
<thead>
<tr>
<th align="left">Run</th>
<th align="right">Size</th>
<th align="left">What it is</th>
</tr>
</thead>
<tbody>
<tr style="background:transparent;">
<td><code style="background:transparent; padding:0; border:none; border-radius:0; white-space: normal; overflow-wrap: anywhere;">run_2026-06-12_04-50-52</code></td>
<td align="right">1012&nbsp;GB</td>
<td>The <b>base simulation campaign</b>, and the source of both re-renders. Clean-synthetic appearance. Holds <code style="background:transparent; padding:0; border:none; border-radius:0;">combined/</code> (880&nbsp;GB), which is what makes re-rendering possible. No model trained on its own <code style="background:transparent; padding:0; border:none; border-radius:0;">pix*</code> renders.</td>
</tr>
<tr style="background:transparent;">
<td><code style="background:transparent; padding:0; border:none; border-radius:0; white-space: normal; overflow-wrap: anywhere;">run_labapp_2026-07-06_20-28-55</code></td>
<td align="right">133&nbsp;GB</td>
<td>v1 bins, <b>lab appearance</b>. The production model's training data — <b>27 trained models</b> point at it.</td>
</tr>
<tr style="background:transparent;">
<td><code style="background:transparent; padding:0; border:none; border-radius:0; white-space: normal; overflow-wrap: anywhere;">run_v2_2026-08-14_05-26-22</code></td>
<td align="right">353&nbsp;GB</td>
<td>v2 median-targeted bins, lab appearance. <b>6 trained models</b> point at it.</td>
</tr>
</tbody>
</table>

### 3.1 Contents and counts **[measured]**

| run | per-bin sample count |
|---|---|
| base `run_2026-06-12_04-50-52` | `pix10` `pix20` `pix30` = 10 030 each; `combined/` 10 030; `preview/` 602 |
| `run_labapp_2026-07-06…` | `pix10` `pix20` `pix30` = 10 030 each |
| `run_v2_2026-08-14_05-26-22` | `med03`–`med26` = 10 030 each; **`med30` = 10 013** |

**Two things worth knowing:**

- **`med30` is 17 samples short**, and that is the safety mechanism working as designed: a
  simulation whose achievable median misses the target by more than `[bins.v2].tolerance` (5%) is
  **skipped rather than written under a label it does not have**. That is the BUG-13 fix. Total v2
  samples: **80 223**.
- **The base campaign has 10 030 simulations over a seed range of 1–10 050**, so **20 seeds are
  missing** — presumably failed tasks never retried. Its own `RUN_INFO.txt` says
  `n_simulations : 10000 (seeds 1..10000)`, which does not match either number. **Trust the file
  count, not `RUN_INFO`, for this campaign.** **[measured]**

### 3.2 Base campaign size breakdown **[measured]**

| | size | used by any model? |
|---|---|---|
| `combined/` | 880 GB | not directly — but it is the source for every re-render |
| `pix10/` `pix20/` `pix30/` | 45 GB each, 135 GB | **no** |
| `preview/` | 219 MB | no |
| `metadata/` `logs/` `code/` | ~160 MB | metadata yes |

The 135 GB of clean `pix*` renders are deletable with no loss of capability — they are regenerable
from `combined/` in minutes, and the one clean-trained model used a different, now-deleted dataset.

### 3.3 Deleted, and why **[measured, 2026-09-01]**

Confirmed referenced by **no** model config, note, script or results file before removal:

| run | size | why |
|---|---|---|
| `run_v2_2026-08-14_04-20-11` | 40 MB | failed first attempt — **0 files in all eight bins** |
| `run_v2_2026-08-14_05-13-05` | 18 GB | a 500-seed pilot, superseded 13 minutes later |
| `run_labapp_2026-08-26_22-05-28` | 90 GB | **incomplete** re-render, 6 800 of 10 030 per bin — the run where 34 of 153 tasks died with exit 127 |

**Gone independently:** `run_2026-06-12_18-06-52`, the dataset the first PWC model trained on
(`bin: pix20`). It lived on `/scratch4` and has been purged. **Only the checkpoint survives; that
training set cannot be reconstructed exactly.**

---

## 4. The three datasets used for training

| # | dataset | appearance | when | consumed by |
|---|---|---|---|---|
| 1 | `pix20` from `run_2026-06-12_18-06-52` | **clean** | Jun 2026 | 5 model configs; **dataset purged** |
| 2 | `pix10+pix20+pix30` from `run_labapp_2026-07-06…` | **lab** | Jul 2026 | 27 model configs, incl. the production model |
| 3 | `med03…med30` from `run_v2_2026-08-14_05-26-22` | **lab** | Aug 2026 | 6 model configs |

**[measured]** — `grep -h "^data_dir:" runs/*/*/config.yaml` in `piv-models`.

**Lab appearance entered at stage 2**, commit `5c45e70` (2026-07-04), because clean synthetic images
transferred poorly to the tank. Three bins rather than one because the `pix10`-only model capped
near 15 px against the lab's 24–33 px near-surface peaks (commit `b3d651a`). **[source]**

**Datasets 2 and 3 differ in exactly one thing: the displacement binning.** Both are re-renders of
`run_2026-06-12_04-50-52`, both with `PIV_LAB_APPEARANCE=1`, both 10 030 seeds, both 12 000
particles per image. **[source — both `RUN_INFO.txt` files]** This corrects an earlier claim that
the v1-vs-v2 comparison was confounded by appearance; it is not.

---

## 5. The physics

All from `params.toml` `[physics]`, read at run time. **Nothing here has been varied across any
production campaign.** **[source]**

| setting | value | notes |
|---|---|---|
| grid | 512 × 512, extent 512, doubly periodic | **one grid cell = one pixel**; this identity is load-bearing |
| solver | Oceananigans `NonhydrostaticModel`, 2-D | |
| advection | `WENO(order=5)` | |
| viscosity | `ν = 1e-5` | low, to keep small-scale structure the network can track |
| particles | **16 384** `LagrangianParticles` | = 512·512/16, one per 4×4 pixel block |
| initial condition | random streamfunction: one jet mode + 21 random modes | |
| `jet_amplitude` | 300.0 | **but see below** |
| `m_jet` | 2 | jet direction is also randomised per seed |
| timestep | `dt_cfl = 0.5·dx/max|u|`, save every `10·dt_cfl` | the save interval is the v1 displacement quantum |
| wizard | cfl 0.7, max_change 1.1, max_Δt 2·dt_cfl | stability during the run only |
| `nt` | 40 frames | every production campaign |

### 5.1 The jet amplitude is not 300

`A = jet_amplitude * (1.5 - rand())`, evaluated **after** `Random.seed!`, so the realised amplitude
is uniform on **[0.5, 1.5] × 300 = 150–450** and differs for every seed. **[source]**
`scripts/2DTurbulence.jl:21`. Measured across 20 metadata sidecars: **156.7 to 449.3**.
**[measured]**

This is deliberate — it is a cheap source of flow diversity. **Read the realised value per
simulation from `ic_spec.A_jet_amplitude` in its metadata sidecar; never assume 300.**

### 5.2 Per-frame displacement

`smax ≈ 5 px` — the motion in one saved interval. **[measured]** This number governs everything
about v1 binning (§6.1) and is why sub-5 px displacements were unreachable before v2.

---

## 6. Displacement binning

### 6.1 v1 (`pixNN`) — quantised, and the labels lie

`dp = max(1, floor(pix / smax))`, an **integer** count of saved frames. With `smax ≈ 5 px`,
achievable displacements are multiples of ~5 px, and anything below `smax` is unreachable but was
still written under the requested label. **That is BUG-13.** The bin names are intended
*maxima*, not typical values.

Measured on real v1 data **[measured]**:

| bin | median | p90 | p99 | max |
|---|---|---|---|---|
| `pix10` | 3.12 | 6.69 | 8.93 | 11.47 |
| `pix20` | 8.43 | 13.83 | 17.85 | 22.73 |
| `pix30` | 13.58 | 22.07 | 26.77 | 33.80 |

So **"pix30" really means a median of 13.6 px.** A fresh 5-simulation run on 2026-09-01 reproduced
3.27 / 8.51 / 13.57 — the pipeline still generates what it generated in June. **[measured]**

### 6.2 v2 (`medNN`) — median-targeted, and honest

`datagen_v2/FracFrame.jl` interpolates particle positions to a **fractional** frame index, so any
target is reachable, and the generator solves for the gap that hits a requested **median**. That is
the BUG-14 fix. Self-test: cubic interpolation error **0.00005 px** against exact analytic
trajectories; periodic seam handled; all eight bins hit target in 2 iterations. **[measured, mock
data]**

Measured on the delivered dataset, 40 samples per bin **[measured, 2026-09-01]**:

| bin | achieved median | p90 | p99 | max (pooled) |
|---|---|---|---|---|
| `med03` | 3.01 | 4.36 | 5.20 | 9.15 |
| `med06` | 6.03 | 8.72 | 10.39 | 18.31 |
| `med09` | 9.04 | 13.08 | 15.59 | 27.46 |
| `med12` | 12.05 | 17.44 | 20.78 | 36.61 |
| `med16` | 16.06 | 23.23 | 27.71 | 48.82 |
| `med20` | 20.09 | 29.06 | 34.63 | 61.02 |
| `med26` | 26.11 | 37.78 | 45.02 | 79.32 |
| `med30` | 30.10 | 43.56 | 51.87 | 91.53 |

**Every bin matches its name to within ~0.1 px.** The targeting works.

### 6.3 The max/median ratio is 1.78–1.89, not 1.67

`DATA_REQUIREMENTS.md` R1 claims 1.67 and concludes the bin set spans "max ~5 → 50 px". Re-measured
on the delivered data **[measured]**:

- per-sample ratio: **mean 1.89, sd 0.33, median 1.78, worst 3.03**
- **identical to two decimals in all eight bins**

The near-constancy claim is confirmed exactly, and it is structural: within one simulation every bin
shares frame A and the same particle subset, so a bin's displacement field is the same velocity
field scaled by its time gap — and scaling does not change max/median.

**But the value is ~12% higher than documented**, so `med30` reaches **53 px in a typical sample,
57 px on average, 91.5 px pooled** — not ~50 px. Coverage is better than designed, but `med30` is
also a harder learning problem than was budgeted for, which is a candidate explanation for the
wide-range model underfitting. **[unverified — worth testing]**

---

## 7. Image appearance

Selected by `[imaging.appearance].mode`, drawn **uniformly per image pair** so the model sees a
range of conditions rather than one. **[source]** `src/ImageGenFunc.jl`, `appearance_draw`.

| parameter | lo | span | resulting range |
|---|---|---|---|
| background | 0.52 | 0.14 | ~44–60 / 255 (gray, like the tank) |
| peak (contrast) | 1.55 | 0.60 | ~3.5–5× |
| particle σ | 0.55 | 0.13 | ~1.8–2.7 px diameter |
| noise σ | 0.060 | 0.035 | ~4–7 / 255 |

Clean mode: background 0, peak 1, σ 1.2, noise 0.

**Seeding density**: 12 000 particles rendered per image out of the 16 384 simulated — roughly 0.046
particles per pixel, chosen to match the lab. The CLI flag for this is `--sample`/`-k`, which reads
like "number of samples" and is not.

---

## 8. Design decisions worth not re-litigating

**One image pair per simulation per bin.** So `samples = n_sims × n_bins`. **[design]**

**One anchor time per simulation.** The velocity field's decorrelation time is **τ ≈ 110 saved
frames ≈ 30 time units**, with autocorrelation of `[u;v]` staying **above 0.99 through the first ~5
time units** — the window pairs are drawn from. A run is only 40 frames, so **the entire simulation
sits inside one decorrelation time** and a second pair from a different start frame would be a
near-duplicate flow. Diversity must come from more seeds. **[measured — `notes/DATA_GENERATION_DESIGN.md` §2]**

**More bins ≠ more data.** All bins from one simulation share frame A *and* the same randomly drawn
particle subset; only frame B differs. The number of **independent flows** is `n_sims` whatever the
bin count. More bins buy displacement coverage, which is useful, and nothing else.

**Therefore: split train/validation by SEED, not by sample.** A per-sample split puts the same flow
on both sides and the validation number becomes fiction. `piv-models` does this correctly today —
`common/data.py`, "Deterministic 80/10/10 split over available seeds". **Preserve this.** **[source]**

**Out-of-tolerance samples are skipped, not relabelled.** `[bins.v2].write_out_of_tolerance = false`.
Writing a mislabelled sample was BUG-13 and it silently capped the trained model's usable
displacement range for two months.

---

## 9. Operational facts **[measured, 2026-09-01]**

From a real 10-simulation run on `uri-cpu`:

| | value |
|---|---|
| queue wait | **4 seconds** |
| fluid simulation, per task | **1 min 15 s** |
| total per task | **4–6 minutes** |
| **overhead fraction** | **~70%** — Julia startup, package load, combine, render |
| concurrent tasks in practice | **~5**, despite a `%100` throttle — cluster capacity, not config |
| implied: 100-sim pilot | ~90–100 minutes |

**Because 70% of each task is fixed overhead and `run_array.sbatch` runs exactly one simulation per
task, batching several simulations per task would give roughly a 2× speedup.** Not implemented; the
trade-off is failure isolation. **[design, open]**

**Storage:** `/work` is **1 TB for the entire PI group**; `/project` is **10 TB**. A 10 000-simulation
campaign with `keep_combined = true` is ~9 TB and does not fit on `/work` at all. **[measured]**

**Two `squeue` gotchas**, both of which have cost time: the JOBID column is 18 characters so a
`%100` array throttle displays as `%10`; and one `submit_run.sh` creates **two** jobs — the
simulation array plus a dependent finalize job.

---

## 10. Known gaps and untested claims

- **The Lagrangian-vs-Eulerian label residual has never been measured on production data** — only
  0.13 px on a mock steady field. It should grow with displacement; `med26`/`med30` are the place
  to check.
- **`field_at` interpolates the velocity field linearly in time**, and the self-test's mock field is
  steady, so that path is exact by construction and therefore untested. On real turbulence it will
  carry some error into the `uB/vB` labels. Cubic interpolation for fields was added in `4277a0a`,
  but the error has not been quantified on a real combined file. `datagen_v2/DATA_REQUIREMENTS.md`
  "Known gap".
- **The "ceiling ≈ where training mass falls below 3.4%" rule** in R1 is calibrated on a single
  data point and is a heuristic, not a law.
- **No free surface anywhere in the training data.** It is 2-D periodic turbulence: no air/water
  interface, no masked air region. The lab fails in the top 2 mm and the training data has no
  "top". Requirement R4, never implemented. An earlier test showed filling the masked air region
  changes predictions by +0.03 px [−0.24, +0.35], i.e. nothing — so this is coverage against a
  blind spot, not a known fix.
- **The base campaign's 20 missing seeds** were never investigated.

---

## 11. How to check any of this

```bash
# per-bin displacement statistics and sample images, for any run
python scripts/make_report.py --root <RUN_DIR> --n 8 --format both

# what a run was actually configured with — the authoritative record
cat <RUN_DIR>/RUN_INFO.txt

# what the physics was, per simulation
cat <RUN_DIR>/metadata/<seed>.toml        # ic_spec.A_jet_amplitude, ic_physics.k_p, ...

# what the code will use right now
julia --project=. scripts/params_export.jl

# which datasets any trained model actually used (run in piv-models)
grep -h "^data_dir:" runs/*/*/config.yaml | sort | uniq -c
```

`RUN_INFO.txt` and the metadata sidecars are the authoritative record of an old dataset — trust them
over `params.toml`, which describes what the *next* run will do.
