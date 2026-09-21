# CLAUDE.md — `ml-training`

**What:** the data generator. Pure-Julia 2-D turbulence simulation → synthetic PIV image pairs
with exact ground-truth displacement.
**Produces:** training datasets for `piv-models` (a sibling repo) — image pairs plus labels.
**State:** v1 and v2 datasets generated and in use. v3 (free surface + shear) generates correctly
and trains to a collapsed model, cause unknown — **do not re-run it without a new hypothesis.**

The name is misleading: **nothing is trained here.** Training and evaluation live in
`../../piv-models`.

---

## The pipeline

```
scripts/2DTurbulence.jl      1. SIMULATE  2-D turbulence + particles, one run per seed
                                          -> combined/seed<N>.jld2
datagen_v2/ImageGenV2.jl     2. RENDER    saved frames -> image pairs at chosen displacements
                                          -> med03/ med06/ ... one directory per bin
scripts/make_report.py       3. CHECK     a PDF of what came out. Always look at it.
```

**Stage 2 re-runs no physics.** It reads frames stage 1 already saved, so a campaign can be
re-rendered at different displacements or a different appearance in minutes rather than thousands
of core-hours. Both datasets in use came from one June simulation campaign this way.

**Therefore `[run].keep_combined` is the most consequential setting in `params.toml`.** Discard
the simulation output and that campaign can never be re-rendered again.

`scripts/ImageGen.jl` is the **legacy** renderer, kept only so the 2026-06 and 2026-07 datasets
stay reproducible. Its displacement targeting is quantised and its bin labels do not mean what
they say. Use `datagen_v2/` for anything new.

---

## The traps

### 1. v3 is a dead end, and the note says so

Free surface + near-surface shear. Generates correctly, passes its own gate, and trains to a
**spatially flat model** at or below the predict-zero baseline, reproducible across three seeds
to three decimals. Four hypotheses eliminated — sub-pixel displacement, non-zero `lambda_div`,
loss normalisation, row-flip augmentation.

`notes/DATA_GENERATION_DESIGN.md` says **"Do not run a fifth arm."** The next step, if v3 is ever
picked up, is to verify the dataset end to end — warp frame A by its own label and confirm it
lands on frame B — and that check has to be *designed* first, because the obvious version fails
on v2 data that trains perfectly well.

### 2. `bugs.md` — read the checklist table, not the entries

**The `## ✅ Status Checklist` table is authoritative.** Most detail blocks do not restate their
status, so scanning the `### BUG-n` sections makes fixed bugs look open. **13 of 15 are fixed.**

Two are open:

- **BUG-6** — jet wavenumber can be zero or negative (`2DTurbulence.jl`)
- **BUG-7** — `CombineAndConquer.jl` loads everything into RAM

Three fixed ones are still worth knowing, because they changed what the data means:

- **BUG-13 / BUG-14 — fixed 2026-08-13 in `datagen_v2/`** (`fc211e7`, `FracFrame.jl`). The v2
  generator solves for a **fractional frame index**, so any target displacement is reachable and
  bins are labelled by **measured median**, not a nominal target. Every v2 bin's median matches
  its name to ~0.1 px. **The legacy `scripts/ImageGen.jl` still has the old behaviour** — one
  more reason not to use it for anything new.
- **BUG-15 — fixed.** `xlim`/`ylim` were `(0, 2π)` against a `(0, 512)` sim grid, so particles
  teleported ~127 px per frame and the images carried no signal about the flow. **Any
  `data/visual/pix*/` dataset generated before that fix is invalid.**

### 3. The measurements that shaped the design

- **`smax ≈ 5 px`** was a *displacement floor*: the motion in one saved interval, below which a
  request could not be honoured. **BUG-14's fix removed it** — fractional-frame interpolation
  makes any `pix` reachable from an existing campaign. Historical context for why the bins look
  the way they do, not a current limit.
- **Decorrelation `τ ≈ 110 frames`** — one simulation yields one useful flow, hence *many short
  sims* rather than few long ones. A physical measurement; still true.

---

## Conventions

- **Julia project:** `julia --project=.` from the repo root. `Project.toml` / `Manifest.toml` are
  committed; do not update them casually.
- **`params.toml` is the single source of truth** for a run. `scripts/params_export.jl` exports it
  to shell variables so the Slurm scripts cannot drift from it — a bug that has already bitten:
  a hardcoded bin list in a submitter would have silently produced a two-bin-short dataset.
- **Derive, never duplicate.** Any list that appears in both `params.toml` and a `.sbatch` is a
  future silent failure.
- **Always run a pilot before a campaign**, and always open the report PDF. See README §6–7.
- Generated `.jld2`/`.mat`/`.tif` never enter the repo — see `.gitignore`.

---

## Where things are

| | |
|---|---|
| `README.md` | the real guide, including the full Unity walkthrough |
| `params.toml` | every parameter for a run |
| `notes/DATA_GENERATION_DESIGN.md` | why the design is what it is, and the v3 negative result |
| `notes/DATA_GENERATION_REFERENCE.md` | reference for the generator |
| `bugs.md` | 15 tracked issues; BUG-13/14 affect label correctness |
| `datagen_v2/` | the current renderer |
| `datagen_v3/` | free-surface renderer — see the trap above |
| `unity/` | Slurm scripts for the cluster |
| `scripts/` | simulation, measurement and reporting tools |

---

## Related

**`../../piv-models`** consumes what this repo produces and has its own `CLAUDE.md`. If a
question is about model accuracy, seed variance or the lab tank, it belongs there. If it is about
how the images were made, it belongs here.
