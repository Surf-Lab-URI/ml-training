# ml-training

**Generates synthetic training data for deep-learning PIV: particle image pairs, plus the flow
field that labels them.** 2-D turbulence with Lagrangian particles, rendered into image pairs at
chosen displacements. Entirely Julia; runs on the URI/UMass Unity cluster.

**Every setting lives in [`params.toml`](params.toml).** That file is commented and is the only
place you should need to edit. If you find yourself changing a number inside a `.jl` or `.sh` file,
stop — the number belongs in `params.toml`, and if it isn't there yet, add it there.

---

# Quick start

Everything runs from one shared directory — you do not need your own copy of anything except Julia.
Each step is explained in full in the **Detailed guide** below; if a command fails, look there.

**1. Connect and go to the shared checkout.**

```bash
ssh YOUR_USERNAME@unity.rc.umass.edu
cd /project/pi_nicholas_pizzo_uri_edu/arup/ml-training
```

**2. Install Julia** — the only per-person step, because home directories on Unity are private and
you cannot use anyone else's install. Once, then it is done forever:

```bash
# Add the two settings, but only once — running this twice would duplicate them.
grep -q juliaup ~/.bashrc || cat >> ~/.bashrc <<'EOF'
export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH=/work/pi_nicholas_pizzo_uri_edu/arup_mazumder/julia_depot
EOF
source ~/.bashrc

# Install only if you do not already have Julia. Re-running the installer on an existing
# ~/.juliaup fails with "that folder already exists", which is confusing but harmless.
julia --version || curl -fsSL https://install.julialang.org | sh
```

You want **1.12.6**, which is what `Manifest.toml` was built with.

The second line points at the group's shared, already-populated package depot, so there is nothing
to install and no `Pkg.instantiate()` to wait through.

**3. Check the configuration resolves.** Nothing needs editing — `params.toml` already points at
the shared dataset directory, and each campaign writes its own timestamped folder there, so several
people can run at once.

```bash
julia --project=. scripts/params_export.jl
```

**4. Generate a pilot.** Start at 100 simulations, always, whatever you eventually want.

```bash
./unity/submit_run.sh 100
squeue --me                                        # watch it
```

It prints the run folder it created, under
`/project/pi_nicholas_pizzo_uri_edu/arup/piv_2dturb_dataset/`.

**5. Look at what you made.**

```bash
module load python/3.11.7
source /work/pi_nicholas_pizzo_uri_edu/arup_mazumder/piv-venv/bin/activate
RUN=/project/pi_nicholas_pizzo_uri_edu/arup/piv_2dturb_dataset/run_2026-09-01_21-12-58   # yours
python scripts/make_report.py --root "$RUN"
```

Copy the resulting `report.pdf` to your laptop and open it. If the images have particles in them and
the displacements match what you asked for, the pipeline is working — scale up by setting
`[run].n_sims` and using `unity/submit_chunked.sh`.

**Want different settings?** Do not edit the shared `params.toml`; everyone reads it. Copy it and
point at your copy instead — every script honours this:

```bash
cp params.toml "${USER}_params.toml"           # already gitignored
export PIV_PARAMS="$PWD/${USER}_params.toml"   # add to ~/.bashrc to make it stick
```

Name it after yourself, not `my_params.toml` — the checkout is shared, so a generic name is one
everyone else would collide with.

---

# Detailed guide

## How it works

```
scripts/2DTurbulence.jl          1. SIMULATE   2-D turbulence + particles, one run per random seed
                                               writes  combined/seed<N>.jld2
datagen_v2/ImageGenV2.jl         2. RENDER     saved frames -> image pairs at chosen displacements
                                               writes  med03/ med06/ ... one dir per bin
scripts/make_report.py           3. CHECK      a PDF of what you generated. Always look at it.
```

**Stage 2 re-runs no physics.** It reads frames stage 1 already saved, so an existing campaign can
be re-rendered with different displacements or a different image appearance in minutes instead of
thousands of core-hours. Both datasets currently in use were made this way, from one June
simulation campaign. This is why `[run].keep_combined` matters: discard the simulation output and
that campaign can never be re-rendered again.

`scripts/ImageGen.jl` is the **legacy** renderer, kept only so the 2026-06 and 2026-07 datasets stay
reproducible. Its displacement targeting is quantised and its bin labels do not mean what they say.
Use `datagen_v2/` for anything new.

---

## Running it on Unity, in full

Follow these in order. Steps 1–5 are once per person; steps 6 onward are how you generate data.

### 1. Connect

```bash
ssh YOUR_USERNAME@unity.rc.umass.edu
```

### 2. Get the code

**Use the shared checkout.** It lives in the group's project space, is group-writable, and is what
`params.toml` already points at:

```bash
cd /project/pi_nicholas_pizzo_uri_edu/arup/ml-training
```

A common misconception: `/project/pi_nicholas_pizzo_uri_edu/arup` also holds the *datasets*, and
each run folder there contains a `code/` directory — but that is a four-file provenance snapshot of
the renderer only, with no `Project.toml` and no `2DTurbulence.jl`. It records what produced that
dataset; you cannot run a campaign from it. The repository above is the runnable copy.

**The branch is `dataSimulation`.** `main` is the lab's April 2026 baseline and contains none of the
data-generation pipeline. The shared checkout is already on the right branch; keep it there.

<details>
<summary><b>Making your own clone instead</b> — if you intend to modify the code rather than run it</summary>

```bash
cd /project/pi_nicholas_pizzo_uri_edu/arup      # or anywhere you have space
git clone git@github.com:Surf-Lab-URI/ml-training.git my-ml-training
cd my-ml-training && git checkout dataSimulation
```

Then set `[unity].project_dir` in your own `params.toml` to that path, since the Slurm jobs run
from whatever it names. Note that `/work` holds only **1 TB for the entire group** while `/project`
holds 10 TB, so prefer `/project` for anything large.
</details>

### 3. Make Julia available

**This step cannot be shared, even inside the group.** Home directories on Unity are `drwx------`,
so nobody can reach anyone else's `~/.juliaup` regardless of its own permissions. Every person
needs their own Julia.

Install [juliaup](https://julialang.org/downloads/) once:

```bash
curl -fsSL https://install.julialang.org | sh
```

Then add it to your PATH — put this line in your `~/.bashrc` so every login and every job has it:

```bash
export PATH="$HOME/.juliaup/bin:$PATH"
```

Confirm you get **1.12.6**, which is what `Manifest.toml` was built with:

```bash
julia --version        # julia version 1.12.6
```

> Do **not** use `module load julia` — the module tree only offers 1.10.5, which does not match the
> manifest.

### 4. Get the packages

**Group members can skip the install entirely.** A populated 4.8 GB Julia depot and a Python
environment already exist on `/work` and are group-readable, so point at them and move on:

```bash
export JULIA_DEPOT_PATH=/work/pi_nicholas_pizzo_uri_edu/arup_mazumder/julia_depot
module load python/3.11.7
source /work/pi_nicholas_pizzo_uri_edu/arup_mazumder/piv-venv/bin/activate
```

Put the `export` in your `~/.bashrc`; the other two are needed only in a shell where you want to
run the report. If two people instantiate into the shared depot at the same time they can race, so
if you ever do need to add a package, do it when nobody else is mid-install.

<details>
<summary><b>Installing from scratch instead</b> — for a new allocation, or outside this group</summary>

```bash
export JULIA_DEPOT_PATH=/work/pi_nicholas_pizzo_uri_edu/$USER/julia_depot
julia --project=. -e 'using Pkg; Pkg.instantiate()'      # 10-20 minutes
```

For Python, **Unity blocks `pip install --user`** (PEP 668, "externally-managed-environment"), so
build a virtual environment:

```bash
module load python/3.11.7
python3 -m venv /work/pi_nicholas_pizzo_uri_edu/$USER/piv-venv
source /work/pi_nicholas_pizzo_uri_edu/$USER/piv-venv/bin/activate
pip install numpy h5py matplotlib pandas pyarrow
```

Reactivate it (`module load` + `source`) in each shell where you run the report.
</details>

### 5. Check `params.toml`

**Nothing needs editing by default.** The shared file already points at the group's dataset
directory, the shared checkout and the shared Julia depot, and every campaign writes its own
timestamped folder, so several people can run at once without colliding.

Confirm what a job will actually use:

```bash
julia --project=. scripts/params_export.jl
```

The settings you are most likely to *want* to change are near the top of the file: `[run].n_sims`
(how many simulations), `[run].nt` (frames per simulation), `[run].keep_combined` (whether the raw
simulations are kept — read the warning there), and `[bins.v2].medians` (which displacements to
render). Change them in a personal copy rather than in the shared file — see below.

The three paths at the bottom, under `[unity]`, only need attention if you are working outside this
group or from your own clone:

| setting | when to change it |
|---|---|
| `[run].output_root` | you want datasets somewhere other than the group's shared directory |
| `[unity].project_dir` | you made your own clone (step 2) |
| `[unity].julia_depot` | you built your own depot instead of using the shared one |

#### Using your own settings in a shared checkout

The group's convention on Unity is **one directory per person** under
`/work/pi_nicholas_pizzo_uri_edu/` — there is already an `Andrew_Goering`, an `arup_mazumder` and a
`Xiaoyi_Zhao`. Your own directory and your own clone is the simplest setup, and steps 2–4 above say
what can be shared if you would rather not duplicate the 4.8 GB depot.

If you do share a checkout, **do not edit its `params.toml`** — it is a tracked file, so your edits
become everyone's local changes and collide the moment two people want different settings. Keep a
personal copy instead:

```bash
cp params.toml "${USER}_params.toml"           # already gitignored
# edit ${USER}_params.toml
export PIV_PARAMS="$PWD/${USER}_params.toml"   # add to ~/.bashrc to make it stick
```

The generators, `params_export.jl` and the Slurm submitters all read `PIV_PARAMS` when it is set
and fall back to `params.toml` otherwise, so a personal copy works everywhere with no change to any
command. In your own checkout you can ignore this and just edit `params.toml`.

### 6. Run a pilot — always

A hundred simulations take a few minutes and catch every configuration mistake that a hundred
thousand would catch three days later.

```bash
./unity/submit_run.sh 100
```

This creates a timestamped folder under `[run].output_root`, snapshots the code it used, writes a
`RUN_INFO.txt` recording exactly what was configured, and launches the Slurm array. Watch it:

```bash
squeue --me
```

### 7. Look at what came out

```bash
python scripts/make_report.py --root "$RUN"
```

Copy `report.pdf` to your laptop and open it. **Check three things before going any further:**
the particle images actually have particles in them; the displacement magnitudes are in the range
you asked for; and the flow-field arrows point somewhere sensible.

### 8. Run the full campaign

Set `[run].n_sims` in `params.toml` to the size you want, then:

```bash
nohup ./unity/submit_chunked.sh > chunked.log 2>&1 &
```

Use `submit_chunked.sh`, not `submit_run.sh`, for anything above a couple of thousand: Unity caps
queued jobs per user at about 2000, and this driver submits in chunks and waits so it never trips
the limit. Run it detached with `nohup` — it stays alive for the whole campaign.

**How big is big?**

```
total samples = n_sims  x  number of bins in [bins.v2].medians
```

so 10 000 simulations with the default eight bins gives 80 000 samples. Storage is roughly 900 MB
per simulation when `keep_combined = true` — about **9 TB for 10 000 simulations**. If you do not
have that, set `keep_combined = false`, but read the warning in `params.toml` first: it cannot be
undone.

### 9. Re-render instead, when you can

If a campaign was run with `keep_combined = true`, you can build a completely new dataset from it
in minutes — different displacement bins, different image appearance — with no new physics:

```bash
# 1. edit [bins.v2].medians and/or [imaging.appearance] in params.toml
# 2. then:
RUN=/project/pi_nicholas_pizzo_uri_edu/arup/piv_2dturb_dataset/run_2026-06-12_04-50-52
./unity/submit_v2.sh "$RUN"
```

This is by far the cheapest thing in the pipeline. Reach for it before starting a new campaign.

---

## Running a few simulations by hand

For debugging, or to look at a single case. This works the same on your laptop and inside an
interactive Unity session:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=. scripts/2DTurbulence.jl --nt 40 --seed 42
python scripts/make_report.py --root data
```

For a handful in a row, `scripts/run_batch.sh` runs them serially, taking its defaults from
`params.toml` like everything else:

```bash
# on Unity, grab an interactive node first:
srun --partition=uri-cpu --time=02:00:00 --mem=8G --cpus-per-task=4 --pty bash
./scripts/run_batch.sh 5 40          # 5 simulations, 40 frames each
```

**It is serial**, so it is for looking at a few cases, not for building a dataset — a thousand
simulations would take days. Use `unity/submit_run.sh` or `unity/submit_chunked.sh` for that.

Simulate now, render later:

```bash
julia --project=. scripts/2DTurbulence.jl --nt 40 --seed 42 --no_image_gen
VARS=$(basename out/*_combined.jld2 _combined.jld2)
julia --project=. datagen_v2/ImageGenV2.jl -f "out/${VARS}_combined.jld2" -v "$VARS"
```

Command-line flags override `params.toml`; anything you leave off comes from the file. Pass `-p` to
also write PNGs — fine for a handful of pairs, far too slow for a campaign.

---

## The three datasets used for training

All three trace back to the June 2026 simulations; only the rendering changed. **No new physics has
been generated since 2026-06-12.**

**1. `pix20` — one displacement bin, clean synthetic (June 2026).** 10 000 simulations, rendered at
a single nominal displacement (`pix20`, median 8.4 px) with noiseless black-background images; the
first trained model used this bin alone. Its dataset lived on `/scratch` and has since been purged —
only the checkpoint survives.

**2. `pix10 / pix20 / pix30` — three bins, lab-matched appearance (July 2026).** A re-render of
campaign `run_2026-06-12_04-50-52` into three displacement bins, and where the lab appearance was
introduced (`PIV_LAB_APPEARANCE=1`, commit `5c45e70`): clean synthetic images transferred poorly to
the tank, so the renderer gained a domain-randomised gray background, reduced contrast, ~2 px
particles and sensor noise matched to `ExpLCL_1_03`. Three bins rather than one because the
`pix10`-only model capped near 15 px against the lab's 24–33 px near-surface peaks. This produced
the production model.

**3. `med03 … med30` — eight median-targeted bins, lab appearance (August 2026).** The same June
campaign re-rendered once more, keeping the lab appearance but binning by **median** displacement
via fractional frame gaps (the BUG-13/14 fix), because the v1 bins were quantised in ~5 px steps and
named by their maximum rather than their typical value. It was built to push past the 22 px
displacement ceiling measured on the production model, and it does reach far wider displacements —
though the model trained on the full range underfit, so this dataset has not yet superseded #2.

## What comes out

```
run_<stamp>/
  RUN_INFO.txt        what this run was configured with — the authoritative record
  combined/           raw simulations (only if keep_combined = true)
  med03/ med06/ ...   one directory per displacement bin, one .jld2 per simulation
  metadata/           one TOML sidecar per simulation: seed, physics, achieved displacements
  manifest.parquet    every sidecar flattened into one table, for filtering
  code/               a snapshot of the generator that produced this run
  logs/               Slurm output
```

Each `.jld2` holds one image pair:

```
pairs/000001/A              512x512 uint8    first frame
pairs/000001/B              512x512 uint8    second frame
pairs/000001/fields/uA,vA   512x512 float32  displacement at A, IN PIXELS
pairs/000001/fields/uB,vB   512x512 float32  displacement at B, in pixels
```

The fields are already displacements in pixels — velocity times the pair's time gap — so
`sqrt(uA² + vA²)` is what a PIV algorithm should recover, with no unit conversion anywhere.

**Split train/validation by SEED, not by sample.** Every bin from one simulation shares a flow
field and a first image, so a per-sample split puts the same flow on both sides and the validation
number becomes meaningless. `piv-models` does this correctly today (`common/data.py`); preserve
that if you write a new loader.

---

## Checking a dataset

```bash
python scripts/make_report.py --root "$RUN" --n 8 --format both
```

Produces `report.pdf` (and with `--format both`, `report.md` plus PNGs) containing:

- **inventory** — bins, sample counts, and the `RUN_INFO.txt` and `params.toml` the run used;
- **displacement distributions per bin**, with median / p90 / p99 / max. This is the page that
  tells you whether the bin labels are honest: a `med20` directory should have a median near 20 px;
- **simulation metadata** — what was constant across the campaign and what varied;
- **random samples** — frame A, frame B, a red/green overlay, a zoom, a displacement-magnitude map,
  and the flow field as arrows drawn 1:1 in pixels.

Flags: `--n` how many samples, `--seed` which ones (reproducible), `--stat-files` how many files
per bin to pool for statistics, `--out` where to write.

---

## When something goes wrong

| symptom | cause and fix |
|---|---|
| `julia: command not found` at the prompt | Step 3 — juliaup is not on your PATH. |
| `julia: command not found`, exit 127, in *some* array tasks only | `$HOME` was not resolvable when the task started. The job scripts already retry five times; if it persists, move your Julia install off `$HOME`. |
| `set run.output_root in params.toml` | You skipped step 5. |
| A bin directory has far fewer files than the others | Expected up to a point — a simulation that cannot reach a bin's target is skipped for that bin rather than mislabelled. A large shortfall means the target is out of reach: widen `[bins.v2].tolerance` or raise `[run].nt`. |
| Tasks killed at the wall clock | Raise `[unity].time_limit`, or lower `[unity].chunk` so each task does less. |
| Jobs rejected, `QOSMaxSubmitJobPerUserLimit` | Too many queued. Use `unity/submit_chunked.sh`. |
| Images look empty, or particles sit outside the frame | `[imaging].width/height/xlim/ylim` no longer match `[physics].grid_n/grid_m`. They must agree — this was BUG-15. |

---

## Things it is easy to get wrong

**More bins is not more data.** Every bin from one simulation shares the same frame A and the same
particles; only frame B differs. More bins gives wider displacement coverage, which is useful, but
the number of *independent flows* is `n_sims` however many bins you configure.

**More pairs per simulation is not offered, deliberately.** The velocity field's decorrelation time
was measured at about 110 saved frames while a simulation is only 40 frames long — the whole run
sits inside one decorrelation time, so a second pair from a different start frame would be a
near-duplicate. Diversity has to come from more seeds.

**Image appearance matters more than you would guess.** A model trained on clean synthetic images
loses most of its accuracy on real laboratory footage. `[imaging.appearance].mode = "lab"` is not
cosmetic; leave it on unless you are running a deliberate ablation.

**The realised jet amplitude is not `jet_amplitude`.** The code computes
`A = jet_amplitude * (1.5 - rand())` after seeding, so it is uniform on 150–450 for the default of
300, and differs for every simulation. Read the actual value per simulation from its metadata
sidecar, never from `params.toml`.

**`--sample` / `-k` is particles per image, not number of samples.** An unfortunate name kept for
compatibility. It is `[imaging].particles_per_image`.

---

## Layout

| path | what it is |
|---|---|
| `params.toml` | every setting, commented. The place to make changes. |
| `scripts/` | entry points: simulation, legacy renderer, report, manifest, diagnostics |
| `src/` | shared code: argument handling, parameter reader, image generation, combine |
| `datagen_v2/` | the current renderer and its design document |
| `unity/` | Slurm submitters and job scripts |
| `notes/` | design notes and the operational runbook |
| `bugs.md` | numbered bug log |

Further reading, in the order it is usually needed:

| file | what it holds |
|---|---|
| `datagen_v2/DATA_REQUIREMENTS.md` | why the v2 bins are what they are, measured against the lab data |
| `notes/DATA_GENERATION_DESIGN.md` | the one-sample-per-simulation decision and the decorrelation measurement |
| `notes/RUNBOOK.md` | operational notes from the original campaigns |
| `unity/README.md` | Slurm specifics |
| `bugs.md` | BUG-13 and BUG-14 explain why `datagen_v2/` replaced `scripts/ImageGen.jl` |
