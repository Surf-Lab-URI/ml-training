# Generating the dataset on Unity (SLURM)

Scales the local pilot up to a large dataset by running many independent
simulations in parallel as a **SLURM array job** — one array task per simulation,
one simulation per independent flow ("sample"), per the one-sim-one-flow design
(`notes/DATA_GENERATION_DESIGN.md` §1–2).

The whole campaign is reproducible from one number: **`seed = BASE_SEED + array-index`**.
Currently configured for **100 sims** (`--array=1-100%50`, seeds 1–100, `nt=40`).

---

## Parallel-safe output (done)

Each task sets `PIV_OUT_DIR` to its own node-local scratch dir, and
`2DTurbulence.jl` / `ImageGen.jl` now honor it (writing under
`$PIV_OUT_DIR/data/{binary,visual}`). So parallel tasks never collide — no
serial throttle or per-task checkout needed.

At the end of each task the sbatch copies the **durable** artifacts into a shared
`dataset/` dir and deletes the scratch:
- **image-pair samples** (`pixN/`) — what you train on (~14 MB/sim)
- **metadata sidecar** (`metadata/`) — the seed + params, so any sim is regenerable (~1.3 KB/sim)

The **raw simulation is discarded** (it's ~13× larger and reconstructible from the
seed — see notes §4). If you expect to re-render later (new displacement bins,
different particle density), uncomment the "keep the combined raw file" block in
the sbatch to retain `_combined.jld2` (~90 MB/sim).

---

## One-time setup on Unity

```bash
git clone git@github.com:Surf-Lab-URI/ml-training.git ~/ml-training
cd ~/ml-training
git checkout dataSimulation
mkdir -p logs dataset
module load julia                      # use the exact module name available on Unity
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

## Edit before submitting

Open `unity/generate_dataset.sbatch` and set the lines marked `TODO`:
- `--array` size and throttle (currently `1-100%50`)
- `--time`, `--mem`, `--partition`, and `--account` (if your allocation needs it)
- `PROJ` (repo path), `BASE_SEED` (currently 0 → seeds 1..100), `NT` (currently 40)
- the `module load julia` version

## Submit / monitor

```bash
sbatch unity/generate_dataset.sbatch
squeue --me                            # watch the array
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # per-task accounting
```

## Output

```
dataset/pix10/seed7_<tag>_pix10.jld2     # image pair A,B (uint8) + labels uA,vA,uB,vB (Float32)
dataset/pix20/seed7_<tag>_pix20.jld2
dataset/pix30/seed7_<tag>_pix30.jld2
dataset/metadata/seed7_metadata_<tag>.toml
```

One image pair per sim per bin. Velocity-field **labels** live inside the pair
files. **Note (BUG-13):** the `pixN` folder names are nominal targets, not exact
displacements — by construction `smax≈5 px`, so actual displacements quantize to
~`{5, 15, 25}` and are inconsistent across sims. The velocity-field labels are
exact regardless; use them, not the folder names, as ground truth. See `bugs.md`.

---

## Sizing notes (from the local pilot)

- ~3–4 min wall per sim on CPU at `nt=40` (Julia startup + ~3 min physics).
- 100 sims as an array with 50 concurrent ≈ a few minutes of wall time (queue permitting).
- Storage: `dataset/` ≈ **14 MB/sim** → ~1.4 GB for 100 (raw, if kept, would be ~18 GB).
- GPU per task is faster per sim but usually wins less for many short jobs than CPU
  concurrency — see the GPU variant block at the bottom of the sbatch.

## Still to do (tracked separately)

- [x] `PIV_OUT_DIR` output-root support in `2DTurbulence.jl` / `ImageGen.jl`.
- [x] Per-sample metadata sidecar (`notes/DATA_GENERATION_DESIGN.md` §6) — in-sim portion.
- [x] One image pair per sim (`ImageGen.jl`).
- [ ] `--pix` as a CLI list (currently `[10,20,30]` hard-coded in `ImageGen.jl`).
- [ ] `--n_particles` / density as a CLI arg (`notes/DATA_GENERATION_DESIGN.md` §8).
- [ ] ImageGen-side metadata (§6-E pix/dp/smax, §6-F rendered ppp) folded into the sidecar.
</content>
