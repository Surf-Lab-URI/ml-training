# PIV Data-Generation Runbook / Handoff

Everything needed to run, monitor, and reason about the synthetic-PIV data-generation
campaign on the Unity HPC cluster. Self-contained — if your machine restarts, start here.

_Last updated: 2026-06-11._

---

## 0. TL;DR — run a campaign in one command

```bash
# on Unity (after one-time setup in §7):
cd /work/pi_nicholas_pizzo_uri_edu/arup_mazumder/ml-training
git pull
./unity/submit_run.sh 1000            # N sims; also: ./unity/submit_run.sh 100000 40 cpu-preempt
```
This makes a timestamped run folder under `/project/.../piv_2dturb_dataset/` containing a code
snapshot, `RUN_INFO.txt`, the samples, metadata, raw sims, a Parquet manifest, and a 2% visual
preview — all automatically.

---

## 1. What the project is

Generate synthetic Particle Image Velocimetry (PIV) **image pairs** with ground-truth
**velocity fields as labels**, to train an ML flow-estimation model. Pure-Julia pipeline:

```
scripts/2DTurbulence.jl   → simulate 2-D turbulence (512², Oceananigans, WENO) + advect particles
src/CombineAndConquer.jl  → merge field + particle output into one *_combined.jld2
scripts/ImageGen.jl       → render image pairs + store velocity-field labels
```
The first script auto-launches the other two. Branch: **`dataSimulation`** (repo is public:
`github.com/Surf-Lab-URI/ml-training`).

---

## 2. Key paths & identifiers

| Thing | Value |
|---|---|
| Unity username | `arup_mazumder_uri_edu` |
| SSH | `ssh arup_mazumder_uri_edu@unity.rc.umass.edu` (key `~/.ssh/id_ed25519`, already registered) |
| Repo on Unity | `/work/pi_nicholas_pizzo_uri_edu/arup_mazumder/ml-training` |
| Julia depot | `/work/pi_nicholas_pizzo_uri_edu/arup_mazumder/julia_depot` |
| **Dataset root** | `/project/pi_nicholas_pizzo_uri_edu/arup/piv_2dturb_dataset` (11 TB) |
| Python venv (finalize) | `/work/pi_nicholas_pizzo_uri_edu/arup_mazumder/piv_venv` |
| Mac repo | `~/Desktop/uri/summerRA/data_simulation/ml-training` |
| SLURM partition | **`uri-cpu`** (URI nodes, ~no queue; the shared `cpu` is congested) |

**Required env (the job scripts set these; set them too for interactive Julia):**
```bash
export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH=/work/pi_nicholas_pizzo_uri_edu/arup_mazumder/julia_depot
export JULIA_CPU_TARGET=generic
```

---

## 3. Connecting to Unity

- **SSH (terminal):** `ssh arup_mazumder_uri_edu@unity.rc.umass.edu`
- **Web shell:** https://ood.unity.rc.umass.edu → Clusters → Unity Shell Access
- Add/check SSH keys at https://unity.rc.umass.edu/panel/account.php → "SSH Keys".
- ⚠️ The **web shell disconnects** on idle and kills interactive/`tmux` sessions — but **NOT
  `sbatch` jobs** (they run on compute nodes regardless). Use SSH from the Mac to avoid this.

---

## 4. Running a campaign (the workflow)

`./unity/submit_run.sh [N_SIMS] [NT] [PARTITION] [BASE_SEED]` does it all:
1. makes `run_<timestamp>/` under the dataset root,
2. snapshots `code/` (scripts, src, Project/Manifest, git SHA),
3. writes `RUN_INFO.txt` (at-a-glance summary),
4. submits the per-sim SLURM array (`run_array.sbatch`, one task = one seed),
5. auto-submits `finalize_run.sbatch` (`--dependency=afterany`) → builds the Parquet
   manifest + renders the 2% preview once sims finish.

Defaults: `N=1000`, `NT=40`, `PARTITION=uri-cpu`, `BASE_SEED=0` (→ seeds 1..N).
Timing: ~6 min/task; 1000 sims at 100-concurrent ≈ **~1 hour**.

> Older/simple path (no run folder): `sbatch --partition=uri-cpu --time=00:40:00 --array=1-1000%100 unity/run100.sbatch`

---

## 5. Output layout & file contents

```
piv_2dturb_dataset/
└── run_2026-06-11_22-45-03/
    ├── RUN_INFO.txt        # seeds, nt, pix bins, git, IC defaults, partition
    ├── code/               # exact scripts + Project/Manifest + git_info.txt for this run
    ├── pix10/  pix20/  pix30/   # image pairs + labels: seed<N>_pixB.jld2
    ├── metadata/           # per-sim sidecars: seed<N>.toml
    ├── combined/           # raw sim: seed<N>.jld2 (kept → re-render without re-simulating)
    ├── preview/            # ~2% random pairs as 2x2 PNGs (Image A|B, speed @A|@B)
    ├── manifest.parquet    # one row/sim, ~48 cols (built by finalize)
    └── logs/
```

**`pixN/seed<N>_pixB.jld2`** (the training sample):
- `pairs/000001/A`, `/B` — the two particle images (512×512 uint8)
- `pairs/000001/fields/uA,vA,uB,vB` — velocity fields = the labels (512×512 float32)

**`metadata/seed<N>.toml`** sections: `[reproducibility]` (seed, args, git, julia, grid/ν),
`[ic_spec]` (jet amp/phase, modes), `[ic_physics]` & `[sampling_final]` (U_max, energy,
enstrophy, k_p, C_achieved), `[displacement]` (smax, dp_pix*, realized_px_pix*), `[particles]`.

**`combined/seed<N>.jld2`** = full sim: `fields/timeseries/{u,v,t}/<frame>` (~41 frames) +
`particles/timeseries/particles/<frame>` (~16384 tracers).

---

## 6. Design decisions (the "why")

- **One sample per simulation.** Decorrelation is long (τ≈110 frames) → extra frames are
  near-duplicates. Diversity comes from MANY independent sims (varied seed), not many frames.
- **Shared-FIRST frame (PIV convention).** All 3 bins share the same first image A; the second
  image B is `+dp` frames later (B = A + dp). A is anchored so pix30's B lands on the final
  (most-developed) frame. Same tracked particles across bins → identical A.
- **"Most-developed":** the IC is artificially smooth (truncated spectrum); the 2-D enstrophy
  cascade fills small scales over time, so later frames are more realistic turbulence.
- **Displacement quantization (BUG-13):** dp = floor(pix/smax), smax≈5 px by construction, on a
  knife-edge → realized displacements land at ~{5,15,25} or {10,20,30} per sim. The velocity-field
  **labels are exact regardless**; only the bin *names* are nominal. The `[displacement]` metadata
  records the TRUE displacement so you can filter by it.
- **Storage:** keep samples + metadata + combined (on 11 TB /project). At 100k scale, drop
  `combined/` (it's regenerable from the seed); ~1.4 TB samples vs ~18 TB raw.
- **`seed` is the universal key:** seed → `pixB/seed<N>_pixB.jld2`, `metadata/seed<N>.toml`, etc.

---

## 7. One-time setup (already done, or do once)

**Julia (done):** Unity's `julia/1.10.5` module has a fatal extension-loader bug — DO NOT use it.
Instead juliaup-installed **Julia 1.12.6** lives in `~/.juliaup`:
```bash
curl -fsSL https://install.julialang.org | sh -s -- -y --default-channel 1.12.6
export PATH="$HOME/.juliaup/bin:$PATH"
# build the PORTABLE package cache once (so every compute node reuses it, no recompile):
export JULIA_DEPOT_PATH=/work/pi_nicholas_pizzo_uri_edu/arup_mazumder/julia_depot
export JULIA_CPU_TARGET=generic
julia --project=. -e 'using Pkg; Pkg.precompile()'
```

**Python venv (pending — for the finalize job):**
```bash
module avail python                  # find the module name (or system python3 may work)
module load python/3.11              # exact name TBD
python -m venv /work/pi_nicholas_pizzo_uri_edu/arup_mazumder/piv_venv
source /work/pi_nicholas_pizzo_uri_edu/arup_mazumder/piv_venv/bin/activate
pip install pandas pyarrow tomli h5py matplotlib numpy
```

---

## 8. Monitoring a run

```bash
squeue --me                                  # running / pending tasks
squeue --me --start                          # estimated start for pending
sacct -X --starttime today --format=JobID%16,State,Elapsed | tail
RUN=/project/pi_nicholas_pizzo_uri_edu/arup/piv_2dturb_dataset/run_<stamp>
for d in pix10 pix20 pix30 metadata combined; do echo "$d: $(ls $RUN/$d 2>/dev/null | wc -l)"; done
cat $RUN/logs/finalize_*.out                 # manifest + preview progress
```

---

## 9. After a run: filter & view

Manifest is auto-built into `run_<stamp>/manifest.parquet`. Filtering → files:
```python
import pandas as pd, h5py
df = pd.read_parquet("manifest.parquet")
sel = df[(df.k_p_ic > 0.04) & (df.realized_px_pix20 > 18)]   # any condition incl. TRUE displacement
for _, r in sel.iterrows():
    f = h5py.File(r.sample_pix20, "r")        # "pix20/seed<N>_pix20.jld2"
    A = f["pairs/000001/A"][:]; uA = f["pairs/000001/fields/uA"][:]   # image + label
```
Visual spot-check: open PNGs in `run_<stamp>/preview/` (2×2: Image A|B top, speed @A|@B bottom).
Manual manifest/preview if needed:
```bash
python scripts/build_manifest.py --in <RUN>/metadata --out <RUN>/manifest.parquet
python scripts/render_preview.py --root <RUN> --frac 0.02
```

---

## 10. Gotchas & fixes (learned the hard way)

- **`uri-cpu` partition**, not `cpu` — the shared `cpu` queue can be hours; `uri-cpu` is instant.
- **`JULIA_CPU_TARGET=generic`** is REQUIRED — without it Julia precompiles per-node "native"
  code and every task recompiles (~35 min). Generic = portable cache, ~6 min/task.
- **No `using CUDA`** — it's commented out (CPU run); it crashed `CUDA_Runtime_jll` on some CPU
  nodes and bloated precompile. Re-enable only for GPU runs.
- **`.gitignore` has `*.sh`** → shell scripts are ignored. `submit_run.sh` is force-tracked via a
  `!unity/submit_run.sh` negation. (`.gitignore` has NO trailing-comment support — own line only.)
- **Web shell / long pastes mangle** multi-line content → use `git` to move files, not paste.
- **Mac local Julia:** juliaup self-update can hang the `julia` shim; call the real binary:
  `~/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/Julia-1.12.app/Contents/Resources/julia/bin/julia`
- **Git flow:** edit on Mac → `git add … && git commit && git push` → on Unity `git pull`.
  (Unity pulls from GitHub; it can't see Mac files.) Always `commit` before `push`.

---

## 11. Pending / next steps

- [ ] Push the per-run workflow files (submit_run.sh, run_array.sbatch, finalize_run.sbatch,
      render_preview.py, .gitignore) from the Mac; `git pull` on Unity.
- [ ] Create the Python `piv_venv` on Unity (§7) so finalize can build manifest + preview.
- [ ] (optional) Tuck the current direct-flow 1000 output into a `run_first_1000/` folder.
- [ ] Calibrate sampling time via an E(k)-saturation study → replace provisional `nt=40`.
- [ ] At 100k scale: drop `combined/` collection (regenerable) to save storage.

---

## 12. Quick command reference

```bash
# connect
ssh arup_mazumder_uri_edu@unity.rc.umass.edu

# update code on Unity
cd /work/pi_nicholas_pizzo_uri_edu/arup_mazumder/ml-training && git pull

# run a campaign (self-documenting run folder)
./unity/submit_run.sh 1000

# monitor
squeue --me ; sacct -X --starttime today --format=JobID%16,State,Elapsed | tail

# build manifest / preview manually
python scripts/build_manifest.py --in <RUN>/metadata --out <RUN>/manifest.parquet
python scripts/render_preview.py --root <RUN> --frac 0.02
```
