#!/bin/bash
# =============================================================================
# Unity (SLURM) array job — generate the PIV training dataset.
#
# Model: ONE array task = ONE independent simulation = ONE flow ("sample"),
# following the one-sim-one-flow design (notes/DATA_GENERATION_DESIGN.md §1-2).
# The whole campaign is reproducible: seed = BASE_SEED + array-index.
#
# Each task writes to its OWN node-local scratch (PIV_OUT_DIR) — now honored by
# 2DTurbulence.jl / ImageGen.jl — then copies the small, durable artifacts
# (image-pair samples + metadata sidecar) into the shared dataset/ dir and
# discards the large raw simulation. So parallel tasks never collide and the
# 18-TB-at-scale raw output never accumulates (see notes §4).
#
# >>> EDIT the lines marked TODO for your Unity account before submitting.
# =============================================================================

#SBATCH --job-name=piv_dataset
#SBATCH --array=1-100%50            # 100 sims, <=50 running at once    # TODO size/throttle
#SBATCH --time=00:30:00            # wall time PER task (pilot: ~3-4 min/sim at nt=40)  # TODO tune
#SBATCH --mem=8G                   # per task                          # TODO tune
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu            # TODO your partition name (e.g. cpu, cpu-preempt)
##SBATCH --account=pi_account      # TODO uncomment + set if your allocation requires it
#SBATCH --output=logs/piv_%A_%a.out
#SBATCH --error=logs/piv_%A_%a.err

set -euo pipefail

# ---------- config ----------
PROJ="$HOME/ml-training"            # TODO path to the repo on Unity
BASE_SEED=0                         # seeds = BASE_SEED + array index → seeds 1..100
NT=40                               # frames per sim (developed-turbulence default; see notes §3)
DATASET="$PROJ/dataset"            # collected samples land here (one place for all tasks)

SEED=$(( BASE_SEED + SLURM_ARRAY_TASK_ID ))

# ---------- environment ----------
module purge
module load julia                   # TODO exact module name/version available on Unity
# Assumes the project is already instantiated & precompiled (Manifest.toml committed).
# If not, run once on a login node:  julia --project="$PROJ" -e 'using Pkg; Pkg.instantiate()'

# ---------- per-task scratch output (prevents cross-task collisions) ----------
# SLURM_TMPDIR is node-local & auto-cleaned; fall back to /tmp if unset.
WORK="${SLURM_TMPDIR:-/tmp}/piv_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WORK"
export PIV_OUT_DIR="$WORK"           # 2DTurbulence.jl/ImageGen.jl write under $WORK/data/{binary,visual}

echo "[task ${SLURM_ARRAY_TASK_ID}] seed=${SEED} nt=${NT} work=${WORK} start=$(date '+%F %T')"

# ---------- run: simulation -> combine -> image pairs (auto-chained) ----------
# Writes:  $WORK/data/binary/{fields,particles,_combined,metadata}_<tag>
#          $WORK/data/visual/pix{10,20,30}/image_pairs_<tag>_pixN.jld2  (ONE pair/sim)
julia --project="$PROJ" "$PROJ/scripts/2DTurbulence.jl" --nt "$NT" --seed "$SEED"

# ---------- collect the DURABLE artifacts into the shared dataset dir ----------
# Keep: the image-pair samples (what you train on) + the metadata sidecar (the seed,
# so any sim is regenerable). Drop: the raw simulation (large, reconstructible).
# Filenames already carry a unique tag; we also prefix with the seed for provenance.
shopt -s nullglob

for bin in pix10 pix20 pix30; do
    src="$WORK/data/visual/$bin"
    [ -d "$src" ] || continue
    mkdir -p "$DATASET/$bin"
    for f in "$src"/*.jld2; do
        cp "$f" "$DATASET/$bin/seed${SEED}_$(basename "$f")"
    done
done

mkdir -p "$DATASET/metadata"
for m in "$WORK"/data/binary/metadata_*.toml; do
    cp "$m" "$DATASET/metadata/seed${SEED}_$(basename "$m")"
done

# ---------- (optional) keep the combined raw file too — uncomment if you want ----
# zero-recompute re-rendering later (costs ~90 MB/sim). Off by default (notes §4).
# mkdir -p "$DATASET/combined"
# for c in "$WORK"/data/binary/*_combined.jld2; do
#     cp "$c" "$DATASET/combined/seed${SEED}_$(basename "$c")"
# done

# ---------- cleanup node-local scratch ----------
rm -rf "$WORK"
echo "[task ${SLURM_ARRAY_TASK_ID}] seed=${SEED} done=$(date '+%F %T')"

# =============================================================================
# GPU variant (optional, faster per sim): each 512x512 sim runs much faster on a
# GPU, but a CPU array usually gets more concurrency out of the queue for many
# short sims. To use GPU:
#   - SBATCH:  --partition=gpu   --gres=gpu:1   (and drop --cpus-per-task if desired)
#   - module:  also `module load cuda`
#   - code:    uncomment the GPU lines in 2DTurbulence.jl (RectilinearGrid(GPU(), ...)
#              and the CuArray particle setup). Currently CPU-mode by default.
# =============================================================================
