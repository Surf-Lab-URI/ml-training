#!/bin/bash
#SBATCH --job-name=piv_dataset
#SBATCH --array=1-100%50
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --output=logs/piv_%A_%a.out
#SBATCH --error=logs/piv_%A_%a.err

set -euo pipefail

PROJ="/work/pi_nicholas_pizzo_uri_edu/arup_mazumder/ml-training"
BASE_SEED=0
NT=40
DATASET="$PROJ/dataset"
SEED=$(( BASE_SEED + SLURM_ARRAY_TASK_ID ))

export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH=/work/pi_nicholas_pizzo_uri_edu/arup_mazumder/julia_depot
export JULIA_CPU_TARGET=generic   # portable precompile cache: build once, load on any compute node (no per-node recompile)

WORK="${SLURM_TMPDIR:-/tmp}/piv_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WORK"
export PIV_OUT_DIR="$WORK"

echo "[task ${SLURM_ARRAY_TASK_ID}] seed=${SEED} nt=${NT} start=$(date '+%F %T')"
julia --project="$PROJ" "$PROJ/scripts/2DTurbulence.jl" --nt "$NT" --seed "$SEED"

shopt -s nullglob
for bin in pix10 pix20 pix30; do
    src="$WORK/data/visual/$bin"
    [ -d "$src" ] || continue
    mkdir -p "$DATASET/$bin"
    for f in "$src"/*.jld2; do cp "$f" "$DATASET/$bin/seed${SEED}_$(basename "$f")"; done
done
mkdir -p "$DATASET/metadata"
for m in "$WORK"/data/binary/metadata_*.toml; do cp "$m" "$DATASET/metadata/seed${SEED}_$(basename "$m")"; done

rm -rf "$WORK"
echo "[task ${SLURM_ARRAY_TASK_ID}] seed=${SEED} done=$(date '+%F %T')"
