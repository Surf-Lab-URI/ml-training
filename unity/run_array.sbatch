#!/bin/bash
# Per-simulation worker for one campaign. NOT submitted directly — submit via
# unity/submit_run.sh, which sets RUN_DIR/PROJ/NT/BASE_SEED and the --array range.
#SBATCH --job-name=piv_run
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

set -euo pipefail

: "${RUN_DIR:?RUN_DIR not set — submit via unity/submit_run.sh}"
: "${PROJ:?PROJ not set — submit via unity/submit_run.sh}"
BASE_SEED=${BASE_SEED:-0}
NT=${NT:-40}
DATASET="$RUN_DIR"
SEED=$(( BASE_SEED + SLURM_ARRAY_TASK_ID ))

# Julia 1.12.6 (juliaup), packages on /work, portable cache (no per-node recompile)
export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH=/work/pi_nicholas_pizzo_uri_edu/arup_mazumder/julia_depot
export JULIA_CPU_TARGET=generic

# per-task node-local scratch so parallel tasks never collide
WORK="${SLURM_TMPDIR:-/tmp}/piv_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WORK"
export PIV_OUT_DIR="$WORK"

echo "[task ${SLURM_ARRAY_TASK_ID}] seed=${SEED} nt=${NT} start=$(date '+%F %T')"
julia --project="$PROJ" "$PROJ/scripts/2DTurbulence.jl" --nt "$NT" --seed "$SEED"

# collect the durable artifacts into the run folder, named purely by seed
shopt -s nullglob
for bin in pix10 pix20 pix30; do
    src="$WORK/data/visual/$bin"
    [ -d "$src" ] || continue
    mkdir -p "$DATASET/$bin"
    for f in "$src"/*.jld2; do cp "$f" "$DATASET/$bin/seed${SEED}_${bin}.jld2"; done
done
mkdir -p "$DATASET/metadata"
for m in "$WORK"/data/binary/metadata_*.toml; do cp "$m" "$DATASET/metadata/seed${SEED}.toml"; done
# Keep the combined raw sim for later re-rendering, UNLESS KEEP_COMBINED=0
# (set 0 at ~100k scale: combined would be ~9 TB; samples alone are ~1.4 TB).
if [ "${KEEP_COMBINED:-1}" = "1" ]; then
    mkdir -p "$DATASET/combined"
    for c in "$WORK"/data/binary/*_combined.jld2; do cp "$c" "$DATASET/combined/seed${SEED}.jld2"; done
fi

rm -rf "$WORK"
echo "[task ${SLURM_ARRAY_TASK_ID}] seed=${SEED} done=$(date '+%F %T')"
