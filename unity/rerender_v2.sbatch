#!/bin/bash
# v2 re-render: EXISTING combined sims -> image pairs binned by MEDIAN displacement.
#
# Same pattern as rerender_lab.sbatch (no re-simulation; reads $SRC_RUN/combined/seed<N>.jld2),
# but calls datagen_v2/ImageGenV2.jl, which picks the B frame at a FRACTIONAL index so any target
# displacement is reachable (BUG-13/14 fix) and bins by median rather than by an integer frame gap.
#
# Bins: med03 med06 med09 med12 med16 med20 med26 med30  ->  max ~ 5 10 15 20 27 33 43 50 px.
# Rationale: datagen_v2/DATA_REQUIREMENTS.md
#
# Submit via unity/submit_v2.sh.
#SBATCH --job-name=piv_v2gen
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

set -euo pipefail
: "${SRC_RUN:?SRC_RUN not set — submit via unity/submit_v2.sh}"
: "${OUT_RUN:?OUT_RUN not set}"
: "${PROJ:?PROJ not set}"
CHUNK=${CHUNK:-200}
KPART=${KPART:-12000}
BASE_SEED=${BASE_SEED:-0}
BINS="med03 med06 med09 med12 med16 med20 med26 med30"

export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH=/work/pi_nicholas_pizzo_uri_edu/arup_mazumder/julia_depot
export JULIA_CPU_TARGET=generic
export PIV_LAB_APPEARANCE=1                    # lab-matched appearance, as in the v1 re-render
# PIV_V2_MEDIANS / PIV_V2_TOL / PIV_V2_LOOSE are passed through by the submitter if set.

FIRST=$(( BASE_SEED + (SLURM_ARRAY_TASK_ID - 1) * CHUNK + 1 ))
LAST=$((  BASE_SEED + SLURM_ARRAY_TASK_ID * CHUNK ))

WORK="${SLURM_TMPDIR:-/tmp}/v2_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
IN="$WORK/in"; export PIV_OUT_DIR="$WORK/out"
mkdir -p "$IN" "$PIV_OUT_DIR/data/binary"

# Symlink this chunk's combined files as *_combined.jld2 so -d picks them up and derives
# vars="seed<N>"  ->  image_pairs_seed<N>_med<NN>.jld2 (one pair per file).
n=0
for s in $(seq "$FIRST" "$LAST"); do
    c="$SRC_RUN/combined/seed${s}.jld2"
    [ -f "$c" ] && ln -sf "$c" "$IN/seed${s}_combined.jld2" && n=$((n+1))
done
echo "[task ${SLURM_ARRAY_TASK_ID}] seeds ${FIRST}..${LAST}  (${n} combined found)  start=$(date '+%F %T')"
if [ "$n" -eq 0 ]; then echo "no combined files in range — nothing to do"; rm -rf "$WORK"; exit 0; fi

julia --project="$PROJ" "$PROJ/datagen_v2/ImageGenV2.jl" -d "$IN" -k "$KPART"

# Copy outputs into the run folder, stripping the image_pairs_ prefix -> seed<N>_med<NN>.jld2
shopt -s nullglob
for bin in $BINS; do
    src="$PIV_OUT_DIR/data/visual_v2/$bin"
    [ -d "$src" ] || continue
    mkdir -p "$OUT_RUN/$bin"
    for f in "$src"/image_pairs_*_${bin}.jld2; do
        base=$(basename "$f"); cp "$f" "$OUT_RUN/$bin/${base#image_pairs_}"
    done
done

# v2 sidecars carry the achieved distribution per sample — keep them, they are how the dataset
# gets filtered later (max displacement alone is not a usable descriptor).
if [ -d "$PIV_OUT_DIR/data/visual_v2/metadata" ]; then
    mkdir -p "$OUT_RUN/metadata_v2"
    cp "$PIV_OUT_DIR/data/visual_v2/metadata"/*.toml "$OUT_RUN/metadata_v2/" 2>/dev/null || true
fi

rm -rf "$WORK"
echo "[task ${SLURM_ARRAY_TASK_ID}] done=$(date '+%F %T')"
