#!/bin/bash
# Re-render EXISTING combined sims with the lab-matched image appearance
# (PIV_LAB_APPEARANCE=1). Does NOT re-simulate — reads $SRC_RUN/combined/seed<N>.jld2 and
# writes $OUT_RUN/pixB/seed<N>_pixB.jld2. Chunked (CHUNK seeds per task) so Julia loads once
# per task, not once per seed. Submit via unity/submit_rerender.sh.
#SBATCH --job-name=piv_rerender
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

set -euo pipefail
: "${SRC_RUN:?SRC_RUN not set — submit via unity/submit_rerender.sh}"
: "${OUT_RUN:?OUT_RUN not set}"
: "${PROJ:?PROJ not set}"
CHUNK=${CHUNK:-200}
KPART=${KPART:-12000}
BASE_SEED=${BASE_SEED:-0}

export PATH="$HOME/.juliaup/bin:$PATH"
# $HOME is on NFS and is occasionally not resolvable the instant a task starts, which makes
# `julia` vanish from PATH and the task die with exit 127. That hit 34 of 153 tasks in the
# 2026-08-26 re-render -- intermittently, on nodes where other tasks of the same array
# succeeded. Resolve the binary explicitly, with retries, and fail loudly rather than at
# line 41 with a bare "command not found".
JULIA=""
for _try in 1 2 3 4 5; do
    JULIA=$(command -v julia 2>/dev/null || true)
    [ -n "$JULIA" ] && [ -x "$JULIA" ] && break
    echo "[warn] julia not on PATH (attempt $_try) -- waiting for \$HOME to settle"
    sleep 10
    export PATH="$HOME/.juliaup/bin:$PATH"
done
[ -n "$JULIA" ] || { echo "[fatal] julia unresolvable after 5 attempts on $(hostname)"; exit 1; }
echo "[info] julia = $JULIA"
export JULIA_DEPOT_PATH=/work/pi_nicholas_pizzo_uri_edu/arup_mazumder/julia_depot
export JULIA_CPU_TARGET=generic
export PIV_LAB_APPEARANCE=1                     # <-- the lab-matched appearance

# seeds handled by this array task
FIRST=$(( BASE_SEED + (SLURM_ARRAY_TASK_ID - 1) * CHUNK + 1 ))
LAST=$((  BASE_SEED + SLURM_ARRAY_TASK_ID * CHUNK ))

WORK="${SLURM_TMPDIR:-/tmp}/rr_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
IN="$WORK/in"; export PIV_OUT_DIR="$WORK/out"
mkdir -p "$IN" "$PIV_OUT_DIR/data/binary"

# Symlink this chunk's combined files as *_combined.jld2 so ImageGen -d picks them up and
# derives vars="seed<N>" -> output image_pairs_seed<N>_pixB.jld2 (one pair per file).
n=0
for s in $(seq "$FIRST" "$LAST"); do
    c="$SRC_RUN/combined/seed${s}.jld2"
    [ -f "$c" ] && ln -sf "$c" "$IN/seed${s}_combined.jld2" && n=$((n+1))
done
echo "[task ${SLURM_ARRAY_TASK_ID}] seeds ${FIRST}..${LAST}  (${n} combined found)  start=$(date '+%F %T')"
if [ "$n" -eq 0 ]; then echo "no combined files in range — nothing to do"; rm -rf "$WORK"; exit 0; fi

"$JULIA" --project="$PROJ" "$PROJ/scripts/ImageGen.jl" -d "$IN" -k "$KPART"

# Rename outputs to seed<N> and copy into the run folder (strip the image_pairs_ prefix).
shopt -s nullglob
for bin in pix10 pix20 pix30; do
    src="$PIV_OUT_DIR/data/visual/$bin"
    [ -d "$src" ] || continue
    mkdir -p "$OUT_RUN/$bin"
    for f in "$src"/image_pairs_*_${bin}.jld2; do
        base=$(basename "$f"); cp "$f" "$OUT_RUN/$bin/${base#image_pairs_}"   # -> seed<N>_pixB.jld2
    done
done

rm -rf "$WORK"
echo "[task ${SLURM_ARRAY_TASK_ID}] done=$(date '+%F %T')"
