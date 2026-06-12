#!/bin/bash
# Submit ONE PIV data-generation campaign.
#
# Creates a timestamped run folder under the dataset root, snapshots the exact code
# used, writes an at-a-glance RUN_INFO.txt, launches the per-sim SLURM array into that
# folder, and auto-submits a finalize job (manifest + 2% preview) that runs afterward.
#
# Usage:
#   ./unity/submit_run.sh [N_SIMS] [NT] [PARTITION] [BASE_SEED]
# e.g.
#   ./unity/submit_run.sh 1000            # 1000 sims, nt=40, uri-cpu, seeds 1..1000
#   ./unity/submit_run.sh 100000 40 cpu-preempt
set -euo pipefail

N=${1:-1000}
NT=${2:-40}
PART=${3:-uri-cpu}
BASE_SEED=${4:-0}

PROJ="/work/pi_nicholas_pizzo_uri_edu/arup_mazumder/ml-training"
ROOT="/project/pi_nicholas_pizzo_uri_edu/arup/piv_2dturb_dataset"
PIX="10,20,30"

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_DIR="$ROOT/run_$STAMP"
mkdir -p "$RUN_DIR/code/scripts" "$RUN_DIR/code/src" "$RUN_DIR/logs"

# --- 1. snapshot the EXACT code used for this run ---
cp "$PROJ/scripts/2DTurbulence.jl" "$PROJ/scripts/ImageGen.jl" \
   "$PROJ/scripts/build_manifest.py" "$PROJ/scripts/render_preview.py" "$RUN_DIR/code/scripts/" 2>/dev/null || true
cp "$PROJ/src/args.jl" "$PROJ/src/CombineAndConquer.jl" "$PROJ/src/ImageGenFunc.jl" "$RUN_DIR/code/src/" 2>/dev/null || true
cp "$PROJ/Project.toml" "$PROJ/Manifest.toml" "$RUN_DIR/code/" 2>/dev/null || true
{
    echo "commit: $(cd "$PROJ" && git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "branch: $(cd "$PROJ" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "uncommitted (git status --porcelain):"
    (cd "$PROJ" && git status --porcelain 2>/dev/null) || true
} > "$RUN_DIR/code/git_info.txt"

# --- 2. at-a-glance RUN_INFO.txt ---
GITSHA=$(cd "$PROJ" && git rev-parse --short HEAD 2>/dev/null || echo unknown)
{
    echo "run             : run_$STAMP"
    echo "date_time       : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "n_simulations   : $N   (seeds $((BASE_SEED+1))..$((BASE_SEED+N)), base_seed=$BASE_SEED)"
    echo "nt (frames)     : $NT"
    echo "pix bins        : $PIX"
    echo "partition       : $PART"
    echo "git_commit      : $GITSHA"
    echo "julia / target  : 1.12.6 / generic"
    echo "grid / physics  : 512x512 periodic, WENO(order=5), ScalarDiffusivity nu=1e-5"
    echo "output_root     : $RUN_DIR"
    echo "--- IC argument defaults (src/args.jl) ---"
    awk '/"--[a-z]/{a=$0} /default[[:space:]]*=/{gsub(/^[[:space:]]+/,"",a); gsub(/^[[:space:]]+/,"",$0); print "  "a"   "$0}' "$PROJ/src/args.jl" 2>/dev/null
} > "$RUN_DIR/RUN_INFO.txt"

# --- 3. submit the per-sim array, pointed at this run folder ---
JID=$(sbatch --parsable \
      --partition="$PART" --time=00:40:00 --array=1-${N}%100 \
      --output="$RUN_DIR/logs/piv_%A_%a.out" --error="$RUN_DIR/logs/piv_%A_%a.err" \
      --export=ALL,RUN_DIR="$RUN_DIR",NT="$NT",BASE_SEED="$BASE_SEED",PROJ="$PROJ" \
      "$PROJ/unity/run_array.sbatch")
echo "[submit] array job    $JID  ->  $RUN_DIR"

# --- 4. finalize (manifest + 2% preview) AFTER the array finishes (any outcome) ---
FID=$(sbatch --parsable \
      --partition="$PART" \
      --dependency=afterany:"$JID" \
      --output="$RUN_DIR/logs/finalize_%j.out" --error="$RUN_DIR/logs/finalize_%j.err" \
      --export=ALL,RUN_DIR="$RUN_DIR",PROJ="$PROJ" \
      "$PROJ/unity/finalize_run.sbatch")
echo "[submit] finalize job $FID  (runs after $JID)"
echo
echo "Run folder : $RUN_DIR"
echo "Watch      : squeue --me"
echo "RUN_INFO   : cat $RUN_DIR/RUN_INFO.txt"
