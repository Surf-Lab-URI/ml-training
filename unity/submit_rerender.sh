#!/bin/bash
# Re-render an EXISTING campaign with the lab-matched image appearance — NO re-simulation.
# Reads $SRC_RUN/combined/, writes a NEW run_labapp_<stamp>/ with the SAME metadata + manifest
# (flow/displacement unchanged, so they're just copied) and lab-appearance images.
#
# Usage: ./unity/submit_rerender.sh <SRC_RUN> [N_SEEDS] [CHUNK] [PARTITION] [KPART]
#   ./unity/submit_rerender.sh /project/pi_nicholas_pizzo_uri_edu/arup/piv_2dturb_dataset/run_2026-06-12_04-50-52 50000
set -euo pipefail

# Defaults come from params.toml — edit that file, not this script.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
eval "$(julia --project="$REPO" "$REPO/scripts/params_export.jl" 2>/dev/null || true)"

SRC_RUN=${1:?"give the SOURCE run dir (must contain combined/)"}
N=${2:-${PIV_N_SIMS:-50000}}
CHUNK=${3:-200}
PART=${4:-${PIV_PARTITION:-uri-cpu}}
KPART=${5:-${PIV_PARTICLES_PER_IMAGE:-12000}}

PROJ="${PIV_PROJECT_DIR:-$REPO}"
ROOT="${PIV_OUTPUT_ROOT:?set run.output_root in params.toml}"
STAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUT_RUN="$ROOT/run_labapp_$STAMP"
NTASKS=$(( (N + CHUNK - 1) / CHUNK ))

[ -d "$SRC_RUN/combined" ] || { echo "ERROR: $SRC_RUN/combined not found — need the kept combined sims"; exit 1; }
mkdir -p "$OUT_RUN/logs" "$OUT_RUN/pix10" "$OUT_RUN/pix20" "$OUT_RUN/pix30" "$OUT_RUN/code/scripts" "$OUT_RUN/code/src"

# Reuse metadata + manifest from the source (flow/displacement unchanged -> identical labels).
cp -r "$SRC_RUN/metadata" "$OUT_RUN/metadata"
cp "$SRC_RUN/manifest.parquet" "$OUT_RUN/manifest.parquet" 2>/dev/null || \
    echo "(no manifest.parquet in source — build later: python scripts/build_manifest.py --in $OUT_RUN/metadata --out $OUT_RUN/manifest.parquet)"

# Snapshot the appearance code + a RUN_INFO.
cp "$PROJ/scripts/ImageGen.jl" "$OUT_RUN/code/scripts/" 2>/dev/null || true
cp "$PROJ/src/ImageGenFunc.jl" "$OUT_RUN/code/src/"     2>/dev/null || true
{
  echo "run          : run_labapp_$STAMP  (LAB-APPEARANCE re-render — no re-simulation)"
  echo "source_run   : $SRC_RUN"
  echo "date_time    : $(date '+%F %T %Z')"
  echo "n_seeds      : $N   (chunk=$CHUNK -> $NTASKS array tasks)"
  echo "k_particles  : $KPART   (lab-like density)"
  echo "appearance   : PIV_LAB_APPEARANCE=1 — gray bg~50, contrast~4x, particle~2px, noise~5 (domain-randomized)"
  echo "git_commit   : $(cd "$PROJ" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "note         : metadata/ + manifest.parquet COPIED from source (only the images differ)"
} > "$OUT_RUN/RUN_INFO.txt"

JID=$(sbatch --parsable \
      --partition="$PART" --time=01:00:00 --array=1-${NTASKS}%100 \
      --output="$OUT_RUN/logs/rr_%A_%a.out" --error="$OUT_RUN/logs/rr_%A_%a.err" \
      --export=ALL,SRC_RUN="$SRC_RUN",OUT_RUN="$OUT_RUN",PROJ="$PROJ",CHUNK="$CHUNK",KPART="$KPART",BASE_SEED=0 \
      "$PROJ/unity/rerender_lab.sbatch")

echo "[submit] re-render array $JID  ->  $OUT_RUN"
echo "Run folder : $OUT_RUN"
echo "Watch      : squeue --me     |     tail -f $OUT_RUN/logs/rr_${JID}_1.out"
echo "Train (on Unity) when done: point piv-models at  $OUT_RUN  (bin pix10 — closest to the lab's ~6 px)"
