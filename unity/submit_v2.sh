#!/bin/bash
# Submit the v2 re-render: existing combined sims -> image pairs binned by MEDIAN displacement.
# No re-simulation. Writes a NEW run_v2_<stamp>/ alongside the source run.
#
# Usage: ./unity/submit_v2.sh <SRC_RUN> [N_SEEDS] [CHUNK] [PARTITION] [KPART] [TIME]
#   ./unity/submit_v2.sh /project/pi_nicholas_pizzo_uri_edu/arup/piv_2dturb_dataset/run_2026-06-12_04-50-52 200
#
# START SMALL. Run a pilot of a few hundred seeds, check the achieved medians in the logs and the
# metadata_v2/ sidecars, and only then scale up. The whole premise — that widening the training
# range moves the model's 22 px ceiling — is untested.
set -euo pipefail

SRC_RUN=${1:?"give the SOURCE run dir (must contain combined/)"}
N=${2:-200}
CHUNK=${3:-50}
PART=${4:-uri-cpu}
KPART=${5:-12000}
# Wall clock per array task. A 200-seed chunk is a lot of work: each seed reads 41
# frames, solves 8 displacement targets and renders 16 images. Measure one task
# before scaling, and raise this if tasks are being killed at the limit.
TIME=${6:-04:00:00}

PROJ="/work/pi_nicholas_pizzo_uri_edu/arup_mazumder/ml-training"
ROOT="/project/pi_nicholas_pizzo_uri_edu/arup/piv_2dturb_dataset"
STAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUT_RUN="$ROOT/run_v2_$STAMP"
NTASKS=$(( (N + CHUNK - 1) / CHUNK ))
BINS="med03 med06 med09 med12 med16 med20 med26 med30"

# A bare number here means $RUN was unset and the arguments shifted along one.
case "$SRC_RUN" in
    /*) ;;
    *)  echo "ERROR: first argument is '$SRC_RUN', which is not an absolute path."
        echo
        echo "This usually means \$RUN was empty (it does not survive a new login), so the"
        echo "seed count was read as the run directory. Set it first:"
        echo "  RUN=/project/pi_nicholas_pizzo_uri_edu/arup/piv_2dturb_dataset/run_<the real name>"
        echo "  ls \$RUN/combined | head -3"
        echo "  ./unity/submit_v2.sh \$RUN 200 50"
        echo
        echo "Runs that actually have combined/:"
        ls -d /project/pi_nicholas_pizzo_uri_edu/arup/piv_2dturb_dataset/run_*/combined 2>/dev/null \
            | sed 's|/combined$||; s|^|  |' || echo "  (none found)"
        exit 1 ;;
esac

[ -d "$SRC_RUN/combined" ] || {
    echo "ERROR: $SRC_RUN/combined not found — need the kept combined sims"
    echo "Runs that do have combined/:"
    ls -d /project/pi_nicholas_pizzo_uri_edu/arup/piv_2dturb_dataset/run_*/combined 2>/dev/null \
        | sed 's|/combined$||; s|^|  |' || echo "  (none found)"
    exit 1; }

# Seed numbering does not start at 1 in every campaign (run_2026-06-12 starts at seed10000).
# Detect what is actually there instead of assuming, so the array covers real files rather than
# scanning an empty range and exiting with "no combined files in range".
SEEDS=$(ls "$SRC_RUN/combined" 2>/dev/null | sed -n 's/^seed\([0-9][0-9]*\)\.jld2$/\1/p' | sort -n)
[ -n "$SEEDS" ] || { echo "ERROR: no seed<N>.jld2 files in $SRC_RUN/combined"; exit 1; }
# Parameter expansion, not `echo | head -1`: with ~10k seeds head closes the pipe after one
# line, echo takes SIGPIPE, and `set -o pipefail` + `set -e` kill the script silently (exit 141).
# That failure only appears at scale, which is exactly when it matters.
MINSEED=${SEEDS%%$'\n'*}
MAXSEED=${SEEDS##*$'\n'}
NAVAIL=$(printf '%s\n' "$SEEDS" | grep -c .)

# BASE_SEED is an offset: the sbatch computes FIRST = BASE_SEED + (task-1)*CHUNK + 1.
BASE_SEED=${BASE_SEED:-$(( MINSEED - 1 ))}

if [ "$N" -gt "$NAVAIL" ]; then
    echo "note: asked for $N seeds, only $NAVAIL exist — using $NAVAIL"
    N=$NAVAIL
fi
NTASKS=$(( (N + CHUNK - 1) / CHUNK ))
LASTWANTED=$(( BASE_SEED + N ))

echo "source     : $SRC_RUN"
echo "available  : $NAVAIL seeds (seed$MINSEED .. seed$MAXSEED)"
echo "generating : seed$(( BASE_SEED + 1 )) .. seed$LASTWANTED   ($N seeds, chunk $CHUNK -> $NTASKS tasks)"
echo "bins       : $BINS"
echo "output     : $OUT_RUN"
echo "estimate   : $(( N * 8 )) samples, ~$(( N * 8 * 45 / 10240 )) GB"
echo "wall clock : $TIME per task (override with the 6th argument)"
echo
if [ "$LASTWANTED" -gt "$MAXSEED" ]; then
    echo "note: the requested range runs past seed$MAXSEED; the trailing tasks will find"
    echo "      nothing and exit cleanly. Harmless, but you will get fewer than $(( N * 8 )) samples."
    echo
fi

mkdir -p "$OUT_RUN/logs" "$OUT_RUN/code/datagen_v2" "$OUT_RUN/metadata_v2"
for b in $BINS; do mkdir -p "$OUT_RUN/$b"; done

# Metadata from the source describes the FLOW, which is unchanged — only the pairing differs.
cp -r "$SRC_RUN/metadata" "$OUT_RUN/metadata" 2>/dev/null || true

# Snapshot the exact generator used, so a run can be reproduced later.
cp "$PROJ/datagen_v2/ImageGenV2.jl"        "$OUT_RUN/code/datagen_v2/" 2>/dev/null || true
cp "$PROJ/datagen_v2/FracFrame.jl"         "$OUT_RUN/code/datagen_v2/" 2>/dev/null || true
cp "$PROJ/datagen_v2/DATA_REQUIREMENTS.md" "$OUT_RUN/code/datagen_v2/" 2>/dev/null || true
cp "$PROJ/src/ImageGenFunc.jl"             "$OUT_RUN/code/" 2>/dev/null || true

{
  echo "run          : run_v2_$STAMP  (v2 re-render — median-binned, no re-simulation)"
  echo "source_run   : $SRC_RUN"
  echo "date_time    : $(date '+%F %T %Z')"
  echo "n_seeds      : $N   (chunk=$CHUNK -> $NTASKS array tasks)"
  echo "seed_range   : seed$(( BASE_SEED + 1 )) .. seed$LASTWANTED   (BASE_SEED=$BASE_SEED, $NAVAIL available)"
  echo "k_particles  : $KPART"
  echo "time_limit   : $TIME per array task"
  echo "bins         : $BINS   (MEDIAN displacement in px; max ~ 1.67x the median)"
  echo "generator    : datagen_v2/ImageGenV2.jl  (fractional B frame — BUG-13/14 fixed)"
  echo "appearance   : PIV_LAB_APPEARANCE=1"
  echo "git_commit   : $(cd "$PROJ" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "note         : out-of-tolerance samples are SKIPPED, not mislabelled — check the logs"
  echo "               for 'skipping' lines and metadata_v2/*.toml for achieved distributions."
} > "$OUT_RUN/RUN_INFO.txt"

JID=$(sbatch --parsable \
      --partition="$PART" --time="$TIME" --array=1-${NTASKS}%100 \
      --output="$OUT_RUN/logs/v2_%A_%a.out" --error="$OUT_RUN/logs/v2_%A_%a.err" \
      --export=ALL,SRC_RUN="$SRC_RUN",OUT_RUN="$OUT_RUN",PROJ="$PROJ",CHUNK="$CHUNK",KPART="$KPART",BASE_SEED="$BASE_SEED" \
      "$PROJ/unity/rerender_v2.sbatch")

echo "[submit] v2 array $JID  ->  $OUT_RUN"
echo
echo "Run folder : $OUT_RUN"
echo "Watch      : squeue --me    |    tail -f $OUT_RUN/logs/v2_${JID}_1.out"
echo
echo "When the pilot finishes, check:"
echo "  grep -h 'skipping'  $OUT_RUN/logs/*.out | head        # targets that could not be reached"
echo "  grep -h 'OFF by'    $OUT_RUN/logs/*.out | head        # and by how much"
echo "  for b in $BINS; do echo -n \"\$b \"; ls $OUT_RUN/\$b | wc -l; done"
