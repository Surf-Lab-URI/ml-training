#!/bin/bash
# Submit the v3 re-render: existing combined sims -> image pairs with the tank's measured
# near-surface shear layer, a free surface, and top-2mm displacement binning.
# No re-simulation. Writes a NEW run_v3_<stamp>/ alongside the source run.
#
# Usage: ./unity/submit_v3.sh <SRC_RUN> [N_SEEDS] [CHUNK] [PARTITION] [KPART] [TIME]
#   ./unity/submit_v3.sh /project/pi_nicholas_pizzo_uri_edu/arup/piv_2dturb_dataset/run_2026-06-12_04-50-52 200
#
# START SMALL. Run a pilot of a few hundred seeds, check the achieved medians in the logs and the
# metadata_v3/ sidecars, and only then scale up. The whole premise — that widening the training
# range moves the model's 22 px ceiling — is untested.
set -euo pipefail

# Defaults come from params.toml — edit that file, not this script. Positional arguments below
# still override it for a one-off run. See params.toml's header for the precedence rules.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
eval "$(julia --project="$REPO" "$REPO/scripts/params_export.jl" 2>/dev/null || true)"

SRC_RUN=${1:?"give the SOURCE run dir (must contain combined/)"}
N=${2:-${PIV_N_SIMS:-200}}
CHUNK=${3:-${PIV_CHUNK:-50}}
PART=${4:-${PIV_PARTITION:-uri-cpu}}
KPART=${5:-${PIV_PARTICLES_PER_IMAGE:-12000}}
# Wall clock per array task. A 200-seed chunk is a lot of work: each seed reads 41
# frames, solves 8 displacement targets and renders 16 images. Measure one task
# before scaling, and raise this if tasks are being killed at the limit.
TIME=${6:-${PIV_TIME_RENDER:-04:00:00}}

PROJ="${PIV_PROJECT_DIR:-$REPO}"
ROOT="${PIV_OUTPUT_ROOT:?set run.output_root in params.toml}"
STAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUT_RUN="$ROOT/run_v3_$STAMP"
NTASKS=$(( (N + CHUNK - 1) / CHUNK ))
# Bin names come from params.toml [bins.v3].surface_medians, via scripts/params_export.jl above.
# There is deliberately NO hardcoded fallback: a stale copy of the list here would make this script
# create surf02..surf24 folders while the generator writes surf06..surf35, so half the output would
# land outside the run and half the folders would stay empty. Fail loudly instead.
BINS="${PIV_V3_BINS:?bins.v3.surface_medians missing from params.toml (or params_export.jl failed) — check: julia --project=. scripts/params_export.jl | grep V3}"

# A bare number here means $RUN was unset and the arguments shifted along one.
case "$SRC_RUN" in
    /*) ;;
    *)  echo "ERROR: first argument is '$SRC_RUN', which is not an absolute path."
        echo
        echo "This usually means \$RUN was empty (it does not survive a new login), so the"
        echo "seed count was read as the run directory. Set it first:"
        echo "  RUN=/project/pi_nicholas_pizzo_uri_edu/arup/piv_2dturb_dataset/run_<the real name>"
        echo "  ls \$RUN/combined | head -3"
        echo "  ./unity/submit_v3.sh \$RUN 200 50"
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

mkdir -p "$OUT_RUN/logs" "$OUT_RUN/code/datagen_v3" "$OUT_RUN/metadata_v3"
for b in $BINS; do mkdir -p "$OUT_RUN/$b"; done

# Metadata from the source describes the FLOW, which is unchanged — only the pairing differs.
cp -r "$SRC_RUN/metadata" "$OUT_RUN/metadata" 2>/dev/null || true

# Snapshot the exact generator used, so a run can be reproduced later.
cp "$PROJ/datagen_v3/ImageGenV3.jl"        "$OUT_RUN/code/datagen_v3/" 2>/dev/null || true
cp "$PROJ/datagen_v3/ShearProfile.jl"      "$OUT_RUN/code/datagen_v3/" 2>/dev/null || true
cp "$PROJ/datagen_v2/FracFrame.jl"         "$OUT_RUN/code/datagen_v2/" 2>/dev/null || true
cp "$PROJ/datagen_v2/DATA_REQUIREMENTS.md" "$OUT_RUN/code/datagen_v2/" 2>/dev/null || true
cp "$PROJ/src/ImageGenFunc.jl"             "$OUT_RUN/code/" 2>/dev/null || true

{
  echo "run          : run_v3_$STAMP  (v3 re-render — sheared + free surface, no re-simulation)"
  echo "source_run   : $SRC_RUN"
  echo "date_time    : $(date '+%F %T %Z')"
  echo "n_seeds      : $N   (chunk=$CHUNK -> $NTASKS array tasks)"
  echo "seed_range   : seed$(( BASE_SEED + 1 )) .. seed$LASTWANTED   (BASE_SEED=$BASE_SEED, $NAVAIL available)"
  echo "k_particles  : $KPART"
  echo "time_limit   : $TIME per array task"
  echo "bins         : $BINS   (TOP-2mm MEDIAN displacement in px)"
  echo "generator    : datagen_v3/ImageGenV3.jl  (shear layer + free surface; field NOT divergence-free)"
  echo "appearance   : PIV_LAB_APPEARANCE=1"
  echo "git_commit   : $(cd "$PROJ" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "warning      : field is NOT divergence-free -- train with lambda_div = 0"
  echo "note         : out-of-tolerance samples are SKIPPED, not mislabelled — check the logs"
  echo "               for 'skipping' lines and metadata_v3/*.toml for achieved distributions."
} > "$OUT_RUN/RUN_INFO.txt"

JID=$(sbatch --parsable \
      --partition="$PART" --time="$TIME" \
      --mem="${PIV_MEM:-8G}" --cpus-per-task="${PIV_CPUS_PER_TASK:-4}" \
      --array=1-${NTASKS}%${PIV_MAX_CONCURRENT:-100} \
      --output="$OUT_RUN/logs/v3_%A_%a.out" --error="$OUT_RUN/logs/v3_%A_%a.err" \
      --export=ALL,SRC_RUN="$SRC_RUN",OUT_RUN="$OUT_RUN",PROJ="$PROJ",CHUNK="$CHUNK",KPART="$KPART",BASE_SEED="$BASE_SEED" \
      "$PROJ/unity/rerender_v3.sbatch")

echo "[submit] v3 array $JID  ->  $OUT_RUN"
echo
echo "Run folder : $OUT_RUN"
echo "Watch      : squeue --me    |    tail -f $OUT_RUN/logs/v3_${JID}_1.out"
echo
echo "When the pilot finishes, check:"
echo "  grep -h 'skipping'  $OUT_RUN/logs/*.out | head        # targets that could not be reached"
echo "  grep -h 'OFF by'    $OUT_RUN/logs/*.out | head        # and by how much"
echo "  for b in $BINS; do echo -n \"\$b \"; ls $OUT_RUN/\$b | wc -l; done"
