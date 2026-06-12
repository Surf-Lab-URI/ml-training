#!/bin/bash
# Submit a campaign LARGER than the QOS submit limit (2000 queued jobs), in chunks,
# all into ONE run folder. Keeps the queue under MAXQ by waiting between chunks, so it
# never trips QOSMaxSubmitJobPerUserLimit. Run it detached so it submits over time:
#
#   nohup ./unity/submit_chunked.sh 10000 40 cpu-preempt > /tmp/chunked.log 2>&1 &
#
# Usage: ./unity/submit_chunked.sh [N_SIMS] [NT] [PARTITION] [BASE_SEED]
# Tunables via env: CHUNK (default 1000), MAXQ (default 1500).
set -uo pipefail        # NOT -e: a transient sbatch hiccup must not kill the whole driver

N=${1:-10000}
NT=${2:-40}
PART=${3:-cpu-preempt}
BASE=${4:-0}
CHUNK=${CHUNK:-1000}    # sims per chunk (< QOS submit limit 2000)
KEEP_COMBINED=${KEEP_COMBINED:-1}   # 1=keep raw combined files; set 0 at 100k scale (~9 TB)
MAXQ=${MAXQ:-900}       # submit next chunk only when queued+running < this.
                        # Must satisfy MAXQ + CHUNK < 2000 (QOS cap) → 900+1000=1900 ✓

PROJ="/work/pi_nicholas_pizzo_uri_edu/arup_mazumder/ml-training"
ROOT="/project/pi_nicholas_pizzo_uri_edu/arup/piv_2dturb_dataset"
PIX="10,20,30"

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_DIR="$ROOT/run_$STAMP"
mkdir -p "$RUN_DIR/code/scripts" "$RUN_DIR/code/src" "$RUN_DIR/logs"

# --- code snapshot ---
cp "$PROJ/scripts/2DTurbulence.jl" "$PROJ/scripts/ImageGen.jl" \
   "$PROJ/scripts/build_manifest.py" "$PROJ/scripts/render_preview.py" "$RUN_DIR/code/scripts/" 2>/dev/null || true
cp "$PROJ/src/args.jl" "$PROJ/src/CombineAndConquer.jl" "$PROJ/src/ImageGenFunc.jl" "$RUN_DIR/code/src/" 2>/dev/null || true
cp "$PROJ/Project.toml" "$PROJ/Manifest.toml" "$RUN_DIR/code/" 2>/dev/null || true
{ echo "commit: $(cd "$PROJ" && git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "uncommitted:"; (cd "$PROJ" && git status --porcelain 2>/dev/null) || true; } > "$RUN_DIR/code/git_info.txt"

# --- RUN_INFO ---
GITSHA=$(cd "$PROJ" && git rev-parse --short HEAD 2>/dev/null || echo unknown)
{
    echo "run             : run_$STAMP   (CHUNKED)"
    echo "date_time       : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "n_simulations   : $N   (seeds $((BASE+1))..$((BASE+N)), base_seed=$BASE)"
    echo "nt (frames)     : $NT"
    echo "pix bins        : $PIX"
    echo "partition       : $PART     chunk=$CHUNK  maxq=$MAXQ"
    echo "git_commit      : $GITSHA"
    echo "julia / target  : 1.12.6 / generic"
    echo "grid / physics  : 512x512 periodic, WENO(order=5), ScalarDiffusivity nu=1e-5"
    echo "output_root     : $RUN_DIR"
} > "$RUN_DIR/RUN_INFO.txt"

echo "[chunked] run folder: $RUN_DIR"
echo "[chunked] $N sims in chunks of $CHUNK on $PART (maxq=$MAXQ)"

# --- submit chunks, waiting for queue room ---
a=1
jids=""
while [ "$a" -le "$N" ]; do
    b=$((a + CHUNK - 1)); [ "$b" -gt "$N" ] && b=$N
    sz=$((b - a + 1))
    off=$((BASE + a - 1))     # seeds for this chunk = off+1 .. off+sz (array idx 1..sz)

    # wait until the queue has room (keeps total under the QOS submit limit)
    while [ "$(squeue -h -u "$USER" -r -t PD,R 2>/dev/null | wc -l)" -ge "$MAXQ" ]; do
        sleep 60
    done

    jid=$(sbatch --parsable \
          --partition="$PART" --time=00:40:00 --array=1-${sz}%1000 \
          --output="$RUN_DIR/logs/piv_%A_%a.out" --error="$RUN_DIR/logs/piv_%A_%a.err" \
          --export=ALL,RUN_DIR="$RUN_DIR",NT="$NT",BASE_SEED="$off",PROJ="$PROJ",KEEP_COMBINED="$KEEP_COMBINED" \
          "$PROJ/unity/run_array.sbatch" 2>>"$RUN_DIR/logs/submit.err")
    if [ -n "${jid:-}" ]; then
        echo "[chunked] $(date '+%T') seeds $((off+1))-$((off+sz)) -> job $jid"
        jids="${jids}${jid}:"
        a=$((b + 1))
    else
        echo "[chunked] $(date '+%T') submit failed (queue full / transient); retry in 120s"
        sleep 120
    fi
done

# --- finalize after ALL chunks finish (manifest + 2% preview) ---
dep="afterany:${jids%:}"
sbatch --partition="$PART" --dependency="$dep" \
       --output="$RUN_DIR/logs/finalize_%j.out" --error="$RUN_DIR/logs/finalize_%j.err" \
       --export=ALL,RUN_DIR="$RUN_DIR",PROJ="$PROJ" \
       "$PROJ/unity/finalize_run.sbatch"

echo "[chunked] ALL $N submitted across chunks. finalize queued ($dep)."
echo "[chunked] run folder: $RUN_DIR"
