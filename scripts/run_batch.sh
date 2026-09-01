#!/usr/bin/env bash
# Run N simulations one after another, in the current shell. Portable: no machine-specific paths.
#
#   ./scripts/run_batch.sh [N] [NT]
#     N   number of simulations, seeds BASE_SEED+1 .. BASE_SEED+N   (default: [run].n_sims)
#     NT  frames recorded per simulation                            (default: [run].nt)
#
# WHEN TO USE THIS, AND WHEN NOT TO
#
#   Use it on your laptop, or inside a single interactive Unity session, to produce a handful of
#   simulations you want to look at:
#
#       srun --partition=uri-cpu --time=02:00:00 --mem=8G --cpus-per-task=4 --pty bash
#       ./scripts/run_batch.sh 5 40
#
#   Do NOT use it for a real campaign. It is serial — one simulation at a time — so a thousand
#   simulations would take days. On Unity use unity/submit_run.sh or unity/submit_chunked.sh,
#   which fan the same work out across the cluster.
#
# Everything not passed on the command line comes from params.toml, so this driver and the Slurm
# submitters cannot disagree about what a "default run" is.
set -uo pipefail        # not -e: one failed simulation must not abandon the rest of the batch

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

# --- Resolve julia -----------------------------------------------------------------------------
# $HOME is NFS-mounted on Unity and is occasionally unresolvable at the moment a job starts, which
# makes julia vanish from PATH. Same retry the Slurm job scripts use, for the same reason.
export PATH="$HOME/.juliaup/bin:$PATH"
JULIA="${JULIA:-}"
for _try in 1 2 3 4 5; do
    [ -n "$JULIA" ] && [ -x "$JULIA" ] && break
    JULIA=$(command -v julia 2>/dev/null || true)
    [ -n "$JULIA" ] && [ -x "$JULIA" ] && break
    echo "[warn] julia not on PATH (attempt $_try) — waiting for \$HOME to settle"
    sleep 5
    export PATH="$HOME/.juliaup/bin:$PATH"
done
if [ -z "$JULIA" ] || [ ! -x "$JULIA" ]; then
    echo "[fatal] julia not found. Install juliaup and add it to PATH:"
    echo "        export PATH=\"\$HOME/.juliaup/bin:\$PATH\""
    echo "        (or set JULIA=/full/path/to/julia)"
    exit 1
fi

# --- Defaults from params.toml -----------------------------------------------------------------
eval "$("$JULIA" --project="$REPO" "$REPO/scripts/params_export.jl" 2>/dev/null || true)"
N="${1:-${PIV_N_SIMS:-20}}"
NT="${2:-${PIV_NT:-40}}"
BASE_SEED="${PIV_BASE_SEED:-0}"

# Keep the depot off $HOME for the same reason as above, when params.toml names one.
[ -n "${PIV_JULIA_DEPOT:-}" ] && export JULIA_DEPOT_PATH="$PIV_JULIA_DEPOT"

LOGDIR="${PIV_LOG_DIR:-$REPO/logs}"
mkdir -p "$LOGDIR"

echo "=== BATCH START $(date '+%F %T %Z') ==="
echo "    julia   $JULIA"
echo "    repo    $REPO"
echo "    sims    $N   (seeds $((BASE_SEED + 1))..$((BASE_SEED + N)))   nt=$NT"
echo "    logs    $LOGDIR"
echo "    NOTE    serial driver — use unity/submit_run.sh for anything large"

fail=0
for i in $(seq 1 "$N"); do
    s=$((BASE_SEED + i))
    printf '=== sim %d/%d  seed=%d  START %s\n' "$i" "$N" "$s" "$(date '+%T')"
    "$JULIA" --project="$REPO" "$REPO/scripts/2DTurbulence.jl" --seed "$s" --nt "$NT" \
        > "$LOGDIR/sim_seed_${s}.log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        fail=$((fail + 1))
        echo "    FAILED rc=$rc — see $LOGDIR/sim_seed_${s}.log"
    fi
    printf '=== sim %d/%d  seed=%d  DONE rc=%d %s\n' "$i" "$N" "$s" "$rc" "$(date '+%T')"
done

echo "=== BATCH COMPLETE $(date '+%F %T %Z') — $((N - fail))/$N succeeded ==="
[ "$fail" -eq 0 ] || echo "    $fail failed; check $LOGDIR"
exit $(( fail > 0 ? 1 : 0 ))
