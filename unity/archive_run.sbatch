#!/bin/bash
# Archive a FINISHED campaign from fast /scratch4 to permanent /project as ONE big tar.
# Reads the many small files from fast scratch and writes a single sequential file to
# /project — the access pattern /project handles well (avoids choking its metadata
# server with hundreds of thousands of tiny files).
#
# Run on the NON-preempt 'cpu' partition so the archive can't be killed midway.
#
# Usage:
#   sbatch unity/archive_run.sbatch /scratch4/workspace/arup_mazumder_uri_edu-piv_dataset/piv_2dturb_dataset/run_2026-06-12_XX-XX-XX
#
# Optional: override the destination with DEST=... in front of sbatch.
#
#SBATCH --job-name=piv_archive
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=archive_%j.out
#SBATCH --error=archive_%j.err

set -euo pipefail

RUN="${1:?usage: sbatch archive_run.sbatch <path-to-run-folder>}"
RUN="${RUN%/}"                       # strip any trailing slash
DEST="${DEST:-/project/pi_nicholas_pizzo_uri_edu/arup/piv_2dturb_dataset}"

name="$(basename "$RUN")"
parent="$(dirname "$RUN")"
out="$DEST/${name}.tar"

[ -d "$RUN" ] || { echo "ERROR: run folder not found: $RUN"; exit 1; }
mkdir -p "$DEST"

echo "[archive] $(date '+%F %T')  src=$RUN"
echo "[archive] counting source files ..."
nsrc=$(find "$RUN" -type f | wc -l)
echo "[archive] $nsrc files to archive -> $out"

# plain tar (NO gzip: .jld2 are already binary; gzip would be slow for ~no gain)
tar -C "$parent" -cf "$out" "$name"

echo "[archive] verifying member count ..."
ntar=$(tar -tf "$out" | grep -c -v '/$' || true)   # count file members (skip dir entries)
echo "[archive] source files=$nsrc  tar members=$ntar  size=$(du -h "$out" | cut -f1)"
if [ "$nsrc" -ne "$ntar" ]; then
    echo "[archive] WARNING: count mismatch — inspect before deleting scratch!"
else
    echo "[archive] OK: counts match. Archive at $out"
fi
echo "[archive] done $(date '+%F %T')"
echo "[archive] NOTE: scratch copy left INTACT. Delete it only after you confirm the tar."
