#!/bin/bash
# Finalize one campaign: build the Parquet manifest + render the 2% visual preview.
# Auto-submitted by unity/submit_run.sh with --dependency=afterany on the array job,
# so it runs once the simulations finish. Sets RUN_DIR/PROJ via --export.
#SBATCH --job-name=piv_finalize
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00

set -euo pipefail

: "${RUN_DIR:?RUN_DIR not set}"
: "${PROJ:?PROJ not set}"

# Python env with pandas, pyarrow, tomli, h5py, matplotlib, numpy.
# One-time setup (see unity/README.md):
#   python -m venv /work/pi_nicholas_pizzo_uri_edu/arup_mazumder/piv_venv
#   source .../piv_venv/bin/activate && pip install pandas pyarrow tomli h5py matplotlib numpy
PIV_VENV="${PIV_VENV:-/work/pi_nicholas_pizzo_uri_edu/arup_mazumder/piv_venv}"
if [ -f "$PIV_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$PIV_VENV/bin/activate"
else
    echo "WARNING: no venv at $PIV_VENV — falling back to 'python' on PATH (deps may be missing)"
fi

echo "[finalize] $(date '+%F %T')  run=$RUN_DIR"
echo "[finalize] samples found: pix10=$(ls "$RUN_DIR"/pix10 2>/dev/null | wc -l)  metadata=$(ls "$RUN_DIR"/metadata 2>/dev/null | wc -l)"

echo "[finalize] building manifest.parquet ..."
python "$PROJ/scripts/build_manifest.py" --in "$RUN_DIR/metadata" --out "$RUN_DIR/manifest.parquet"

echo "[finalize] rendering 2% preview ..."
python "$PROJ/scripts/render_preview.py" --root "$RUN_DIR" --frac 0.02

echo "[finalize] done -> $RUN_DIR"
