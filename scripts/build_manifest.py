#!/usr/bin/env python3
"""
Build a single Parquet manifest from the per-sample metadata TOML sidecars.

Each simulation writes its own `metadata_*.toml` (parallel-safe). This script
runs ONCE after the campaign, flattens every sidecar into one row, and writes a
single columnar table for fast filtering later.

Usage:
    python scripts/build_manifest.py [--in dataset/metadata] [--out dataset/manifest.parquet]

Dependencies: pandas + pyarrow (`pip install pandas pyarrow`), and a TOML reader
(`tomllib` is built into Python >= 3.11; otherwise `pip install tomli`).
"""
import argparse
import glob
import os
import sys

try:
    import tomllib  # Python >= 3.11
    def load_toml(path):
        with open(path, "rb") as f:
            return tomllib.load(f)
except ModuleNotFoundError:
    import tomli  # pip install tomli
    def load_toml(path):
        with open(path, "rb") as f:
            return tomli.load(f)

import pandas as pd

# physics keys that appear in BOTH ic_physics and sampling_final → suffix to disambiguate
_DUP = ("U_max", "energy", "enstrophy", "k_p")


def flatten(meta, path):
    """Flatten one sidecar's nested TOML sections into a single flat row dict."""
    row = {}
    row.update(meta.get("reproducibility", {}))
    row.update(meta.get("ic_spec", {}))
    for k, v in meta.get("ic_physics", {}).items():
        row[f"{k}_ic" if k in _DUP else k] = v          # e.g. U_max_ic; dt_save stays
    for k, v in meta.get("sampling_final", {}).items():
        row[f"{k}_final" if k in _DUP else k] = v        # e.g. U_max_final; t_final/C_achieved stay
    row.update(meta.get("particles", {}))
    row.update(meta.get("displacement", {}))   # smax, dp_pix10/20/30, realized_px_pix10/20/30
    # Dataset-relative paths to this sample's files, derived from the seed (the sbatch
    # names everything by seed). This is the join: filter the manifest -> read these paths.
    seed = row.get("seed")
    if seed is not None:
        row["sample_pix10"] = f"pix10/seed{seed}_pix10.jld2"
        row["sample_pix20"] = f"pix20/seed{seed}_pix20.jld2"
        row["sample_pix30"] = f"pix30/seed{seed}_pix30.jld2"
        row["combined_raw"] = f"combined/seed{seed}.jld2"
    row["metadata_file"] = os.path.basename(path)
    return row


def main():
    ap = argparse.ArgumentParser(description="Aggregate per-sample metadata TOMLs into one Parquet manifest.")
    ap.add_argument("--in", dest="indir", default="dataset/metadata",
                    help="directory containing metadata_*.toml (default: dataset/metadata)")
    ap.add_argument("--out", dest="out", default="dataset/manifest.parquet",
                    help="output Parquet path (default: dataset/manifest.parquet)")
    ap.add_argument("--csv", action="store_true",
                    help="also write a .csv alongside the Parquet (handy for a quick look)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "*.toml")))
    if not files:
        sys.exit(f"No .toml files found in {args.indir!r}")

    rows, bad = [], 0
    for p in files:
        try:
            rows.append(flatten(load_toml(p), p))
        except Exception as e:
            bad += 1
            print(f"  ! skipped {os.path.basename(p)}: {e}", file=sys.stderr)

    df = pd.DataFrame(rows)
    if "seed" in df.columns:
        df = df.sort_values("seed").reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out, index=False)          # needs pyarrow
    print(f"Wrote {len(df)} rows x {len(df.columns)} cols -> {args.out}"
          + (f"  ({bad} skipped)" if bad else ""))

    if args.csv:
        csv_path = os.path.splitext(args.out)[0] + ".csv"
        df.to_csv(csv_path, index=False)
        print(f"Also wrote {csv_path}")

    print("\nColumns:")
    print("  " + ", ".join(df.columns))


if __name__ == "__main__":
    main()
