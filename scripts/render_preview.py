#!/usr/bin/env python3
"""
Render a random fraction of image pairs as PNGs for visual QA.

For each picked sample it makes a 2x2 panel:
    Image A          |  Image B
    speed @ A (heat) |  speed @ B (heat)
so you can eyeball that the particle images look sensible and the velocity field
matches. Output goes to <root>/preview/.

Usage:
    python render_preview.py --root <RUN_DIR> [--frac 0.02] [--seed 0]

Deps: numpy, h5py, matplotlib.
"""
import argparse
import glob
import os
import random

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")          # headless (no display on compute nodes)
import matplotlib.pyplot as plt


def render_one(path, out_png):
    with h5py.File(path, "r") as f:
        pk = "pairs/" + sorted(f["pairs"].keys())[0]   # first pair (robust to its index)
        A = f[f"{pk}/A"][:]
        B = f[f"{pk}/B"][:]
        uA, vA = f[f"{pk}/fields/uA"][:], f[f"{pk}/fields/vA"][:]
        uB, vB = f[f"{pk}/fields/uB"][:], f[f"{pk}/fields/vB"][:]

    spdA = np.sqrt(uA**2 + vA**2)
    spdB = np.sqrt(uB**2 + vB**2)

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax[0, 0].imshow(A, cmap="gray", origin="lower");  ax[0, 0].set_title("Image A")
    ax[0, 1].imshow(B, cmap="gray", origin="lower");  ax[0, 1].set_title("Image B")
    i2 = ax[1, 0].imshow(spdA, cmap="viridis", origin="lower"); ax[1, 0].set_title("speed @ A")
    i3 = ax[1, 1].imshow(spdB, cmap="viridis", origin="lower"); ax[1, 1].set_title("speed @ B")
    fig.colorbar(i2, ax=ax[1, 0], fraction=0.046, pad=0.04)
    fig.colorbar(i3, ax=ax[1, 1], fraction=0.046, pad=0.04)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(os.path.basename(path), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Render a random %% of image pairs to PNG for QA.")
    ap.add_argument("--root", required=True, help="run folder (contains pix10/ pix20/ pix30/)")
    ap.add_argument("--frac", type=float, default=0.02, help="fraction of pairs to render (default 0.02 = 2%%)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the random pick (reproducible)")
    ap.add_argument("--bins", default="pix10,pix20,pix30")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    outdir = os.path.join(args.root, "preview")
    os.makedirs(outdir, exist_ok=True)

    files = []
    for b in args.bins.split(","):
        files += sorted(glob.glob(os.path.join(args.root, b, "*.jld2")))
    if not files:
        print(f"No sample files found under {args.root}")
        return

    k = max(1, round(len(files) * args.frac))
    picked = rng.sample(files, min(k, len(files)))
    print(f"Rendering {len(picked)} / {len(files)} pairs ({args.frac:.0%}) -> {outdir}")

    bad = 0
    for p in picked:
        rel = os.path.relpath(p, args.root).replace(os.sep, "__").replace(".jld2", ".png")
        try:
            render_one(p, os.path.join(outdir, rel))
        except Exception as e:
            bad += 1
            print(f"  ! {os.path.basename(p)}: {e}")
    print(f"Done ({len(picked) - bad} written" + (f", {bad} failed)" if bad else ")"))


if __name__ == "__main__":
    main()
