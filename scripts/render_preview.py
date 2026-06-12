#!/usr/bin/env python3
"""
Render a random fraction of image pairs as PNGs for visual QA, with panels that make
the particle DISPLACEMENT between the two frames obvious.

Each picked sample -> a 2x3 panel:
    Image A         | Image B          | Overlay (A=red, B=green)
    Zoom of overlay | |displacement|   | displacement quiver (arrows = pixels)
Stationary particles look yellow in the overlay; moved ones show a red->green offset.
The stored fields uA,vA are already the displacement in PIXELS (velocity x dt_pair),
so |sqrt(uA^2+vA^2)| is the per-point displacement and the quiver arrows are 1:1 in px.

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_one(path, out_png):
    with h5py.File(path, "r") as f:
        pk = "pairs/" + sorted(f["pairs"].keys())[0]
        A = f[f"{pk}/A"][:].astype(float)
        B = f[f"{pk}/B"][:].astype(float)
        uA = f[f"{pk}/fields/uA"][:].astype(float)   # already displacement in PIXELS
        vA = f[f"{pk}/fields/vA"][:].astype(float)

    H, W = A.shape
    An = A / max(A.max(), 1.0)            # normalize images to [0,1]
    Bn = B / max(B.max(), 1.0)

    disp = np.sqrt(uA**2 + vA**2)         # per-point displacement magnitude (px)
    dmax, dmean = float(disp.max()), float(disp.mean())

    # red/green overlay: A in red channel, B in green channel
    overlay = np.zeros((H, W, 3))
    overlay[..., 0] = An
    overlay[..., 1] = Bn

    # zoom crop (centered) so individual red/green particle offsets are visible
    cs = 110
    y0, x0 = H // 2 - cs // 2, W // 2 - cs // 2
    crop = overlay[y0:y0 + cs, x0:x0 + cs]

    # quiver downsample
    step = 28
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    U, V = uA[::step, ::step], vA[::step, ::step]

    fig, ax = plt.subplots(2, 3, figsize=(13.5, 9))
    ax[0, 0].imshow(An, cmap="gray", origin="lower"); ax[0, 0].set_title("Image A")
    ax[0, 1].imshow(Bn, cmap="gray", origin="lower"); ax[0, 1].set_title("Image B")
    ax[0, 2].imshow(overlay, origin="lower"); ax[0, 2].set_title("Overlay  (A=red, B=green)")

    ax[1, 0].imshow(crop, origin="lower",
                    extent=[x0, x0 + cs, y0, y0 + cs])
    ax[1, 0].set_title(f"Zoom {cs}px  (A=red, B=green)")

    im = ax[1, 1].imshow(disp, cmap="viridis", origin="lower")
    ax[1, 1].set_title(f"|displacement| (px)   max≈{dmax:.1f}")
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046, pad=0.04)

    ax[1, 2].imshow(disp, cmap="gray", origin="lower")
    ax[1, 2].quiver(xs, ys, U, V, color="deepskyblue",
                    angles="xy", scale_units="xy", scale=1.0, width=0.004)
    ax[1, 2].set_title("displacement field (arrows = px)")

    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{os.path.basename(path)}    displacement: max≈{dmax:.1f}px, mean≈{dmean:.1f}px",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=85)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Render a random %% of image pairs to PNG (displacement-focused QA).")
    ap.add_argument("--root", required=True, help="run folder (contains pix10/ pix20/ pix30/)")
    ap.add_argument("--frac", type=float, default=0.02, help="fraction of pairs to render (default 0.02)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the random pick")
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
