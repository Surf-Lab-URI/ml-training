#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np


def render_particles_gaussian(x, y,
                              width=256, height=256,
                              xlim=(0.0, 255),
                              ylim=(0.0, 255),
                              sigma_px=1.2,
                              peak=1.0,
                              background=0.0):
    img = np.full((height, width), background, dtype=np.float32)

    xmin, xmax = xlim
    ymin, ymax = ylim

    u = (x - xmin) / (xmax - xmin) * (width - 1)
    v = (y - ymin) / (ymax - ymin) * (height - 1)

    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)

    radius = max(1, int(np.ceil(3.0 * sigma_px)))
    grid = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-(grid ** 2) / (2 * sigma_px ** 2))
    kernel = np.outer(g, g)
    kernel /= kernel.max()
    kernel *= peak

    ksz = kernel.shape[0]

    for ui, vi in zip(u, v):
        cx = int(round(ui))
        cy = int(round(vi))

        x0 = cx - radius
        y0 = cy - radius
        x1 = x0 + ksz
        y1 = y0 + ksz

        ix0 = max(0, x0)
        iy0 = max(0, y0)
        ix1 = min(width, x1)
        iy1 = min(height, y1)

        kx0 = ix0 - x0
        ky0 = iy0 - y0
        kx1 = kx0 + (ix1 - ix0)
        ky1 = ky0 + (iy1 - iy0)

        img[iy0:iy1, ix0:ix1] += kernel[ky0:ky1, kx0:kx1]

    return img


def to_uint8(img, clip_max=3.0):
    img = np.clip(img, 0.0, clip_max) / clip_max
    return (img * 255.0).astype(np.uint8)


def _fit_field_2d(arr, Ny, Nx):
    """Return a 2D array resized to (Ny, Nx) by center-cropping or edge-padding.

    This handles minor off-by-one differences in stored field sizes between outputs.
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        a = np.squeeze(a)
        if a.ndim != 2:
            raise ValueError("Expected 2D field, got shape {}".format(arr.shape))

    inNy, inNx = a.shape

    # Center-crop if needed
    if inNy > Ny:
        start = (inNy - Ny) // 2
        a = a[start:start + Ny, :]
        inNy = Ny
    if inNx > Nx:
        start = (inNx - Nx) // 2
        a = a[:, start:start + Nx]
        inNx = Nx

    # Edge-pad if needed
    pad_y = max(0, Ny - inNy)
    pad_x = max(0, Nx - inNx)
    if pad_y or pad_x:
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
        a = np.pad(a, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")

    # Final safety
    if a.shape != (Ny, Nx):
        raise ValueError("Field resize failed: got {}, expected {}".format(a.shape, (Ny, Nx)))

    return a


def build_dataset_from_npz(
    input_npz,
    out_dir="ml_dataset",
    k=2000,
    seed=42,
    width=256,
    height=256,
    xlim=(0.0, 255),
    ylim=(0.0, 255),
    sigma_px=1.2,
    peak=1.0,
    clip_max=3.0,
    max_pairs=100,
    dt = 0.1,
    Dt = 0.1
):
    """Generate particle image pairs plus pair-aligned particle positions and background fields.

    Expects `input_npz` to contain at least:
      - x: (n_frames, n_particles)
      - y: (n_frames, n_particles)
      - t: (n_frames,)
      - field_a: (n_frames, Ny, Nx)
      - field_b: (n_frames, Ny, Nx)

    Writes to `out_dir`:
      - images/pair_000000_a.png and images/pair_000000_b.png
      - positions_pairs.npy float32 (n_pairs, 2, k, 2)        # [A/B, particle, x/y]
      - times_pairs.npy     float32 (n_pairs, 2)              # [tA, tB]
      - field_pairs.npy     float32 (n_pairs, 2, 2, Ny, Nx)   # [A/B, field_a/field_b, y, x]
      - idx.npy             int32   (k,)                      # particle indices used
      - dataset.npz         compressed bundle of the arrays above
      - meta.json
    """

    import imageio

    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    data = np.load(input_npz)
    X = data["x"]
    Y = data["y"]
    t = data["t"]
    field_a = data["field_a"]
    field_b = data["field_b"]

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("x and y must be 2D arrays (n_frames, n_particles)")
    if t.ndim != 1:
        raise ValueError("t must be 1D array (n_frames,)")
    if field_a.ndim != 3 or field_b.ndim != 3:
        raise ValueError("field_a and field_b must be 3D arrays (n_frames, Ny, Nx)")

    n_frames, n_particles = X.shape
    if Y.shape != (n_frames, n_particles):
        raise ValueError("y shape does not match x")
    if t.shape[0] != n_frames:
        raise ValueError("t length does not match number of frames")
    if field_a.shape[0] != n_frames or field_b.shape[0] != n_frames:
        raise ValueError("field_a/field_b first dimension must match n_frames")

    n_pairs = n_frames // 2

    # Choose fixed random subsample indices once
    rng = np.random.RandomState(seed)
    k_eff = min(int(k), int(n_particles))
    idx = rng.choice(n_particles, size=k_eff, replace=False).astype(np.int32)

    if max_pairs is not None:
        n_pairs = min(n_pairs, int(max_pairs))

    Ny = max(int(field_a.shape[1]), int(field_b.shape[1]))
    Nx = max(int(field_a.shape[2]), int(field_b.shape[2]))

    field_pairs = np.empty((n_pairs, 2, 2, Ny, Nx), dtype=np.float32)

    for p in range(n_pairs):
        # Image pairs
        fA = p
        fB = p + int(round(Dt/dt))

        if fB >= n_frames:
            break

        xA = X[fA, idx].astype(np.float32, copy=False)
        yA = Y[fA, idx].astype(np.float32, copy=False)
        xB = X[fB, idx].astype(np.float32, copy=False)
        yB = Y[fB, idx].astype(np.float32, copy=False)

        fa0 = _fit_field_2d(field_a[fA], Ny, Nx).astype(np.float32, copy=False)
        fb0 = _fit_field_2d(field_b[fA], Ny, Nx).astype(np.float32, copy=False)
        fa1 = _fit_field_2d(field_a[fB], Ny, Nx).astype(np.float32, copy=False)
        fb1 = _fit_field_2d(field_b[fB], Ny, Nx).astype(np.float32, copy=False)

        # writing fields to big array. For PIV fa0 is u at initial time, 
        # fb0 is v at initial time, etc. and they are muliplied by int(round(Dt/dt))*dt
        # so that the units are pixels per pair, assuming unit of length was already pixels.
        field_pairs[p, 0, 0] = fa0*(fB-fA)*dt
        field_pairs[p, 0, 1] = fb0*(fB-fA)*dt
        field_pairs[p, 1, 0] = fa1*(fB-fA)*dt
        field_pairs[p, 1, 1] = fb1*(fB-fA)*dt

        imgA = render_particles_gaussian(
            xA, yA,
            width=width, height=height,
            xlim=xlim, ylim=ylim,
            sigma_px=sigma_px, peak=peak
        )
        imgB = render_particles_gaussian(
            xB, yB,
            width=width, height=height,
            xlim=xlim, ylim=ylim,
            sigma_px=sigma_px, peak=peak
        )

        uA_img = to_uint8(imgA, clip_max=clip_max)
        uB_img = to_uint8(imgB, clip_max=clip_max)

        imageio.imwrite(os.path.join(images_dir, "pair_{:06d}_a.png".format(p)), uA_img)
        imageio.imwrite(os.path.join(images_dir, "pair_{:06d}_b.png".format(p)), uB_img)
        np.save(os.path.join(images_dir, "pair_{:06d}_flow.npy".format(p)), field_pairs[p])

    np.save(os.path.join(out_dir, "field_pairs.npy"), field_pairs)

    np.savez_compressed(
        os.path.join(out_dir, "dataset.npz"),
        field_pairs=field_pairs
    )

    meta = {
        "input_npz": input_npz,
        "n_frames": int(n_frames),
        "n_particles": int(n_particles),
        "k": int(k_eff),
        "seed": int(seed),
        "n_pairs": int(n_pairs),
        "image_width": int(width),
        "image_height": int(height),
        "xlim": [float(xlim[0]), float(xlim[1])],
        "ylim": [float(ylim[0]), float(ylim[1])],
        "sigma_px": float(sigma_px),
        "peak": float(peak),
        "clip_max": float(clip_max),
        "field_shape": [int(Ny), int(Nx)],
        "field_pairs_shape": list(field_pairs.shape)
    }

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Done. Wrote {} pairs to '{}/' (k={}).".format(n_pairs, out_dir, k_eff))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate particle image pairs + aligned fields from a combined NPZ."
    )
    ap.add_argument("input_npz", help="Combined NPZ from load_jld2_particles.py (contains x,y,t,field_a,field_b)")
    ap.add_argument("--out_dir", default="ml_dataset")
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--x0", type=float, default=0.0)
    ap.add_argument("--x1", type=float, default=255)
    ap.add_argument("--y0", type=float, default=0.0)
    ap.add_argument("--y1", type=float, default=255)
    ap.add_argument("--sigma_px", type=float, default=1.2)
    ap.add_argument("--peak", type=float, default=1.0)
    ap.add_argument("--clip_max", type=float, default=3.0)
    ap.add_argument("--max_pairs", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.1, help="simulation output time step")
    ap.add_argument("--Dt", type=float, default=0.1, help="defined time step between A and B in the image pair")
    args = ap.parse_args()

    build_dataset_from_npz(
        args.input_npz,
        out_dir=args.out_dir,
        k=args.k,
        seed=args.seed,
        width=args.width,
        height=args.height,
        xlim=(args.x0, args.x1),
        ylim=(args.y0, args.y1),
        sigma_px=args.sigma_px,
        peak=args.peak,
        clip_max=args.clip_max,
        max_pairs=args.max_pairs,
        dt=args.dt,
        Dt=args.Dt
    )

# to run:
# python image_gen.py combined.npz --dt *simulation time step* --Dt *desired time step between A and B in the image pair* 