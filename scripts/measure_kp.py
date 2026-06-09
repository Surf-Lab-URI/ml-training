#!/usr/bin/env python3
"""
Run-length calibration for the 2D-turbulence PIV sims.

Answers "how long should the simulation run before we sample a pair?" using the
eddy-turnover time (notes/DATA_GENERATION_DESIGN.md §3):

    eddy-turnover time   tau(t) = 1 / (U_max * k_p)
    centroid wavenumber  k_p    = sqrt( sum(|k|^2 E(k)) / sum(E(k)) )   [= sqrt(enstrophy/energy)]
    U_max                       = max |velocity|

For each saved frame it prints U_max, k_p, tau, and the *dimensionless age* t/tau0
(tau0 = eddy-turnover time at t=0). The flow is "developed but not collapsed" over
some window of t/tau0 — read the chosen constant C off the k_p(t) curve and the
vorticity snapshots in the saved figure, then set the run length to t_sample = C/(U*k_p).

Usage:
    python3 scripts/measure_kp.py                       # first *_combined.jld2 in data/binary
    python3 scripts/measure_kp.py path/to/x_combined.jld2
    python3 scripts/measure_kp.py f1.jld2 f2.jld2 ...   # overlay several seeds
"""
import sys, glob
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def frame_keys(f):
    ks = [k for k in f["fields/timeseries/u"].keys() if k != "serialized"]
    return sorted(ks, key=int)


def load_uv(f, k):
    u = np.asarray(f[f"fields/timeseries/u/{k}"])
    v = np.asarray(f[f"fields/timeseries/v/{k}"])
    u = np.squeeze(u); v = np.squeeze(v)
    return u, v


def spectrum_stats(u, v):
    """Return (U_max, k_p, energy, enstrophy) for one velocity field.
    Grid spacing is 1 px (512 grid over 512 extent), so wavenumbers are in rad/px."""
    ny, nx = u.shape
    uh = np.fft.fft2(u); vh = np.fft.fft2(v)
    E2d = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2) / (nx * ny) ** 2  # normalization cancels in k_p
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=1.0)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=1.0)
    KX, KY = np.meshgrid(kx, ky)
    kmag2 = KX ** 2 + KY ** 2
    energy = E2d.sum()
    enstrophy = (kmag2 * E2d).sum()
    k_p = np.sqrt(enstrophy / energy) if energy > 0 else np.nan
    U_max = float(np.sqrt(u ** 2 + v ** 2).max())
    return U_max, float(k_p), float(energy), float(enstrophy)


def vorticity(u, v):
    # finite-difference curl (periodic), purely for the snapshot panels
    dvdx = np.gradient(v, axis=1)
    dudy = np.gradient(u, axis=0)
    return dvdx - dudy


def analyze(path):
    rows = []
    snaps = []  # (t_over_tau0, omega) for a few times
    with h5py.File(path, "r") as f:
        keys = frame_keys(f)
        times = np.array([float(np.asarray(f[f"fields/timeseries/t/{k}"])) for k in keys])
        for i, k in enumerate(keys):
            u, v = load_uv(f, k)
            U, kp, E, Z = spectrum_stats(u, v)
            tau = 1.0 / (U * kp) if (U > 0 and kp > 0) else np.nan
            rows.append((times[i], U, kp, E, Z, tau))
        tau0 = rows[0][5]
        # grab ~6 evenly spaced snapshots for the figure
        idx = np.linspace(0, len(keys) - 1, min(6, len(keys))).astype(int)
        for j in idx:
            u, v = load_uv(f, keys[j])
            snaps.append((times[j] / tau0, vorticity(u, v)))
    return np.array(rows), tau0, snaps


def main():
    args = [a for a in sys.argv[1:]]
    if not args:
        args = sorted(glob.glob("data/binary/*_combined.jld2"))
        if not args:
            sys.exit("No *_combined.jld2 in data/binary/ — run a sim with --no_image_gen first.")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    print(f"{'file':<28} {'t':>7} {'U_max':>7} {'k_p':>8} {'tau':>7} {'t/tau0':>7}")
    print("-" * 70)

    for ci, path in enumerate(args):
        rows, tau0, snaps = analyze(path)
        name = Path(path).name[:26]
        t, U, kp, E, Z, tau = rows.T
        for r in rows:
            print(f"{name:<28} {r[0]:7.3f} {r[1]:7.2f} {r[2]:8.4f} {r[5]:7.3f} {r[0]/tau0:7.2f}")
        print(f"  -> tau0 = {tau0:.3f}   (eddy-turnover time at t=0)")
        # line plots: k_p(t)/k_p0 and energy(t)/energy0 vs dimensionless age
        age = t / tau0
        axes[0, 0].plot(age, kp / kp[0], marker=".", label=name[:14])
        axes[0, 1].plot(age, E / E[0], marker=".", label=name[:14])
        axes[0, 2].plot(age, U / U[0], marker=".", label=name[:14])
        # vorticity snapshots only for the first file
        if ci == 0:
            for s, (a_over, om) in enumerate(snaps):
                ax = axes[1, s % 3] if s < 3 else axes[1, s % 3]
        # (snapshots handled below for first file)

    axes[0, 0].set(title="k_p(t) / k_p(0)", xlabel="t / tau0", ylabel="relative k_p")
    axes[0, 1].set(title="energy(t) / energy(0)", xlabel="t / tau0")
    axes[0, 2].set(title="U_max(t) / U_max(0)", xlabel="t / tau0")
    for ax in axes[0]:
        ax.axhline(1.0, color="grey", lw=0.5); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # bottom row: vorticity snapshots of the first file, labeled by t/tau0
    rows0, tau0_0, snaps0 = analyze(args[0])
    pick = [snaps0[i] for i in np.linspace(0, len(snaps0) - 1, 3).astype(int)]
    for s, (a_over, om) in enumerate(pick):
        ax = axes[1, s]
        m = np.abs(om).max()
        ax.imshow(om, cmap="RdBu_r", vmin=-m, vmax=m, origin="lower")
        ax.set(title=f"vorticity @ t/tau0 = {a_over:.1f}")
        ax.set_xticks([]); ax.set_yticks([])

    out = "data/binary/calibration_kp.png"
    fig.suptitle("Run-length calibration: pick C where flow is developed but not collapsed", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\nFigure saved: {out}")
    print("Read the chosen C off the curves/snapshots; run length = C / (U_max * k_p).")


if __name__ == "__main__":
    main()
