#!/usr/bin/env python3
"""What the real tank actually contains — measured from the XCPIV collection, not assumed.

Every requirement in DATA_REQUIREMENTS.md that talks about the laboratory should be traceable to a
number this script prints. It reads the traditional-PIV output for many frames (200 are available,
against the 4 that have hand-tracked points) and reports the displacement profile with depth, the
surface geometry, and the data-quality flags.

    python scripts/characterize_tank.py --n 60
    python scripts/characterize_tank.py --n 200 --dir /path/to/XCPIV/ExpLCL_1_03-200

Needs h5py and numpy. On Unity the collection lives at
/project/pi_nicholas_pizzo_uri_edu/arup/piv_lab_datasets/XCPIV/.
"""
import argparse, glob, sys
import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("needs h5py:  pip install h5py")

DEPTH_EDGES = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 40]

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/project/pi_nicholas_pizzo_uri_edu/arup/piv_lab_datasets/"
                                 "XCPIV/ExpLCL_1_03-200")
ap.add_argument("--n", type=int, default=60)
a = ap.parse_args()

files = sorted(glob.glob(a.dir + "/*_PIV.mat"))[:a.n]
if not files:
    sys.exit(f"no *_PIV.mat under {a.dir}")

per_frame, prof, bad = [], {i: [] for i in range(len(DEPTH_EDGES) - 1)}, 0
for f in files:
    try:
        with h5py.File(f, "r") as h:
            cv = h["compVel"]
            DX_MM = float(np.array(cv["DX"]).ravel()[0]) * 1000
            DT = float(np.array(cv["DT"]).ravel()[0])
            # delta_x (511^2) is the PIV's native grid; delta_x1 is the same field interpolated to
            # 2048^2 and 16x the I/O for no extra information.
            dx = np.array(cv["delta_x"]).astype(float).T
            dz = np.array(cv["delta_z"]).astype(float).T
            water = np.array(cv["mask"]).astype(float).T > 0.5
            dcor = np.array(cv["dcor"]).astype(float).T
            xP = np.array(cv["xPIV"]).ravel(); zP = np.array(cv["zPIV"]).ravel()
            sa = np.array(h["imSurfa"]["surfacePIVImg"]).astype(float).ravel()
            sb = np.array(h["imSurfb"]["surfacePIVImg"]).astype(float).ravel()
            bad += int(bool(np.array(h["imSurfa"]["badFrameBool"]).ravel()[0]) or
                       bool(np.array(h["imSurfb"]["badFrameBool"]).ravel()[0]))
    except (OSError, KeyError):
        continue

    col = np.clip(np.round(xP).astype(int), 0, len(sa) - 1)
    depth = (zP[:, None] - sa[col][None, :]) * DX_MM        # mm below the interface
    mag = np.hypot(dx, dz)
    ok = water & np.isfinite(mag)
    t2 = ok & (depth >= 0) & (depth < 2)
    slope = np.abs(np.gradient(sa))

    per_frame.append(dict(
        med=np.nanmedian(mag[ok]), p99=np.nanpercentile(mag[ok], 99), mx=np.nanmax(mag[ok]),
        t2med=np.nanmedian(mag[t2]) if t2.sum() else np.nan,
        t2max=np.nanmax(mag[t2]) if t2.sum() else np.nan,
        air=100 * (1 - water.mean()),
        nan_all=100 * np.isnan(dx[water]).mean(),
        nan_t2=100 * np.isnan(dx[water & (depth >= 0) & (depth < 2)]).mean(),
        slope50=np.percentile(slope, 50), slope99=np.percentile(slope, 99),
        dsurf=np.nanmean(np.abs(sb - sa)), dsurf_max=np.nanmax(np.abs(sb - sa)),
        dcor=np.nanmedian(dcor[water]), DX=DX_MM, DT=DT))
    for i, (lo, hi) in enumerate(zip(DEPTH_EDGES[:-1], DEPTH_EDGES[1:])):
        s = ok & (depth >= lo) & (depth < hi)
        if s.sum() > 30:
            prof[i].append(np.nanmedian(mag[s]))

R = {k: np.array([p[k] for p in per_frame]) for k in per_frame[0]}
print(f"frames: {len(per_frame)} of {len(files)}   badFrameBool set: {bad}")
print(f"DX = {R['DX'][0]:.4f} mm/px   DT = {R['DT'][0]:.4f} s\n")

print("DISPLACEMENT PROFILE WITH DEPTH  (median px, averaged over frames)")
prof_v = []
for i, (lo, hi) in enumerate(zip(DEPTH_EDGES[:-1], DEPTH_EDGES[1:])):
    if prof[i]:
        v = float(np.mean(prof[i])); prof_v.append((lo, hi, v))
        print(f"   {f'{lo}-{hi} mm':>10s}  {v:6.2f}")
if prof_v:
    print(f"\n   surface layer is {prof_v[0][2]/prof_v[-1][2]:.0f}x faster than the deepest bin")

print("\nPER-FRAME SUMMARY                     mean     min     max")
for k, lab in [("med", "displacement median (frame)"), ("p99", "displacement p99"),
               ("mx", "displacement max"), ("t2med", "TOP 2mm displacement median"),
               ("t2max", "TOP 2mm displacement max"), ("slope50", "surface slope median"),
               ("slope99", "surface slope p99"), ("dsurf", "surface motion A->B mean (px)"),
               ("dsurf_max", "surface motion A->B max (px)"), ("air", "air fraction (%)"),
               ("nan_all", "PIV rejected, all water (%)"),
               ("nan_t2", "PIV rejected, top 2mm (%)"), ("dcor", "median correlation")]:
    v = R[k][np.isfinite(R[k])]
    print(f"  {lab:34s} {v.mean():7.2f} {v.min():7.2f} {v.max():7.2f}")
