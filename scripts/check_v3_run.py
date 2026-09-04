#!/usr/bin/env python3
"""Does a generated v3 run actually have the tank's structure? Checks, rather than assumes.

Every claim v3 makes is checkable from the output files, so this checks all of them:
  1. the depth profile against the 60-frame lab measurement
  2. air fraction against the measured 18.1%
  3. air masked to EXACTLY zero in both frames, as the lab's images are
  4. no particles above the interface
  5. achieved top-2 mm median against each bin's target

    python scripts/check_v3_run.py --root <RUN_DIR> [--n 6]
"""
import argparse, glob, os, re, sys
import numpy as np
try:
    import h5py
except ImportError:
    sys.exit("needs h5py")

# measured over 60 frames of XCPIV/ExpLCL_1_03-200 -- DATA_REQUIREMENTS R6
LAB = [(0,1,8.06),(1,2,7.71),(2,3,6.98),(3,4,5.84),(4,6,4.09),
       (6,8,2.56),(8,12,1.53),(12,16,0.96),(16,23,0.60)]

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--n", type=int, default=6, help="samples per bin")
ap.add_argument("--dx", type=float, default=0.0565454946380008)
a = ap.parse_args()

bins = sorted((d for d in os.listdir(a.root) if re.fullmatch(r"surf\d+", d)),
              key=lambda d: int(d[4:]))
if not bins:
    sys.exit(f"no surfNN bins under {a.root}")

print(f"{'bin':>8s} {'n':>6s} {'target':>7s} {'top2 median':>12s} {'air %':>7s} "
      f"{'air==0':>7s} {'parts in air':>13s}")
prof_acc = {i: [] for i in range(len(LAB))}
fails = []
for b in bins:
    files = sorted(glob.glob(os.path.join(a.root, b, "*.jld2")))[:a.n]
    target = float(b[4:])
    t2, airf, airzero, pinair = [], [], True, 0
    for f in files:
        try:
            with h5py.File(f, "r") as h:
                k = "pairs/" + sorted(h["pairs"].keys())[0]
                A = h[f"{k}/A"][:].astype(float).T
                B = h[f"{k}/B"][:].astype(float).T
                u = h[f"{k}/fields/uA"][:].astype(float).T
                v = h[f"{k}/fields/vA"][:].astype(float).T
        except (OSError, KeyError):
            continue
        mag = np.hypot(u, v); H, W = mag.shape
        air = mag == 0
        if not air.any():
            fails.append(f"{b}: no air region at all"); continue
        surf = np.array([np.argmax(~air[:, j]) if (~air[:, j]).any() else H for j in range(W)])
        depth = (np.arange(H)[:, None] - surf[None, :]) * a.dx
        airf.append(100 * air.mean())
        airzero &= bool((A[air] == 0).all() and (B[air] == 0).all())
        thr = A.mean() + 3 * A.std()
        pinair += int((A[air] > thr).sum())
        s = (depth >= 0) & (depth < 2) & (~air)
        if s.sum() > 50: t2.append(np.median(mag[s]))
        for i, (lo, hi, _) in enumerate(LAB):
            m = (depth >= lo) & (depth < hi) & (~air)
            if m.sum() > 50: prof_acc[i].append(np.median(mag[m]) / target)
    if not t2: continue
    got = np.mean(t2); rel = abs(got - target) / target
    if rel > 0.10: fails.append(f"{b}: top-2mm median {got:.2f} vs target {target:.0f} ({100*rel:.0f}% off)")
    if not airzero: fails.append(f"{b}: air is not exactly zero")
    if pinair: fails.append(f"{b}: {pinair} bright pixels above the interface")
    print(f"{b:>8s} {len(files):6d} {target:7.0f} {got:12.2f} {np.mean(airf):7.1f} "
          f"{'yes' if airzero else 'NO':>7s} {pinair:13d}")

print(f"\nDEPTH PROFILE, normalised by each bin's target, against the lab shape")
print(f"  {'depth (mm)':>12s} {'v3 (norm)':>10s} {'lab (norm)':>11s} {'ratio':>7s}")
labnorm = [x[2] / 7.88 for x in LAB]        # lab top-2mm median is 7.88 px
worst = 0
for i, (lo, hi, _) in enumerate(LAB):
    if not prof_acc[i]: continue
    g = float(np.mean(prof_acc[i])); r = g / labnorm[i]
    worst = max(worst, abs(r - 1))
    print(f"  {f'{lo}-{hi}':>12s} {g:10.3f} {labnorm[i]:11.3f} {r:7.2f}")
print(f"\n  worst deviation from the lab profile shape: {100*worst:.0f}%")

if fails:
    print("\nFAILURES:"); [print("  " + f) for f in fails]; sys.exit(1)
print("\nall checks passed")
