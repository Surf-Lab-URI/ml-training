#!/usr/bin/env python3
"""
Build a human-readable report on a generated dataset: what is in it, whether it looks right, and
a handful of randomly chosen image pairs with their flow fields.

This is the "did the campaign actually work?" check. Run it after every generation run and look at
it before training anything. It has caught mislabelled bins, off-frame particles and empty images
faster than any automated test we have.

    python scripts/make_report.py --root /path/to/run_v2_2026-08-14_05-26-22
    python scripts/make_report.py --root data --n 8 --format both --out reports/v2_check

--root accepts either a run directory (one that holds pix*/ or med*/ bin folders) or a directory
containing data/visual/ — the local layout a single simulation writes.

Outputs (--format):
    pdf    one self-contained PDF          (default)
    md     Markdown + a folder of PNGs     (for pasting into a wiki or a GitHub issue)
    both   both of the above

Dependencies: numpy, h5py, matplotlib. tomllib (Python >= 3.11) or tomli for the metadata summary.
"""
import argparse
import glob
import os
import random
import re
import sys
import textwrap
from datetime import datetime

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    import tomllib
    def load_toml(p):
        with open(p, "rb") as f:
            return tomllib.load(f)
except ModuleNotFoundError:      # Python < 3.11
    try:
        import tomli
        def load_toml(p):
            with open(p, "rb") as f:
                return tomli.load(f)
    except ModuleNotFoundError:  # metadata summary degrades, images still work
        load_toml = None

BIN_RE = re.compile(r"^(pix|med)\d+$")
SEED_RE = re.compile(r"seed(\d+)")


# ----------------------------------------------------------------------------------------------
# Discovery — find the bins, the samples and the metadata, whichever layout we were pointed at
# ----------------------------------------------------------------------------------------------

def find_bin_dirs(root):
    """Bin directories, in displacement order.

    Handles the three layouts this pipeline produces: a Unity run folder with the bins directly
    underneath, and the two local ones — data/visual/ from the legacy renderer and data/visual_v2/
    from ImageGenV2. v2 is checked first so that a local tree holding both reports the current
    generator's output rather than the legacy one.
    """
    for candidate in (root,
                      os.path.join(root, "data", "visual_v2"),
                      os.path.join(root, "visual_v2"),
                      os.path.join(root, "visual"),
                      os.path.join(root, "data", "visual")):
        if not os.path.isdir(candidate):
            continue
        bins = sorted(
            (d for d in os.listdir(candidate) if BIN_RE.match(d)),
            key=lambda d: int(re.sub(r"\D", "", d)),
        )
        if bins:
            return candidate, bins
    return root, []


def find_metadata(root):
    """Every metadata sidecar we can find, from whichever directory this run used."""
    for sub in ("metadata", "metadata_v2", os.path.join("data", "binary"), "binary", "."):
        hits = sorted(glob.glob(os.path.join(root, sub, "*.toml")))
        if hits:
            return hits
    return []


def seed_of(path):
    """Seed number from a filename, or None when the name does not carry one."""
    m = SEED_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def read_pair(path):
    """First image pair in a .jld2 -> (A, B, uA, vA). uA/vA are ALREADY displacement in pixels."""
    with h5py.File(path, "r") as f:
        key = "pairs/" + sorted(f["pairs"].keys())[0]
        return (f[f"{key}/A"][:].astype(float),
                f[f"{key}/B"][:].astype(float),
                f[f"{key}/fields/uA"][:].astype(float),
                f[f"{key}/fields/vA"][:].astype(float))


# ----------------------------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------------------------

def page_text(title, lines, fontsize=9):
    """A plain text page. Used for the cover and the metadata summary."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.06, 0.95, title, fontsize=16, weight="bold", va="top")
    fig.text(0.06, 0.90, "\n".join(lines), fontsize=fontsize, va="top", family="monospace")
    return fig


def cover_page(root, bins, counts, run_info, params_text, n_meta):
    total = sum(counts.values())
    lines = [
        f"dataset      : {os.path.abspath(root)}",
        f"generated    : {datetime.now().strftime('%Y-%m-%d %H:%M %Z').strip()}",
        f"bins         : {len(bins)}  ({', '.join(bins)})",
        f"samples      : {total:,}   = {total // max(len(bins),1):,} simulations x {len(bins)} bins",
        f"metadata     : {n_meta:,} sidecar file(s)",
        "",
        "SAMPLES PER BIN",
        "  " + "  ".join(f"{b:>7s}" for b in bins),
        "  " + "  ".join(f"{counts[b]:7,d}" for b in bins),
        "",
    ]
    if any(c != counts[bins[0]] for c in counts.values()):
        lines += [
            "  NOTE: the bins do not all hold the same number of samples. That is expected for a",
            "  v2 run — a simulation whose achievable displacement misses a bin's target is SKIPPED",
            "  for that bin rather than written under a label it does not match. A large shortfall",
            "  in one bin means that target is out of reach for this campaign's flows.",
            "",
        ]
    if run_info:
        lines += ["RUN_INFO.txt", *("  " + l for l in run_info.splitlines()), ""]
    if params_text:
        lines += ["params.toml (non-comment lines)", *("  " + l for l in params_text.splitlines())]
    return page_text("Dataset report", lines)


def stats_page(bin_dir, bins, per_bin_files, max_files):
    """Displacement distribution per bin — the single most useful plot in this report.

    The bin names are a PROMISE about displacement (v1: the maximum; v2: the median). This page is
    where you check the promise was kept. A bin whose distribution sits well away from its name
    means the generator could not hit that target and the label is wrong.
    """
    stats, samples = {}, {}
    for b in bins:
        mags = []
        for p in per_bin_files[b][:max_files]:
            try:
                _, _, uA, vA = read_pair(p)
            except (OSError, KeyError):
                continue
            mags.append(np.sqrt(uA**2 + vA**2).ravel())
        if mags:
            allm = np.concatenate(mags)
            samples[b] = allm
            stats[b] = (np.median(allm), np.percentile(allm, 90),
                        np.percentile(allm, 99), allm.max(), len(mags))

    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Displacement distribution by bin", fontsize=15, weight="bold", y=0.97)

    ax = fig.add_axes([0.10, 0.56, 0.84, 0.34])
    for b in bins:
        if b in samples:
            ax.hist(samples[b], bins=120, histtype="step", label=b, density=True, linewidth=1.2)
    ax.set_xlabel("displacement magnitude (px)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(f"pooled over up to {max_files} samples per bin", fontsize=9)

    rows = ["      bin   median      p90      p99      max   files",
            "  " + "-" * 52]
    for b in bins:
        if b in stats:
            m, p90, p99, mx, n = stats[b]
            rows.append(f"  {b:>9s}  {m:7.2f}  {p90:7.2f}  {p99:7.2f}  {mx:7.2f}   {n:4d}")
        else:
            rows.append(f"  {b:>9s}  {'unreadable':>34s}")
    rows += [
        "", "  How to read this:",
        "    v2 bins (medNN) are named by their MEDIAN, so the median column should match the",
        "    name. v1 bins (pixNN) are named by their intended MAXIMUM and are quantised to whole",
        "    frames, so pix30 really has a median near 13.6 px — that mismatch is BUG-13, and it",
        "    is the reason v2 exists.",
    ]
    fig.text(0.08, 0.46, "\n".join(rows), fontsize=8.5, family="monospace", va="top")
    return fig


def metadata_page(meta_files, max_files):
    """Summarise the per-simulation metadata sidecars: what physics produced this dataset."""
    if load_toml is None:
        return page_text("Simulation metadata",
                         ["No TOML reader available (pip install tomli) — skipped."])
    rows, seeds = [], []
    grabbed = {}
    for p in meta_files[:max_files]:
        try:
            m = load_toml(p)
        except Exception:
            continue
        rep, ic, phys, fin = (m.get("reproducibility", {}), m.get("ic_spec", {}),
                              m.get("ic_physics", {}), m.get("sampling_final", {}))
        seeds.append(rep.get("seed"))
        rows.append((ic.get("A_jet_amplitude"), phys.get("U_max"), phys.get("k_p"),
                     phys.get("dt_save"), fin.get("C_achieved")))
        grabbed = grabbed or rep

    labels = ["jet amplitude A", "U_max (px/t)", "k_p", "dt_save", "C_achieved"]
    # reshape keeps this 2-D even when no sidecar was readable, so the column loop below is safe
    arr = np.array([[np.nan if v is None else v for v in r] for r in rows],
                   dtype=float).reshape(-1, len(labels))
    if not rows:
        return page_text("Simulation metadata", [
            "No readable metadata sidecars found under this dataset.",
            "",
            "Looked in: metadata/, metadata_v2/, data/binary/, binary/ and the root itself.",
            "A run without sidecars can still be trained on, but nothing records which physics",
            "produced it — check that the campaign copied its metadata/ directory across.",
        ])

    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Simulation metadata", fontsize=15, weight="bold", y=0.97)

    lines = ["CONSTANT ACROSS THE CAMPAIGN (from the first sidecar read)", ""]
    for k in ("grid_N", "grid_M", "extent", "viscosity_nu", "advection", "n_max", "m_jet",
              "nt", "jet_amp_arg", "julia_version", "git_sha", "git_dirty", "params_file"):
        if k in grabbed:
            lines.append(f"  {k:<16s} {grabbed[k]}")
    valid = [s for s in seeds if s is not None]
    if valid:
        lines += ["", f"  seeds read       {len(valid)} files, range {min(valid)}–{max(valid)}"]
    lines += [
        "", "VARIES PER SIMULATION (this is where the dataset's diversity comes from)", "",
        "     quantity          min      mean       max",
        "  " + "-" * 44,
    ]
    for i, lab in enumerate(labels):
        col = arr[:, i]
        col = col[~np.isnan(col)]
        if col.size:
            lines.append(f"  {lab:<16s} {col.min():9.4g} {col.mean():9.4g} {col.max():9.4g}")
    lines += [
        "",
        "  Note that the jet amplitude VARIES even though jet_amp_arg is a single number: the",
        "  code computes A = jet_amp_arg * (1.5 - rand()) after seeding, so every simulation gets",
        "  a different jet strength. That is deliberate, and it is a cheap source of flow",
        "  diversity. See params.toml [physics].jet_amplitude.",
    ]
    fig.text(0.06, 0.90, "\n".join(lines), fontsize=8.5, family="monospace", va="top")

    if arr.shape[0] > 1:
        ax = fig.add_axes([0.12, 0.30, 0.76, 0.20])
        col = arr[:, 0][~np.isnan(arr[:, 0])]
        if col.size:
            ax.hist(col, bins=30, color="#4477aa")
            ax.set_xlabel("realised jet amplitude A")
            ax.set_ylabel("simulations")
            ax.set_title("Flow-strength spread across the campaign", fontsize=9)
    return fig


def sample_page(path, bin_name):
    """Six panels for one sample: the two frames, what moved, and the flow field that labels it."""
    A, B, uA, vA = read_pair(path)
    H, W = A.shape
    An = A / max(A.max(), 1.0)
    Bn = B / max(B.max(), 1.0)
    disp = np.sqrt(uA**2 + vA**2)

    overlay = np.zeros((H, W, 3))
    overlay[..., 0] = An      # frame A in red
    overlay[..., 1] = Bn      # frame B in green -> stationary particles look yellow

    fig, axes = plt.subplots(2, 3, figsize=(11, 8.0))
    fig.suptitle(f"{bin_name}   ·   {os.path.basename(path)}", fontsize=10, weight="bold")

    axes[0, 0].imshow(An, cmap="gray"); axes[0, 0].set_title("frame A", fontsize=9)
    axes[0, 1].imshow(Bn, cmap="gray"); axes[0, 1].set_title("frame B", fontsize=9)
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title("overlay: A red, B green\n(yellow = did not move)", fontsize=9)

    # A zoom makes individual particle offsets visible; the full frame never does.
    cs = 110
    y0, x0 = H // 2 - cs // 2, W // 2 - cs // 2
    axes[1, 0].imshow(overlay[y0:y0 + cs, x0:x0 + cs])
    axes[1, 0].set_title(f"overlay, {cs}x{cs} zoom", fontsize=9)

    im = axes[1, 1].imshow(disp, cmap="viridis")
    axes[1, 1].set_title("|displacement| (px)", fontsize=9)
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

    # Quiver in image convention: rows increase downward, so v is plotted as-is against imshow.
    s = max(1, W // 32)
    yy, xx = np.mgrid[0:H:s, 0:W:s]
    axes[1, 2].quiver(xx, yy, uA[::s, ::s], vA[::s, ::s],
                      angles="xy", scale_units="xy", scale=1, width=0.004)
    axes[1, 2].set_xlim(0, W); axes[1, 2].set_ylim(H, 0); axes[1, 2].set_aspect("equal")
    axes[1, 2].set_title("flow field (arrows 1:1 in px)", fontsize=9)

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    fig.text(0.5, 0.015,
             f"displacement  median {np.median(disp):.2f} px   p90 {np.percentile(disp,90):.2f}   "
             f"max {disp.max():.2f}      image  mean {A.mean():.1f}/255  "
             f"min {A.min():.0f}  max {A.max():.0f}",
             ha="center", fontsize=8.5, family="monospace")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig


# ----------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Report on a generated PIV dataset: inventory, statistics and sample images.")
    ap.add_argument("--root", required=True,
                    help="run directory (holding pix*/ or med*/), or one containing data/visual/")
    ap.add_argument("--n", type=int, default=6, help="random samples to show (default 6)")
    ap.add_argument("--seed", type=int, default=0, help="seed for choosing them (reproducible)")
    ap.add_argument("--out", default="", help="output path without extension (default <root>/report)")
    ap.add_argument("--format", choices=("pdf", "md", "both"), default="pdf")
    ap.add_argument("--stat-files", type=int, default=20,
                    help="samples per bin to pool for the statistics page (default 20)")
    ap.add_argument("--meta-files", type=int, default=500,
                    help="metadata sidecars to summarise (default 500)")
    a = ap.parse_args()

    bin_dir, bins = find_bin_dirs(a.root)
    if not bins:
        sys.exit(f"no bin directories (pix*/ or med*/) found under {a.root}")

    per_bin_files = {b: sorted(glob.glob(os.path.join(bin_dir, b, "*.jld2"))) for b in bins}
    counts = {b: len(per_bin_files[b]) for b in bins}
    if not sum(counts.values()):
        sys.exit(f"bin directories exist under {bin_dir} but contain no .jld2 files")

    run_info = ""
    ri = os.path.join(a.root, "RUN_INFO.txt")
    if os.path.isfile(ri):
        run_info = open(ri).read().strip()

    params_text = ""
    for cand in (os.path.join(a.root, "params.toml"),
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "params.toml")):
        if os.path.isfile(cand):
            # Comments stripped, and capped so the cover stays on one page — the file itself is
            # the reference, this is just a record of what the run was configured with.
            body = [l.rstrip() for l in open(cand)
                    if l.strip() and not l.lstrip().startswith("#")]
            if len(body) > 48:
                body = body[:48] + [f"... ({len(body) - 48} more lines; see params.toml)"]
            params_text = "\n".join(body)
            break

    meta_files = find_metadata(a.root)

    # Pick the samples: spread across bins so the report shows the whole displacement range
    # rather than N draws from whichever bin happens to be biggest.
    rng = random.Random(a.seed)
    picks = []
    for i in range(a.n):
        b = bins[i % len(bins)]
        if per_bin_files[b]:
            picks.append((b, rng.choice(per_bin_files[b])))

    out = a.out or os.path.join(a.root, "report")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

    figs = [
        ("cover", cover_page(a.root, bins, counts, run_info, params_text, len(meta_files))),
        ("statistics", stats_page(bin_dir, bins, per_bin_files, a.stat_files)),
        ("metadata", metadata_page(meta_files, a.meta_files)),
    ]
    for i, (b, p) in enumerate(picks, 1):
        try:
            figs.append((f"sample{i}_{b}", sample_page(p, b)))
        except (OSError, KeyError) as err:
            print(f"  ! could not read {p}: {err}", file=sys.stderr)

    if a.format in ("pdf", "both"):
        with PdfPages(out + ".pdf") as pdf:
            for _, fig in figs:
                pdf.savefig(fig)
        print(f"wrote {out}.pdf  ({len(figs)} pages)")

    if a.format in ("md", "both"):
        figdir = out + "_figs"
        os.makedirs(figdir, exist_ok=True)
        md = [f"# Dataset report — `{os.path.basename(os.path.abspath(a.root))}`", "",
              f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}.", "",
              "| bin | samples |", "|---|---|"]
        md += [f"| `{b}` | {counts[b]:,} |" for b in bins]
        md += ["", f"**{sum(counts.values()):,} samples** = "
                   f"{sum(counts.values()) // len(bins):,} simulations x {len(bins)} bins.", ""]
        if run_info:
            md += ["## Run info", "", "```", run_info, "```", ""]
        for name, fig in figs:
            png = os.path.join(figdir, name + ".png")
            fig.savefig(png, dpi=110, bbox_inches="tight")
            md += [f"## {name}", "", f"![{name}]({os.path.relpath(png, os.path.dirname(out))})", ""]
        with open(out + ".md", "w") as f:
            f.write("\n".join(md))
        print(f"wrote {out}.md and {figdir}/")

    for _, fig in figs:
        plt.close(fig)


if __name__ == "__main__":
    main()
