#!/usr/bin/env python3
"""
load_jld2_particles.py

Read particle positions from a .jld2 file (JLD2/HDF5) and optionally load
background fields (e.g., ω and s) from a second JLD2 file.

Typical Oceananigans layout:
  /timeseries/particles/<step>  with fields x,y,z (often HDF5 object references)
  /timeseries/t/<step>          or /timeseries/particles/t (optional time)

Outputs:
  steps : (n_frames,) int
  X,Y,Z : (n_frames, n_particles) float arrays (Z may be None if not present)
  T     : (n_frames,) float array or None

Also provides a CLI to export to NPZ for easy loading in another program.
"""

import argparse
from typing import Optional, Tuple, List, Any

import numpy as np
import h5py


class ParticleSeries(object):
    def __init__(self, steps, x, y, z=None, t=None):
        self.steps = steps                # (n_frames,)
        self.x = x                        # (n_frames, n_particles)
        self.y = y                        # (n_frames, n_particles)
        self.z = z                        # (n_frames, n_particles) or None
        self.t = t                        # (n_frames,) or None


class ParticleFieldSeries(ParticleSeries):
    """Particles plus 2D background fields aligned by frame key."""
    def __init__(
        self,
        steps,
        x,
        y,
        z=None,
        t=None,
        field_a=None,
        field_b=None,
        field_a_name="",
        field_b_name=""
    ):
        super(ParticleFieldSeries, self).__init__(steps, x, y, z=z, t=t)
        self.field_a = field_a            # (n_frames, Ny, Nx)
        self.field_b = field_b            # (n_frames, Ny, Nx)
        self.field_a_name = field_a_name
        self.field_b_name = field_b_name


def _is_numeric_key(k):
    # Oceananigans steps are often stringified integers like "0", "14", "37"
    return k.isdigit()


def _sorted_numeric_keys(group):
    keys = [k for k in group.keys() if _is_numeric_key(k)]
    keys.sort(key=lambda s: int(s))
    return keys


def _find_timeseries_group(h5, candidates):
    """Return /timeseries/<name> group for the first matching candidate name."""
    if "timeseries" not in h5 or not isinstance(h5["timeseries"], h5py.Group):
        return None
    ts = h5["timeseries"]
    for name in candidates:
        if name in ts and isinstance(ts[name], h5py.Group):
            return ts[name]
    return None


def _deref_if_needed(h5, obj):
    """
    JLD2 often stores arrays as HDF5 object references.
    This function returns the numeric ndarray regardless of whether obj is:
      - a dataset
      - an object reference
      - a numpy object array containing a reference
    """
    # Direct dataset
    if isinstance(obj, h5py.Dataset):
        return np.array(obj[()])

    # Direct HDF5 reference
    if isinstance(obj, h5py.Reference):
        return np.array(h5[obj][()])

    # Some JLD2 fields come as numpy arrays of dtype object, containing a ref
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
        inner = obj.flat[0]
        if isinstance(inner, h5py.Reference):
            return np.array(h5[inner][()])
        if isinstance(inner, h5py.Dataset):
            return np.array(inner[()])
        return np.array(inner)

    # Fallback: try converting to numpy
    return np.array(obj)


def _read_field_frame(h5, group, key):
    """Read a 2D field frame from /timeseries/<field>/<key>, dereferencing if needed."""
    ds = group[key]
    data = ds[()]
    arr = _deref_if_needed(h5, data)
    out = np.array(arr)
    # Many Oceananigans 2D fields are stored as (Nx, Ny, 1) or (Ny, Nx)
    out = np.squeeze(out)
    return out


def load_particles_and_fields_jld2(
    particles_path,
    fields_path,
    particles_group="/timeseries/particles",
    field_a_candidates=("u"),
    field_b_candidates=("v"),
):
    """
    Load particle positions from `particles_path` and two background fields
    from `fields_path`.

    Returns ParticleFieldSeries with:
      - steps, x, y, z, t from particles file
      - field_a and field_b as (n_frames, Ny, Nx) arrays aligned to the same numeric keys
    """
    # Load particles first (full arrays)
    ps = load_particles_jld2(particles_path, particles_group=particles_group)

    # Open particles again to get the sorted frame keys
    with h5py.File(particles_path, "r") as hp:
        pg = hp[particles_group]
        frame_keys = _sorted_numeric_keys(pg)

    with h5py.File(fields_path, "r") as hf:
        ga = _find_timeseries_group(hf, field_a_candidates)
        gb = _find_timeseries_group(hf, field_b_candidates)
        if ga is None or gb is None:
            available = list(hf["timeseries"].keys()) if "timeseries" in hf else list(hf.keys())
            raise KeyError(
                "Could not find requested field timeseries groups in fields file. "
                "Tried field_a_candidates={} and field_b_candidates={}. "
                "Available keys: {}".format(field_a_candidates, field_b_candidates, available)
            )

        # Align by intersection of numeric frame keys present in BOTH files.
        common_keys = [k for k in frame_keys if (k in ga and k in gb)]
        if not common_keys:
            raise KeyError(
                "No overlapping frame keys between particles and fields files. "
                "First 10 particle keys: {}. "
                "First 10 fieldA keys: {}. "
                "First 10 fieldB keys: {}.".format(
                    frame_keys[:10], list(ga.keys())[:10], list(gb.keys())[:10]
                )
            )

        # Build index mapping from particle frame_keys -> row indices in ps arrays
        key_to_index = dict((k, i) for i, k in enumerate(frame_keys))
        keep_idx = np.array([key_to_index[k] for k in common_keys], dtype=int)

        # Subset particle arrays to the aligned frames
        steps_aligned = ps.steps[keep_idx]
        x_aligned = ps.x[keep_idx]
        y_aligned = ps.y[keep_idx]
        z_aligned = ps.z[keep_idx] if ps.z is not None else None
        t_aligned = ps.t[keep_idx] if ps.t is not None else None

        # Read aligned fields
        A_list = [_read_field_frame(hf, ga, k) for k in common_keys]
        B_list = [_read_field_frame(hf, gb, k) for k in common_keys]

        field_a = np.stack(A_list, axis=0).astype(np.float32, copy=False)
        field_b = np.stack(B_list, axis=0).astype(np.float32, copy=False)

        # Print a small warning if some particle frames were dropped
        dropped = len(frame_keys) - len(common_keys)
        if dropped > 0:
            print("[Warn] Dropped {} particle frames that were missing in fields file. Using {} aligned frames.".format(
                dropped, len(common_keys)
            ))

        field_a_name = str(getattr(ga, "name", "")).split("/")[-1] if ga is not None else ""
        field_b_name = str(getattr(gb, "name", "")).split("/")[-1] if gb is not None else ""

    return ParticleFieldSeries(
        steps=steps_aligned,
        x=x_aligned,
        y=y_aligned,
        z=z_aligned,
        t=t_aligned,
        field_a=field_a,
        field_b=field_b,
        field_a_name=field_a_name,
        field_b_name=field_b_name,
    )


def _read_particle_frame(h5, frame_ds):
    """
    Read one particle frame dataset that has fields x,y,(z).
    """
    if frame_ds.dtype.names is None:
        raise ValueError(
            "Frame dataset {} is not a compound dtype. dtype={}".format(
                frame_ds.name, frame_ds.dtype
            )
        )

    names = set(frame_ds.dtype.names)
    if "x" not in names or "y" not in names:
        raise ValueError(
            "Frame dataset {} missing x/y fields. Has: {}".format(
                frame_ds.name, frame_ds.dtype.names
            )
        )

    # Read fields individually (works with JLD2 object types)
    x_raw = frame_ds.fields("x")[()]
    y_raw = frame_ds.fields("y")[()]
    z_raw = frame_ds.fields("z")[()] if "z" in names else None

    # Dereference if needed
    x = _deref_if_needed(h5, x_raw).astype(np.float64, copy=False)
    y = _deref_if_needed(h5, y_raw).astype(np.float64, copy=False)
    z = _deref_if_needed(h5, z_raw).astype(np.float64, copy=False) if z_raw is not None else None

    # Flatten to 1D
    x = np.ravel(x)
    y = np.ravel(y)
    if z is not None:
        z = np.ravel(z)

    return x, y, z


def _try_read_times(h5, steps):
    """
    Try common time locations. Returns (n_frames,) or None.
    """
    # 1) /timeseries/t/<step>
    if "timeseries" in h5 and isinstance(h5["timeseries"], h5py.Group):
        ts = h5["timeseries"]
        if "t" in ts and isinstance(ts["t"], h5py.Group):
            tg = ts["t"]
            tvals = []
            for s in steps:
                k = str(int(s))
                if k in tg:
                    tvals.append(np.array(tg[k][()]).reshape(-1))
                else:
                    return None
            out = np.array([float(v[0]) if v.size else np.nan for v in tvals], dtype=float)
            return out

        # 2) /timeseries/particles/t (sometimes)
        if "particles" in ts and isinstance(ts["particles"], h5py.Group):
            pg = ts["particles"]
            if "t" in pg and isinstance(pg["t"], h5py.Dataset):
                arr = np.array(pg["t"][()])
                return np.ravel(arr).astype(float, copy=False)

    return None


def load_particles_jld2(path, particles_group="/timeseries/particles"):
    """
    Load particle positions from a JLD2 file.

    Parameters
    ----------
    path : str
        Path to .jld2 file.
    particles_group : str
        HDF5 group where particle frames live (default: /timeseries/particles)

    Returns
    -------
    ParticleSeries
    """
    with h5py.File(path, "r") as h5:
        if particles_group.lstrip("/") not in h5 and particles_group not in h5:
            try:
                pg = h5[particles_group]
            except Exception as e:
                raise KeyError(
                    "Could not find particles group '{}' in file. Top-level keys: {}".format(
                        particles_group, list(h5.keys())
                    )
                )
        pg = h5[particles_group]

        if not isinstance(pg, h5py.Group):
            raise TypeError("'{}' exists but is not a group. Type: {}".format(
                particles_group, type(pg)
            ))

        frame_keys = _sorted_numeric_keys(pg)
        if not frame_keys:
            raise KeyError(
                "No numeric frame keys found under {}. Available keys: {}".format(
                    particles_group, list(pg.keys())
                )
            )

        steps = np.array([int(k) for k in frame_keys], dtype=int)

        X_list = []
        Y_list = []
        Z_list = []
        z_present = None

        for k in frame_keys:
            ds = pg[k]
            if not isinstance(ds, h5py.Dataset):
                continue

            x, y, z = _read_particle_frame(h5, ds)
            X_list.append(x)
            Y_list.append(y)

            if z is None:
                if z_present is None:
                    z_present = False
            else:
                if z_present is None:
                    z_present = True
                Z_list.append(z)

        n_particles = len(X_list[0])
        for i, arr in enumerate(X_list):
            if len(arr) != n_particles:
                raise ValueError(
                    "Particle count changes at frame index {} (step {}): {} vs {}".format(
                        i, steps[i], len(arr), n_particles
                    )
                )

        X = np.stack(X_list, axis=0)
        Y = np.stack(Y_list, axis=0)

        Z = None
        if z_present:
            if len(Z_list) != len(X_list):
                raise ValueError("Some frames have z while others do not. Mixed dimensionality.")
            Z = np.stack(Z_list, axis=0)

        T = _try_read_times(h5, steps)

    return ParticleSeries(steps=steps, x=X, y=Y, z=Z, t=T)


def main():
    ap = argparse.ArgumentParser(
        description="Load particle positions from a .jld2 file and optionally export arrays for ML."
    )
    ap.add_argument("path", help="Path to particle .jld2 file (e.g., *_particles.jld2)")
    ap.add_argument("--group", default="/timeseries/particles",
                    help="Particles group path (default: /timeseries/particles)")
    ap.add_argument("--export_npz", default=None,
                    help="If set, write an .npz with steps,x,y,(z),(t) from the particle file")

    ap.add_argument("--fields_path", default=None,
                    help="Optional: path to fields .jld2 file")
    ap.add_argument("--field_a", default=None,
                    help="Optional: field A name override (default candidates: u)")
    ap.add_argument("--field_b", default=None,
                    help="Optional: field B name override (default candidates: v)")
    ap.add_argument("--export_imagegen_npz", default=None,
                    help="If set, write a combined NPZ containing particles + fields for the image generator")
    args = ap.parse_args()

    # Base particle series (always load)
    series = load_particles_jld2(args.path, particles_group=args.group)

    # Optional combined particle+field series for image generator export
    pf_series = None
    if args.export_imagegen_npz is not None:
        if args.fields_path is None:
            raise SystemExit("--export_imagegen_npz requires --fields_path to be set.")

        fa = (args.field_a,) if args.field_a else ("u",)
        fb = (args.field_b,) if args.field_b else ("v",)

        pf_series = load_particles_and_fields_jld2(
            particles_path=args.path,
            fields_path=args.fields_path,
            particles_group=args.group,
            field_a_candidates=fa,
            field_b_candidates=fb,
        )

    print("Loaded: {}".format(args.path))
    print("Frames: {}".format(series.steps.size))
    print("Particles per frame: {}".format(series.x.shape[1]))
    print("X shape: {}   Y shape: {}".format(series.x.shape, series.y.shape))
    if series.z is not None:
        print("Z shape: {}".format(series.z.shape))
    else:
        print("Z: not present")
    if series.t is not None:
        print("t shape: {}  t[0:5]={}".format(series.t.shape, series.t[:5]))
    else:
        print("t: not present")

    if pf_series is not None:
        print("Field A shape: {}  Field B shape: {}".format(
            pf_series.field_a.shape, pf_series.field_b.shape
        ))

    if args.export_npz:
        out = {
            "steps": series.steps,
            "x": series.x,
            "y": series.y,
        }
        if series.z is not None:
            out["z"] = series.z
        if series.t is not None:
            out["t"] = series.t
        np.savez_compressed(args.export_npz, **out)
        print("Wrote {}".format(args.export_npz))

    if args.export_imagegen_npz and pf_series is not None:
        out = {
            "steps": pf_series.steps,
            "x": pf_series.x,
            "y": pf_series.y,
            "t": pf_series.t,
            "field_a": pf_series.field_a,
            "field_b": pf_series.field_b,
        }
        if pf_series.z is not None:
            out["z"] = pf_series.z
        np.savez_compressed(args.export_imagegen_npz, **out)
        print("Wrote {}".format(args.export_imagegen_npz))


if __name__ == "__main__":
    main()

# run with:
# python load_jld2_particles.py "2D_Turbulance(particles)_particles.jld2" --fields_path "2D_Turbulance(particles).jld2" --field_a u --field_b v --export_imagegen_npz combined.npz