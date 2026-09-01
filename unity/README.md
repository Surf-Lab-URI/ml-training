# Slurm job scripts

**How to run a campaign is in the top-level [`README.md`](../README.md); every setting is in
[`params.toml`](../params.toml).** This file only covers what is in this directory and how to watch
a running job.

## What is here

You run the `submit_*.sh` drivers. They read `params.toml`, create a timestamped run folder,
snapshot the code, write `RUN_INFO.txt`, and launch the matching `.sbatch` array — you never call
an `.sbatch` directly.

| script | what it does |
|---|---|
| `submit_run.sh` | one simulation campaign, single array → `run_array.sbatch` |
| `submit_chunked.sh` | the same, in chunks, for campaigns above the ~2000-job queue limit |
| `submit_rerender.sh` | re-render an existing campaign with the v1 bins → `rerender_lab.sbatch` |
| `submit_v2.sh` | re-render with the v2 median bins → `rerender_v2.sbatch` |
| `finalize_run.sbatch` | runs after a campaign: builds `manifest.parquet` and a preview |
| `archive_run.sbatch` | packs a finished run for long-term storage |

**Superseded, kept only for reference:** `generate_dataset.sbatch` and `run100.sbatch`. No submitter
calls them and they predate `params.toml`, so their hardcoded array sizes, seeds and paths are not
what a run will use. Do not start from them.

## Monitoring

```bash
squeue --me                                            # what is queued and running
tail -f <RUN_DIR>/logs/*_1.out                         # the first array task's output
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # per-task accounting after the fact
scancel <jobid>                                        # stop a campaign
```

Each task writes to node-local scratch (`PIV_OUT_DIR`) and copies the durable artifacts into the
run folder at the end, so parallel tasks never collide.

## Sizing

Measured on CPU at `nt = 40`:

- **~3–4 minutes per simulation** (Julia startup plus roughly 3 minutes of physics).
- **~14 MB per simulation** of image pairs and metadata — the part you train on.
- **~900 MB per simulation** for the raw `combined/` file, if `[run].keep_combined = true`.
  That is the default and it is what makes re-rendering possible; at 10 000 simulations it is
  about 9 TB, so check your quota before a large campaign. Setting it false is irreversible for
  that campaign.
- GPU is faster per simulation but usually loses to CPU concurrency for many short jobs.

## Notes

- The whole campaign is reproducible from one number: **`seed = base_seed + array index`**.
- Do not put your Julia depot on `$HOME`. It is NFS-mounted and intermittently unresolvable at task
  start, which killed 34 of 153 tasks in the 2026-08-26 re-render. Set `[unity].julia_depot` to a
  path on `/work`; the job scripts also retry five times before giving up.
- **v1 (`pixN`) bin names are nominal, not exact** — displacements quantise to multiples of
  `smax ≈ 5 px`, so `pix10/20/30` deliver roughly 5/15/25 px and vary between simulations. This is
  BUG-13, fixed in the v2 generator. The velocity-field labels inside each pair file are exact
  either way: use them, never the folder name, as ground truth. See `../bugs.md`.
