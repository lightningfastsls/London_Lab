"""Combine per-cohort patches.npz + manifest into a single training dataset.

Reads the per-cohort outputs of mass_apply_contour_mask.py for each of the
4 cohorts (5970, 3452, 9252, lab_131204) and writes:

    results/masked_patches/combined_all_cohorts/
        patches.npz                 (N_total, 257, 234) float32
        patches_manifest.parquet    same schema as per-cohort + new ``cohort``
                                    column with one of {"5970","3452","9252",
                                    "lab_131204"}

Stored shape is `np.savez_compressed(..., patches=arr)` — load with
``np.load("patches.npz")["patches"]``.

Rationale: the Phase 5 cross-cohort cage-confound diagnostic needs a single
latents.parquet with a `cohort` column. The simplest pipeline-side change is
to assemble one combined patches.npz so the existing train_contour_vae_v2.py
runs unchanged.

Usage:
    /home/shachar/projects/mickey_london_lab/.venv/bin/python \\
        scripts/assemble_combined_patches.py
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cohort-dirs", nargs="+", type=Path,
        default=[
            _REPO_ROOT / "results/masked_patches/5970_focus",
            _REPO_ROOT / "results/masked_patches/3452_focus",
            _REPO_ROOT / "results/masked_patches/9252_focus",
            _REPO_ROOT / "results/masked_patches/lab_131204_focus",
        ],
        help="One directory per cohort. Each must contain patches.npz "
             "and patches_manifest.parquet.",
    )
    p.add_argument(
        "--cohort-names", nargs="+", type=str,
        default=["5970", "3452", "9252", "lab_131204"],
        help="Labels matched 1:1 to --cohort-dirs.",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_REPO_ROOT / "results/masked_patches/combined_all_cohorts",
        help="Where to write the combined patches.npz + manifest.parquet.",
    )
    p.add_argument(
        "--patches-key", type=str, default="patches",
        help="Key under which the per-cohort .npz stores its array.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.cohort_dirs) != len(args.cohort_names):
        print(f"FAIL: cohort-dirs ({len(args.cohort_dirs)}) and cohort-names "
              f"({len(args.cohort_names)}) lengths must match.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Memmap-streaming concat: peak RAM is ~one chunk (CHUNK_ROWS * patch_size)
    # rather than 2x the full combined dataset. Without this we OOM on hosts
    # with < ~32 GB free; with it we run fine on ~3 GB.
    print("=== Assembling combined patches (memmap-streaming) ===")
    per_cohort_manifests: list[pd.DataFrame] = []
    per_cohort_paths: list[Path] = []
    per_cohort_row_counts: list[int] = []
    patch_shape: tuple[int, ...] | None = None
    patch_dtype: np.dtype | None = None
    freqs_kHz_ref: np.ndarray | None = None
    for d, name in zip(args.cohort_dirs, args.cohort_names):
        npz_path = d / "patches.npz"
        mf_path = d / "patches_manifest.parquet"
        if not npz_path.exists():
            print(f"  FAIL: missing {npz_path}", file=sys.stderr)
            return 1
        if not mf_path.exists():
            print(f"  FAIL: missing {mf_path}", file=sys.stderr)
            return 1
        with np.load(npz_path, mmap_mode="r") as data:
            keys = list(data.keys())
            if args.patches_key not in keys:
                print(f"  FAIL: '{args.patches_key}' not in {npz_path.name} "
                      f"(keys: {keys})", file=sys.stderr)
                return 1
            patches_view = data[args.patches_key]
            n_rows = patches_view.shape[0]
            cohort_shape = patches_view.shape[1:]
            cohort_dtype = patches_view.dtype
            # Preserve freqs_kHz so train_contour_vae_v2.py can find it in
            # the combined npz. Validate consistency across cohorts.
            if "freqs_kHz" in keys:
                f = np.array(data["freqs_kHz"])
                if freqs_kHz_ref is None:
                    freqs_kHz_ref = f
                elif not np.allclose(freqs_kHz_ref, f):
                    print(f"  FAIL: {name}: freqs_kHz differs from first cohort",
                          file=sys.stderr)
                    return 1
        manifest = pd.read_parquet(mf_path)
        if len(manifest) != n_rows:
            print(f"  FAIL: {name}: manifest len {len(manifest)} != patches "
                  f"axis0 {n_rows}", file=sys.stderr)
            return 1
        if patch_shape is None:
            patch_shape = cohort_shape
            patch_dtype = cohort_dtype
        elif cohort_shape != patch_shape:
            print(f"  FAIL: {name}: shape {cohort_shape} != {patch_shape}",
                  file=sys.stderr)
            return 1
        elif cohort_dtype != patch_dtype:
            print(f"  FAIL: {name}: dtype {cohort_dtype} != {patch_dtype}",
                  file=sys.stderr)
            return 1
        manifest = manifest.copy()
        manifest["cohort"] = name
        size_gb = n_rows * int(np.prod(cohort_shape)) * cohort_dtype.itemsize / 1e9
        print(f"  {name}: {n_rows} patches, shape {(n_rows, *cohort_shape)}, {size_gb:.2f} GB")
        per_cohort_manifests.append(manifest)
        per_cohort_paths.append(npz_path)
        per_cohort_row_counts.append(n_rows)

    assert patch_shape is not None and patch_dtype is not None
    total_rows = sum(per_cohort_row_counts)
    total_gb = total_rows * int(np.prod(patch_shape)) * patch_dtype.itemsize / 1e9
    print()
    print(f"--- Combined: {total_rows} patches × {patch_shape}, {total_gb:.2f} GB ---")

    npz_out = args.output_dir / "patches.npz"
    mf_out = args.output_dir / "patches_manifest.parquet"
    npy_out = npz_out.with_suffix(".npy")

    print()
    print(f"--- Streaming patches into {npy_out} (memmap-backed) ---")
    out_mm = np.lib.format.open_memmap(
        str(npy_out), mode="w+",
        shape=(total_rows, *patch_shape),
        dtype=patch_dtype,
    )
    CHUNK_ROWS = 5000  # ~1.2 GB per slice copy at (5000, 257, 234) float32
    offset = 0
    for npz_path, name, n_rows in zip(per_cohort_paths, args.cohort_names,
                                       per_cohort_row_counts):
        with np.load(npz_path, mmap_mode="r") as data:
            patches_view = data[args.patches_key]
            for s in range(0, n_rows, CHUNK_ROWS):
                e = min(s + CHUNK_ROWS, n_rows)
                out_mm[offset + s:offset + e] = np.ascontiguousarray(
                    patches_view[s:e]
                )
        offset += n_rows
        print(f"  {name}: copied {n_rows} rows  (running offset {offset}/{total_rows})",
              flush=True)
    out_mm.flush()
    del out_mm

    print()
    print(f"--- Wrapping {npy_out.name} as {npz_out.name} (ZIP_STORED) ---", flush=True)
    # Stash freqs_kHz to a tmp .npy and add it alongside patches.npy.
    # train_contour_vae_v2.py expects both keys in the combined .npz.
    freqs_tmp = args.output_dir / "freqs_kHz.npy"
    if freqs_kHz_ref is not None:
        np.save(str(freqs_tmp), freqs_kHz_ref)
    with zipfile.ZipFile(str(npz_out), "w", zipfile.ZIP_STORED) as zf:
        zf.write(str(npy_out), arcname="patches.npy")
        if freqs_kHz_ref is not None:
            zf.write(str(freqs_tmp), arcname="freqs_kHz.npy")
    npy_out.unlink()
    if freqs_tmp.exists():
        freqs_tmp.unlink()

    with np.load(npz_out, mmap_mode="r") as verify:
        v = verify[args.patches_key]
        if v.shape != (total_rows, *patch_shape):
            print(f"  FAIL: verify shape {v.shape} != ({total_rows}, *{patch_shape})",
                  file=sys.stderr)
            return 1
        if v.dtype != patch_dtype:
            print(f"  FAIL: verify dtype {v.dtype} != {patch_dtype}",
                  file=sys.stderr)
            return 1

    combined_manifest = pd.concat(per_cohort_manifests, ignore_index=True)
    if combined_manifest.shape[0] != total_rows:
        print(f"  FAIL: manifest rows {combined_manifest.shape[0]} != "
              f"total_rows {total_rows}", file=sys.stderr)
        return 1
    # Each per-cohort manifest restarts patch_idx at 0, so after concat patch_idx is
    # not globally unique (e.g. patch_idx=0 appears 4 times across 4 cohorts).
    # Preserve the cohort-relative id for traceability, then rebuild a global id
    # that matches the positional row order of patches.npz.
    combined_manifest["patch_idx_per_cohort"] = combined_manifest["patch_idx"]
    combined_manifest["patch_idx"] = np.arange(len(combined_manifest), dtype=np.int64)
    combined_manifest.to_parquet(mf_out, index=False)

    print()
    print(f"--- Done ---")
    print(f"  patches.npz: {npz_out.stat().st_size / 1e9:.2f} GB")
    print(f"  manifest.parquet: {mf_out.stat().st_size / 1e6:.1f} MB "
          f"({len(combined_manifest)} rows)")
    print(f"  per-cohort counts:")
    for cname, count in combined_manifest["cohort"].value_counts().items():
        print(f"     {cname}: {count}")
    print(f"\nwall_clock_s: {time.time() - t0:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
