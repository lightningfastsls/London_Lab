"""Human-INSPECTION patches — tight crop around the target call (A5 audit + γ).

NOT the oracle's label substrate. The oracle labels the fixed 0.22 s v1-faithful
patch (render_v1_faithful_patches.py) — that must stay byte-identical to v1
training. THIS render is purely for the human to *see* the target call's shape:
it reuses the exact same cleaning pipeline (`_extract_patch` + percentile
CleaningConfig) but with a TIGHT, per-call window so neighbouring USVs are pushed
out of frame.

window_s = max(call_duration * (1 + 2*pad_frac), min_window_s), centred on the
detection midpoint (same centring as the oracle patch).

Renders only the call_ids the human actually views: the high-confidence label set
(--labels) ∪ an extra manifest (--extra-manifest, e.g. the γ sample). Output:
data/alpha3_human_patches/<call_id>.png + manifest.csv (call_id, path, wav_stem).
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments"))

# Reuse the oracle render's exact cleaning pipeline (import runs its sys.path +
# archived-symbol reattach). _extract_patch returns (227x227x3 uint8, is_all_zero).
from render_v1_faithful_patches import _extract_patch, CleaningConfig  # noqa: E402


def _make_call_id(wav_stem: str, det_index) -> str:
    return f"{wav_stem}__det{int(det_index)}"


def _isolate_target(rgb: np.ndarray, det_start: float, det_end: float,
                    window_s: float, dim: float, pad_frac: float) -> np.ndarray:
    """Ghost everything outside the target call's time span so a neighbouring
    USV that clips into the crop can't be mistaken for part of the target.

    The patch is `window_s` wide, centred on the call midpoint, time increasing
    left→right (flipud only flips the freq axis), uniformly resized to W cols.
    So target columns map linearly. Columns outside [target ± pad] are multiplied
    by `dim` (viewing-only; this render is NOT an analysis substrate)."""
    H, W = rgb.shape[:2]
    mid = 0.5 * (det_start + det_end)
    ws = mid - window_s / 2.0
    c0 = (det_start - ws) / window_s * W
    c1 = (det_end - ws) / window_s * W
    pad = pad_frac * W
    lo = max(0, int(np.floor(c0 - pad)))
    hi = min(W, int(np.ceil(c1 + pad)))
    out = rgb.astype(np.float32)
    keep = np.zeros(W, dtype=bool)
    keep[lo:hi] = True
    out[:, ~keep, :] *= dim
    return np.clip(out, 0, 255).astype(np.uint8)


def _render_one(task: dict) -> dict:
    row = {"call_id": task["call_id"], "path": task["rel_path"],
           "wav_stem": task["wav_stem"], "ok": False, "error": ""}
    try:
        out_path = Path(task["out_path"])
        if out_path.exists() and not task["overwrite"]:
            row["ok"] = True
            return row
        cfg = CleaningConfig(baseline_mode="percentile")  # MANDATORY (v1 watch-out #1)
        call_dur = max(float(task["det_end_s"]) - float(task["det_start_s"]), 1e-4)
        window_s = max(call_dur * (1.0 + 2.0 * task["pad_frac"]), task["min_window_s"])
        rgb, _is_zero = _extract_patch(
            Path(task["wav_path"]), task["det_start_s"], task["det_end_s"], window_s, cfg
        )
        if task.get("isolate"):
            rgb = _isolate_target(rgb, task["det_start_s"], task["det_end_s"],
                                  window_s, task["isolate_dim"], task["isolate_pad_frac"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(out_path)
        row["ok"] = True
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=REPO_ROOT / "classified_detections_lab_131204_clean.csv")
    p.add_argument("--labels", type=Path, default=REPO_ROOT / "data/labels_vocalmat_v1_on_131204.csv",
                   help="render the high_confidence==True call_ids from here")
    p.add_argument("--extra-manifest", type=Path, default=REPO_ROOT / "data/alpha3_gamma_manifest.csv",
                   help="also render every call_id in this manifest (e.g. the γ sample)")
    p.add_argument("--wav-root", type=Path, default=REPO_ROOT / "USV_lab_131204_chunked_2s_full")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/alpha3_human_patches")
    p.add_argument("--pad-frac", type=float, default=0.2, help="padding each side as frac of call duration")
    p.add_argument("--min-window-s", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--only-call-ids", type=Path, default=None,
                   help="restrict the render to the call_ids in this file (one per line, "
                        "or a CSV with a 'call_id' column). Ignores --labels/--extra-manifest.")
    p.add_argument("--isolate", action="store_true",
                   help="ghost the spectrogram outside the target call's time span so a "
                        "neighbouring USV clipping into the crop can't be mislabeled.")
    p.add_argument("--isolate-dim", type=float, default=0.22,
                   help="brightness multiplier for the ghosted (non-target) region.")
    p.add_argument("--isolate-pad-frac", type=float, default=0.06,
                   help="keep-region pad each side, as a fraction of the crop width.")
    args = p.parse_args(argv)

    det = pd.read_csv(args.csv)
    det["call_id"] = det["wav_stem"].astype(str) + "__det" + det["det_index"].astype(int).astype(str)

    want: set[str] = set()
    if args.only_call_ids is not None:
        # explicit call-id list mode (e.g. the 50 multi-USV γ patches to re-render)
        txt = args.only_call_ids.read_text().splitlines()
        if txt and "call_id" in txt[0]:  # CSV with header
            want = set(pd.read_csv(args.only_call_ids)["call_id"].astype(str))
        else:
            want = {ln.strip() for ln in txt if ln.strip()}
    else:
        if args.labels.exists():
            lab = pd.read_csv(args.labels)
            want |= set(lab.loc[lab["high_confidence"] == True, "call_id"].astype(str))  # noqa: E712
        if args.extra_manifest.exists():
            want |= set(pd.read_csv(args.extra_manifest)["call_id"].astype(str))

    sub = det[det["call_id"].isin(want)].drop_duplicates("call_id")
    out_dir = Path(args.out_dir)

    print("=" * 72)
    print("render_human_view_patches — TIGHT-CROP human inspection view")
    print("=" * 72)
    print(f"  csv            : {args.csv}")
    print(f"  high-conf from : {args.labels}")
    print(f"  extra manifest : {args.extra_manifest}")
    print(f"  out-dir        : {out_dir}")
    print(f"  pad-frac       : {args.pad_frac}   min-window-s: {args.min_window_s}")
    print(f"  window rule    : max(call_dur*(1+2*pad), min_window), centred on midpoint")
    print(f"  call_ids wanted: {len(want)}   resolvable in CSV: {len(sub)}")

    tasks = []
    miss = 0
    for r in sub.itertuples(index=False):
        wav_path = args.wav_root / f"{getattr(r, 'wav_stem')}.wav"
        if not wav_path.exists():
            miss += 1
            continue
        cid = getattr(r, "call_id")
        tasks.append({
            "call_id": cid, "wav_stem": str(getattr(r, "wav_stem")),
            "wav_path": str(wav_path), "out_path": str(out_dir / f"{cid}.png"),
            "rel_path": str((out_dir / f"{cid}.png").as_posix()),
            "det_start_s": float(getattr(r, "det_start_s")),
            "det_end_s": float(getattr(r, "det_end_s")),
            "pad_frac": args.pad_frac, "min_window_s": args.min_window_s,
            "overwrite": args.overwrite,
            "isolate": args.isolate, "isolate_dim": args.isolate_dim,
            "isolate_pad_frac": args.isolate_pad_frac,
        })
    print(f"  missing WAVs   : {miss}   tasks: {len(tasks)}")

    rows, n_ok, n_err = [], 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_render_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            n_ok += int(row["ok"])
            n_err += int(not row["ok"])
            if i % 1000 == 0:
                print(f"    ... {i}/{len(tasks)} (ok={n_ok}, err={n_err})")

    good = sorted((r for r in rows if r["ok"]), key=lambda r: r["call_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["call_id", "path", "wav_stem"])
        for r in good:
            w.writerow([r["call_id"], r["path"], r["wav_stem"]])
    print(f"\n  rendered ok    : {n_ok}   errors: {n_err}")
    print(f"  manifest rows  : {len(good)} → {out_dir / 'manifest.csv'}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
