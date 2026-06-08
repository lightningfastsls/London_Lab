"""α₃-C Phase A5 — spot-check audit HTML (the A5 GATE).

Renders a self-contained HTML contact sheet so the USER can eyeball whether the
lab_classifier_v1 oracle's labels (A4) on our A3-rendered 131204 patches "look
right." This is the **A5 GATE**: ≥~80% of patches per class must look correct,
else A3 is retried at 300 kHz or we fall back to γ-only.

Pipeline context::

    A3 (rig)  → data/alpha3_patches/manifest.csv   (call_id, path, wav_stem, cohort)
    A4        → data/labels_vocalmat_v1_on_131204.csv  (label_patches_v1.py output)
    A5 (here) → $CLAUDE_JOB_DIR/alpha3_label_audit.html

A4 label schema (confirmed from label_patches_v1.py lines ~167-178):
    call_id, top1_class, top1_idx, top1_prob, high_confidence (bool),
    softmax_12 (JSON string of 12 floats)
The labels CSV is keyed on the SHARED ``call_id`` join key with the manifest
(the manifest also carries the PNG ``path``). ``call_id`` only appears in the
A4 CSV when the manifest exposed an id column — A4's main() inserts it
conditionally — so we join on call_id when present and otherwise fall back to a
positional (row-order) join, which is sound because A4 preserves manifest order.

Both A3 and A4 outputs may NOT EXIST YET (A3 runs on the rig). Absence is handled
gracefully: print the expected path and exit 0.

NOT-to-touch: this script only READS the manifest, the labels CSV, and the PNGs
it points to. It writes only the audit HTML. It never touches corpus.py, Stack 4,
the model dir, or the production detection pipeline.

Usage (after A4 completes)::

    PYTHONPATH=src .venv/bin/python scripts/experiments/render_label_audit_html.py \\
        --manifest data/alpha3_patches/manifest.csv \\
        --labels   data/labels_vocalmat_v1_on_131204.csv
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import random
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.classifier.dataset import GRIMSLEY_12_CLASSES  # noqa: E402

DEFAULT_JOB_DIR = "/home/shachar/.claude/jobs/6ddc7ce2"
# Threshold used by A4 (label_patches_v1.py HIGH_CONF_THRESHOLD); for display only.
A4_HIGH_CONF_THRESHOLD = 0.85
WSL_URL_BASE = "file://wsl.localhost/Ubuntu"


def resolve_path(p: str) -> Path:
    """Resolve a manifest PNG path (repo-relative or absolute)."""
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp)


def img_to_data_uri(path: Path) -> str | None:
    """Read a PNG and return a base64 data: URI, or None if unreadable."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_html(
    *,
    sections: list[dict],
    total_patches: int,
    high_conf_only: bool,
    per_class: int,
    seed: int,
    manifest_path: str,
    labels_path: str,
) -> str:
    """Assemble the self-contained audit HTML string."""
    filt_label = "high-confidence only" if high_conf_only else "ALL patches (high-conf filter OFF)"

    # Per-class counts table.
    rows = []
    for sec in sections:
        cls = html.escape(sec["class"])
        n_eligible = sec["n_eligible"]
        n_shown = len(sec["items"])
        flag = ""
        if n_eligible == 0:
            flag = ' style="background:#3a1414;color:#ff8a8a;"'
        rows.append(
            f"<tr{flag}><td>{cls}</td><td style='text-align:right'>{n_eligible}</td>"
            f"<td style='text-align:right'>{n_shown}</td></tr>"
        )
    counts_table = "\n".join(rows)

    # Per-class sections.
    section_html_parts = []
    for sec in sections:
        cls = html.escape(sec["class"])
        n_eligible = sec["n_eligible"]
        items = sec["items"]
        if n_eligible == 0:
            body = (
                '<p class="empty">ZERO eligible patches for this class. '
                "(Expected for some classes — e.g. the handoff flags Chevron got "
                "2/844 on held_out_844.)</p>"
            )
        else:
            cap = (
                f"showing {len(items)} of {n_eligible}"
                if len(items) < n_eligible
                else f"showing all {n_eligible}"
            )
            thumbs = []
            for it in items:
                uri = it["data_uri"]
                cid = html.escape(str(it["call_id"]))
                prob = it["top1_prob"]
                prob_str = f"{prob:.3f}" if isinstance(prob, float) else html.escape(str(prob))
                if uri is None:
                    img = '<div class="missing">PNG missing</div>'
                else:
                    img = f'<img src="{uri}" alt="{cid}">'
                thumbs.append(
                    f'<figure>{img}<figcaption>{cid}<br><span class="prob">p={prob_str}</span></figcaption></figure>'
                )
            body = f'<p class="cap">{cap}</p><div class="grid">{"".join(thumbs)}</div>'
        section_html_parts.append(
            f'<section><h2>{cls} <span class="n">({n_eligible} eligible)</span></h2>{body}</section>'
        )
    sections_html = "\n".join(section_html_parts)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>α₃-C A5 — v1 oracle label audit</title>
<style>
  body {{ font-family: system-ui, sans-serif; background:#111; color:#eee; margin:0; padding:24px; }}
  h1 {{ font-size:1.4rem; }}
  h2 {{ border-bottom:1px solid #444; padding-bottom:4px; margin-top:36px; }}
  .n {{ color:#888; font-weight:normal; font-size:0.9rem; }}
  .gate {{ background:#142a14; border:1px solid #2e7d32; padding:12px 16px; border-radius:6px; margin:16px 0; }}
  .meta {{ color:#aaa; font-size:0.85rem; }}
  table {{ border-collapse:collapse; margin:16px 0; }}
  th, td {{ border:1px solid #444; padding:4px 12px; }}
  th {{ background:#222; text-align:left; }}
  .grid {{ display:flex; flex-wrap:wrap; gap:12px; }}
  figure {{ margin:0; background:#1c1c1c; border:1px solid #333; border-radius:4px; padding:6px; width:180px; }}
  figure img {{ width:100%; height:auto; display:block; background:#000; }}
  figcaption {{ font-size:0.72rem; text-align:center; margin-top:4px; word-break:break-all; }}
  .prob {{ color:#7ec8ff; }}
  .cap {{ color:#aaa; font-size:0.85rem; }}
  .empty {{ color:#ff8a8a; font-style:italic; }}
  .missing {{ width:100%; height:80px; display:flex; align-items:center; justify-content:center;
             color:#ff8a8a; background:#222; font-size:0.7rem; }}
</style></head><body>
<h1>α₃-C Phase A5 — v1 oracle label audit</h1>
<div class="gate"><strong>A5 GATE:</strong> ≥~80% of patches per class should look correct.
If a class fails, A3 is retried at 300 kHz or we fall back to γ-only.</div>
<p class="meta">
  total patches in manifest: <strong>{total_patches}</strong><br>
  filter: <strong>{html.escape(filt_label)}</strong>
  (A4 high-conf threshold = {A4_HIGH_CONF_THRESHOLD})<br>
  per-class cap: {per_class} &nbsp;|&nbsp; seed: {seed}<br>
  manifest: {html.escape(manifest_path)}<br>
  labels: {html.escape(labels_path)}
</p>
<h2 style="margin-top:24px">Per-class eligible counts</h2>
<table>
<tr><th>Class (GRIMSLEY_12 order)</th><th>eligible</th><th>shown</th></tr>
{counts_table}
</table>
{sections_html}
</body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description="α₃-C A5 label-audit HTML contact sheet")
    ap.add_argument("--manifest", default="data/alpha3_patches/manifest.csv")
    ap.add_argument("--labels", default="data/labels_vocalmat_v1_on_131204.csv")
    ap.add_argument("--output", default=None,
                    help="output HTML (default: $CLAUDE_JOB_DIR/alpha3_label_audit.html)")
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    hc = ap.add_mutually_exclusive_group()
    hc.add_argument("--high-conf-only", dest="high_conf_only", action="store_true", default=True,
                    help="only show high_confidence==True patches (default)")
    hc.add_argument("--all-patches", dest="high_conf_only", action="store_false",
                    help="show all patches regardless of high_confidence")
    args = ap.parse_args()

    import os
    job_dir = os.environ.get("CLAUDE_JOB_DIR", DEFAULT_JOB_DIR)
    output = Path(args.output) if args.output else Path(job_dir) / "alpha3_label_audit.html"

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    labels_path = Path(args.labels)
    if not labels_path.is_absolute():
        labels_path = REPO_ROOT / labels_path

    print("=" * 72)
    print("α₃-C Phase A5 — label-audit HTML")
    print("=" * 72)
    print(f"  manifest          : {manifest_path}")
    print(f"  labels            : {labels_path}")
    print(f"  output            : {output}")
    print(f"  per-class cap     : {args.per_class}")
    print(f"  seed              : {args.seed}")
    print(f"  filter            : {'high-conf only' if args.high_conf_only else 'ALL patches'}")
    print(f"  A4 high-conf thr  : {A4_HIGH_CONF_THRESHOLD}")
    print(f"  classes (12)      : {list(GRIMSLEY_12_CLASSES)}")

    # --- Graceful absence handling (A3/A4 may not exist yet — they run on the rig) ---
    if not manifest_path.exists():
        print()
        print(f"  [WAIT] A3 manifest not found at: {manifest_path}")
        print("         A3 (patch render) runs on the rig and has not produced output yet.")
        print("         Re-run this script once A3 + A4 complete. Exiting cleanly (0).")
        return 0
    if not labels_path.exists():
        print()
        print(f"  [WAIT] A4 labels not found at: {labels_path}")
        print("         A4 (label_patches_v1.py) has not produced output yet.")
        print("         Re-run this script once A4 completes. Exiting cleanly (0).")
        return 0

    manifest = pd.read_csv(manifest_path)
    labels = pd.read_csv(labels_path)
    print(f"\n  manifest rows     : {len(manifest)}")
    print(f"  labels rows       : {len(labels)}")
    print(f"  manifest columns  : {list(manifest.columns)}")
    print(f"  labels columns    : {list(labels.columns)}")

    # --- Join manifest (has PNG path) with labels (has top1_class / prob / high_conf) ---
    # Prefer the shared call_id key; fall back to positional join (A4 preserves
    # manifest row order) when call_id is absent from either side.
    if "path" not in manifest.columns:
        raise SystemExit(f"manifest has no 'path' column; columns={list(manifest.columns)}")
    for col in ("top1_class", "top1_prob", "high_confidence"):
        if col not in labels.columns:
            raise SystemExit(f"labels CSV missing '{col}'; columns={list(labels.columns)}")

    if "call_id" in manifest.columns and "call_id" in labels.columns:
        join_mode = "call_id"
        merged = manifest.merge(
            labels[[c for c in labels.columns if c != "path"]],
            on="call_id", how="inner",
        )
    else:
        join_mode = "positional"
        if len(manifest) != len(labels):
            raise SystemExit(
                f"positional join requires equal lengths but manifest={len(manifest)} "
                f"!= labels={len(labels)}; cannot align without a call_id key"
            )
        merged = manifest.reset_index(drop=True).copy()
        for col in ("top1_class", "top1_prob", "high_confidence"):
            merged[col] = labels[col].reset_index(drop=True).values
        if "call_id" not in merged.columns:
            merged["call_id"] = [f"row_{i}" for i in range(len(merged))]
    print(f"  join mode         : {join_mode}  ->  {len(merged)} merged rows")

    # Normalize high_confidence to bool (CSV may store True/False as strings).
    def to_bool(v) -> bool:
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("true", "1", "1.0", "yes")

    merged["_high_conf"] = merged["high_confidence"].map(to_bool)
    merged["top1_prob"] = pd.to_numeric(merged["top1_prob"], errors="coerce")

    total_patches = len(merged)
    rng = random.Random(args.seed)

    # --- Build one section per class, in GRIMSLEY_12_CLASSES order ---
    sections: list[dict] = []
    print("\n  per-class eligible counts:")
    for cls in GRIMSLEY_12_CLASSES:
        sub = merged[merged["top1_class"] == cls]
        if args.high_conf_only:
            sub = sub[sub["_high_conf"]]
        idxs = list(sub.index)
        n_eligible = len(idxs)
        print(f"    {cls:<18} {n_eligible:>6}")

        chosen = list(idxs)
        rng.shuffle(chosen)
        chosen = chosen[: args.per_class]

        items = []
        for i in chosen:
            row = merged.loc[i]
            uri = img_to_data_uri(resolve_path(str(row["path"])))
            items.append({
                "call_id": row.get("call_id", i),
                "top1_prob": float(row["top1_prob"]) if pd.notna(row["top1_prob"]) else "n/a",
                "data_uri": uri,
            })
        sections.append({"class": cls, "n_eligible": n_eligible, "items": items})

    page = build_html(
        sections=sections,
        total_patches=total_patches,
        high_conf_only=args.high_conf_only,
        per_class=args.per_class,
        seed=args.seed,
        manifest_path=str(manifest_path),
        labels_path=str(labels_path),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")

    print(f"\n  wrote audit HTML  : {output}")
    # feedback_wsl_file_viewing (MANDATORY): surface the WSL URL for every HTML.
    wsl_url = f"{WSL_URL_BASE}{output.resolve()}"
    print(f"  OPEN IN WINDOWS   : {wsl_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
