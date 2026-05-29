"""Run-after-prep reconciliation per the handoff's decision matrix.

Triggered by the completion watcher when scripts/cnn_prepare_training_data.py
exits. Inspects outputs in data/lab_cnn_training/, applies the handoff's
expectations, and writes a self-contained HTML status report to
$CLAUDE_JOB_DIR/prep_complete/index.html.

The HTML embeds:
- Per-cohort patch counts (lab, wild, vocalmat-only manifest)
- Per-class counts in train/val/test splits
- domain_unlabeled.csv row count
- Handoff decision matrix outcomes (PASS/WARN/FAIL per row)
- Inline thumbnails of a few sanity patches per cohort (for the human gate)
- Direct WSL URL to the sanity_patches/ directory

Idempotent: safe to re-run after a partial prep. Reports whatever state
exists.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter
from pathlib import Path

OUTPUT_DIR = Path("data/lab_cnn_training")
JOB_DIR = Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp"))
REPORT_DIR = JOB_DIR / "prep_complete"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _class_counts(rows: list[dict]) -> Counter:
    return Counter(r.get("class", "?") for r in rows)


def _copy_sanity_thumbnails(rel_dir: Path, n_per_cohort: int = 6) -> dict[str, list[str]]:
    """Copy up to N sanity patches per cohort to the report dir; return relative paths."""
    import shutil
    thumbs: dict[str, list[str]] = {}
    sanity_root = OUTPUT_DIR / "sanity_patches"
    if not sanity_root.is_dir():
        return thumbs
    for cohort_dir in sorted(sanity_root.iterdir()):
        if not cohort_dir.is_dir():
            continue
        files = sorted(cohort_dir.glob("*.png"))[:n_per_cohort]
        copied: list[str] = []
        for src in files:
            dst = REPORT_DIR / f"sanity_{cohort_dir.name}_{src.name}"
            try:
                shutil.copy2(src, dst)
                copied.append(dst.name)
            except Exception as exc:
                print(f"  warn: copy {src} -> {dst} failed: {exc}", file=sys.stderr)
        thumbs[cohort_dir.name] = copied
    return thumbs


def _decision_row(label: str, status: str, detail: str) -> str:
    color = {"PASS": "#3a3", "WARN": "#aa3", "FAIL": "#a33"}.get(status, "#888")
    return f"<tr><td style='color:{color};font-weight:bold'>{status}</td><td>{label}</td><td><code>{detail}</code></td></tr>"


def main() -> int:
    print(f"reconcile: OUTPUT_DIR={OUTPUT_DIR}")

    # --- Inventory ---
    manifest_all = _safe_read_manifest(OUTPUT_DIR / "manifest_all.csv")
    train = _safe_read_manifest(OUTPUT_DIR / "train" / "manifest.csv")
    val = _safe_read_manifest(OUTPUT_DIR / "val" / "manifest.csv")
    test = _safe_read_manifest(OUTPUT_DIR / "test" / "manifest.csv")
    domain_unlabeled = _safe_read_manifest(OUTPUT_DIR / "domain_unlabeled.csv")

    lab_patches = len(list((OUTPUT_DIR / "patches" / "lab").glob("*.png"))) if (OUTPUT_DIR / "patches" / "lab").is_dir() else 0
    wild_patches = len(list((OUTPUT_DIR / "patches" / "wild").glob("*.png"))) if (OUTPUT_DIR / "patches" / "wild").is_dir() else 0

    print(f"  manifest_all: {len(manifest_all)} rows")
    print(f"  splits: train={len(train)} val={len(val)} test={len(test)}")
    print(f"  domain_unlabeled: {len(domain_unlabeled)} rows")
    print(f"  patches: lab={lab_patches} wild={wild_patches}")

    # --- Decision matrix from handoff ---
    train_classes = _class_counts(train)
    val_classes = _class_counts(val)
    test_classes = _class_counts(test)
    all_classes = _class_counts(manifest_all)

    decisions = []
    expected_classes = {
        "Noise", "Step up", "Down-FM", "Short", "Chevron", "Up-FM",
        "Flat", "Two steps", "Step down", "Complex", "Reverse Chevron",
        "Multi-steps",
    }

    # Row 1: 12 classes in all three splits
    missing_per_split = {}
    for name, c in (("train", train_classes), ("val", val_classes), ("test", test_classes)):
        missing = expected_classes - set(c.keys())
        if missing:
            missing_per_split[name] = sorted(missing)
    if not missing_per_split:
        decisions.append(_decision_row("All 12 classes present in train/val/test", "PASS", "ok"))
    elif list(missing_per_split.keys()) == ["test"] and missing_per_split["test"] == ["Multi-steps"]:
        decisions.append(_decision_row("All 12 classes present in train/val/test", "WARN",
            "Multi-steps absent from test (8 recordings * 10% rounds down) -- acceptable"))
    else:
        decisions.append(_decision_row("All 12 classes present in train/val/test", "FAIL",
            f"missing: {missing_per_split}"))

    # Row 2: total VocalMat rows >= 12,000
    total_vm = len(manifest_all)
    if total_vm >= 12_000:
        decisions.append(_decision_row("Total VocalMat rows >= 12,000", "PASS", f"{total_vm}"))
    else:
        decisions.append(_decision_row("Total VocalMat rows >= 12,000", "WARN",
            f"{total_vm} -- partial download or filter loss"))

    # Row 3: domain_unlabeled non-empty
    if len(domain_unlabeled) > 0:
        decisions.append(_decision_row("domain_unlabeled.csv non-empty", "PASS", f"{len(domain_unlabeled)} patches"))
    elif lab_patches > 0 or wild_patches > 0:
        decisions.append(_decision_row("domain_unlabeled.csv non-empty", "FAIL",
            f"lab={lab_patches} wild={wild_patches} but domain_unlabeled empty"))
    else:
        decisions.append(_decision_row("domain_unlabeled.csv non-empty", "WARN", "no lab/wild patches yet"))

    # Row 4: lab + wild patches written
    if lab_patches > 1000 and wild_patches > 100:
        decisions.append(_decision_row("Lab + wild patches written to disk", "PASS",
            f"lab={lab_patches} wild={wild_patches}"))
    else:
        decisions.append(_decision_row("Lab + wild patches written to disk", "WARN",
            f"lab={lab_patches} wild={wild_patches} -- prep may still be running"))

    # --- Sanity thumbnails ---
    print("  copying sanity-patch thumbnails ...")
    thumbs = _copy_sanity_thumbnails(REPORT_DIR, n_per_cohort=8)

    # --- HTML report ---
    print("  writing HTML report ...")
    html = _render_html(
        manifest_all=manifest_all, train=train, val=val, test=test,
        domain_unlabeled=domain_unlabeled,
        train_classes=train_classes, val_classes=val_classes, test_classes=test_classes,
        all_classes=all_classes,
        lab_patches=lab_patches, wild_patches=wild_patches,
        decisions=decisions, thumbs=thumbs,
    )
    html_path = REPORT_DIR / "index.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"\nDONE. Report: {html_path}")
    print(f"WSL URL: file://wsl.localhost/Ubuntu{html_path}")
    print(f"Sanity patches dir: file://wsl.localhost/Ubuntu{(OUTPUT_DIR / 'sanity_patches').resolve()}")
    return 0


def _render_html(*, manifest_all, train, val, test, domain_unlabeled,
                 train_classes, val_classes, test_classes, all_classes,
                 lab_patches, wild_patches, decisions, thumbs) -> str:
    def classes_table(name: str, c: Counter) -> str:
        rows = "".join(
            f"<tr><td>{cls}</td><td style='text-align:right'>{c.get(cls, 0)}</td></tr>"
            for cls in sorted(set(c.keys()) | {
                "Noise", "Step up", "Down-FM", "Short", "Chevron", "Up-FM",
                "Flat", "Two steps", "Step down", "Complex", "Reverse Chevron",
                "Multi-steps",
            })
        )
        return f"<h3>{name} ({sum(c.values())} total)</h3><table>{rows}</table>"

    decisions_html = "\n".join(decisions)

    thumbs_html = []
    for cohort, files in thumbs.items():
        if not files:
            continue
        thumbs_html.append(f"<h3>{cohort}</h3><div class='thumbs'>")
        for f in files:
            thumbs_html.append(f"<figure><img src='{f}' alt='{f}'><figcaption>{f.replace('sanity_' + cohort + '_', '')}</figcaption></figure>")
        thumbs_html.append("</div>")
    thumbs_html = "\n".join(thumbs_html)

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Module 18.2b prep reconciliation</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 1rem; background: #111; color: #eee; }}
h1 {{ font-size: 1.4rem; }}
h2 {{ font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #444; padding-bottom: .2rem; }}
table {{ border-collapse: collapse; margin: .5rem 0; }}
td, th {{ padding: .25rem .8rem; border-bottom: 1px solid #333; }}
.cols {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }}
.thumbs {{ display: flex; flex-wrap: wrap; gap: .5rem; }}
.thumbs figure {{ margin: 0; }}
.thumbs img {{ width: 180px; height: 180px; border: 1px solid #555; image-rendering: pixelated; }}
.thumbs figcaption {{ color: #888; font-size: .75rem; text-align: center; max-width: 180px; word-break: break-all; }}
code {{ background: #222; padding: 1px 4px; }}
</style></head><body>
<h1>Module 18.2b prep reconciliation</h1>
<p>Auto-generated when <code>scripts/cnn_prepare_training_data.py</code> finished.</p>

<h2>Handoff decision matrix</h2>
<table>
<tr><th>Status</th><th>Check</th><th>Detail</th></tr>
{decisions_html}
</table>

<h2>Supervised manifest (VocalMat-only)</h2>
<p>Total rows in <code>manifest_all.csv</code>: <b>{len(manifest_all)}</b></p>
<div class="cols">
  <div>{classes_table('train', train_classes)}</div>
  <div>{classes_table('val', val_classes)}</div>
  <div>{classes_table('test', test_classes)}</div>
</div>

<h2>Domain-unlabeled patches (for Module 18.4 DANN)</h2>
<table>
<tr><td>domain_unlabeled.csv rows</td><td style='text-align:right'>{len(domain_unlabeled)}</td></tr>
<tr><td>lab patches on disk</td><td style='text-align:right'>{lab_patches}</td></tr>
<tr><td>wild patches on disk</td><td style='text-align:right'>{wild_patches}</td></tr>
</table>

<h2>Sanity-patch thumbnails (the CHECKPOINT)</h2>
<p>Open the full <code>sanity_patches/</code> dir in WSL Explorer for the rest.</p>
{thumbs_html}

<h2>What to do next</h2>
<ul>
<li>Spot-check the thumbnails above against original VocalMat PNGs and lab/wild WAV spectrograms.</li>
<li>If patches look right: append a dated entry to <code>IMPLEMENTATION_PROGRESS.md</code> + flip Module 18.2b CLOSED in <code>ops/goals.md</code>.</li>
<li>If patches look wrong (USVs missing, inverted contrast, lots of horizontal lines): STOP, iterate on cleaning stack.</li>
</ul>
</body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
