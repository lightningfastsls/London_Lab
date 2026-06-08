#!/usr/bin/env python3
"""Build an HTML gallery of VocalMat labeled images for human audit.

Shachar flagged the VocalMat labels as unreliable. This samples N images per
class (deterministic seed), copies them into a self-contained gallery folder
with stable-ID filenames, and lays them out grouped by the proposed cascade:
  Stage-1 (crisp) types first, then the Stage-2 (step/complex) residual.

Goal: let the user SEE (a) where labels are wrong, (b) which types are crisp vs
which blur — to inform the Stage-1 grouping. Pure viewing; no model touched.
"""
from __future__ import annotations
import random
import shutil
from pathlib import Path

SRC = Path("data/vocalmat_full")
OUT = Path("results/label_audit_gallery")
N_PER_CLASS = 48
SEED = 1729

# (manifest_dirname, display_name, stage)
LAYOUT = [
    ("noise",       "Noise",            "Stage-1 (crisp)"),
    ("short",       "Short",            "Stage-1 (crisp)"),
    ("flat",        "Flat",             "Stage-1 (crisp)"),
    ("down_fm",     "Down-FM",          "Stage-1 (crisp)"),
    ("up_fm",       "Up-FM",            "Stage-1 (crisp)"),
    ("chevron",     "Chevron",          "Stage-1 (crisp)"),
    ("rev_chevron", "Reverse Chevron",  "Stage-1 (crisp)"),
    ("step_up",     "Step up",          "Stage-2 (residual)"),
    ("step_down",   "Step down",        "Stage-2 (residual)"),
    ("two_steps",   "Two steps",        "Stage-2 (residual)"),
    ("mult_steps",  "Multi-steps",      "Stage-2 (residual)"),
    ("complex",     "Complex",          "Stage-2 (residual)"),
]


def main() -> None:
    rng = random.Random(SEED)
    if OUT.exists():
        shutil.rmtree(OUT)
    img_dir = OUT / "img"
    img_dir.mkdir(parents=True)

    print(f"Source     : {SRC}")
    print(f"Out        : {OUT}")
    print(f"N per class: {N_PER_CLASS}   seed: {SEED}")
    print()

    sampled: dict[str, list[str]] = {}
    for dirname, _disp, _stage in LAYOUT:
        cdir = SRC / dirname
        files = sorted(p.name for p in cdir.glob("*.png"))
        take = min(N_PER_CLASS, len(files))
        chosen = rng.sample(files, take) if take < len(files) else files
        local_names = []
        for i, fn in enumerate(chosen):
            stable = f"{dirname}__{i:03d}__{fn}"
            shutil.copy(cdir / fn, img_dir / stable)
            local_names.append(stable)
        sampled[dirname] = local_names
        print(f"  {dirname:>14s}: sampled {take:3d} / {len(files)}")

    # --- emit HTML ---
    css = """
    body{font-family:system-ui,Arial,sans-serif;background:#111;color:#eee;margin:0;padding:20px}
    h1{font-size:20px} h2{margin-top:34px;border-bottom:2px solid #444;padding-bottom:6px}
    .stage{font-size:13px;color:#9cf;text-transform:uppercase;letter-spacing:1px;margin-top:40px}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px}
    .cell{background:#1c1c1c;border:1px solid #333;border-radius:4px;padding:3px}
    .cell img{width:100%;display:block;border-radius:2px}
    .cap{font-size:10px;color:#888;word-break:break-all;margin-top:2px}
    .count{color:#fc6;font-weight:bold}
    """
    parts = [f"<!doctype html><html><head><meta charset='utf-8'>",
             f"<title>VocalMat label audit</title><style>{css}</style></head><body>",
             "<h1>VocalMat label audit &mdash; grouped by proposed cascade</h1>",
             f"<p>{N_PER_CLASS} samples/class (seed {SEED}). "
             "Look for: (1) mislabeled images, (2) which types blur into each other. "
             "Stage-1 types should look crisp; Stage-2 (step/complex) is expected to blur.</p>"]

    last_stage = None
    for dirname, disp, stage in LAYOUT:
        if stage != last_stage:
            parts.append(f"<div class='stage'>{stage}</div>")
            last_stage = stage
        names = sampled[dirname]
        parts.append(f"<h2>{disp} <span class='count'>(showing {len(names)})</span></h2>")
        parts.append("<div class='grid'>")
        for nm in names:
            parts.append(f"<div class='cell'><img loading='lazy' src='img/{nm}'>"
                         f"<div class='cap'>{nm.split('__',2)[2]}</div></div>")
        parts.append("</div>")
    parts.append("</body></html>")

    html_path = OUT / "index.html"
    html_path.write_text("\n".join(parts))
    print()
    print(f"Gallery: {html_path}  ({sum(len(v) for v in sampled.values())} images)")


if __name__ == "__main__":
    main()
