#!/usr/bin/env python3
"""hand_label_200.py — γ hand-label tool (α₃-C Phase A7).

PURPOSE
-------
A single-screen PyQt6 tool for the USER to hand-label ~200 spectrogram patches
by SHAPE. The output (``data/manual_shape_labels.csv``) is a PERMANENT external
eval anchor: a substrate-independent human ground truth for shape clustering.
It ships regardless of whether the α₃ VAE (A6) succeeds — if the VAE is killed,
this becomes the human gold standard the registration baseline is judged against
(see the decision gate in
``docs/handoffs/2026-05-29_alpha3-C-execution-A3-A8.md``).

WHERE THE PATCHES COME FROM
---------------------------
PREFERRED: point ``--manifest`` at the A3 render output
``data/alpha3_patches/manifest.csv`` (columns ``call_id, path`` minimum, as
produced by ``scripts/experiments/render_vocalmat_style_patches.py``). When a
manifest is supplied we sample *from it* and never re-derive render logic. If the
manifest carries a ``cohort`` and/or ``slope`` column those are used directly for
stratification; otherwise cohort is inferred from the ``wav_stem`` / ``call_id``
naming convention and slope-sign stratification is skipped with a warning.

FALLBACK (no manifest): we sample directly from the per-cohort
``classified_detections*.csv`` tables so the tool still produces a deterministic,
stratified *sample plan*. In that mode there are no rendered PNGs to display, so
the GUI cannot run — use ``--no-gui-selftest`` (or supply a manifest) in that
case. The sample plan is still written so a later render step can fulfil it.

SAMPLING
--------
200 patches, stratified two ways:
  1. by COHORT — 50 each from {5970, 3452, 9252, lab_131204} (``--per-cohort``),
     falling back gracefully to whatever cohorts are actually available.
  2. within each cohort, by dominant slope-sign bucket
     {down (slope<-T), flat (|slope|<=T), up (slope>T)} so chevrons / jumps /
     flats / complex calls all appear and one shape can't dominate. ``slope`` is
     read from the cohort CSV (or manifest if present). Threshold T is
     ``--slope-flat-threshold`` (Hz/ms-ish units as stored in the CSV).

Sampling is deterministic for a given ``--seed`` (repo rule). All parameters and
per-cohort / per-slope-bucket counts are printed at startup.

OUTPUT
------
``data/manual_shape_labels.csv`` with columns:
    call_id, cohort, shape_label, labeled_at_index
``call_id`` is the join key for A8 (oracle ↔ γ cross-tab). Autosaved after every
keypress. On relaunch, already-labeled ``call_id``\\s are skipped (resume).

LABELS (keypress) — the FULL 12-class VocalMat taxonomy (same as the oracle)
----------------------------------------------------------------------------
    C = Chevron (∧)      R = Reverse Chevron (∨ / upside-down-U / valley)
    U = Up-FM            D = Down-FM
    1 = Step up          2 = Step down      3 = Two steps    4 = Multi-steps
    F = Flat             X = Complex        S = Short        N = Noise
    ? = unclear (human escape hatch; excluded from A8 via --exclude-unclear)

    NOTE: γ now mirrors VocalMat's 12 classes so A8 is a direct human-vs-oracle
          agreement (confusion matrix / κ). [U] = Up-FM sweep, NOT "U-shape" —
          the valley is [R] Reverse Chevron. The 12 can be folded to coarse shape
          families for A6 downstream (Chevron+Reverse Chevron→chevron, etc.).

CONSTRAINTS (do NOT violate)
----------------------------
- 🔒 Does not edit corpus.py / Stack 4 / models/ / results/lab_classifier_v1/ /
  the production pipeline. Corpus constants are imported, never redeclared.
- argparse: --manifest --n --per-cohort --output --seed
  plus --slope-flat-threshold, --cohort-csv (override mapping), --no-gui-selftest.

VERIFY (headless)
-----------------
    cd /home/shachar/projects/mickey_london_lab && \\
      PYTHONPATH=src .venv/bin/python -m py_compile scripts/labeling/hand_label_200.py
    PYTHONPATH=src .venv/bin/python scripts/labeling/hand_label_200.py --no-gui-selftest

USER COMMAND (to actually label)
--------------------------------
    cd /home/shachar/projects/mickey_london_lab && \\
      PYTHONPATH=src .venv/bin/python scripts/labeling/hand_label_200.py \\
        --manifest data/alpha3_patches/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Repo root resolution (script lives at scripts/labeling/hand_label_200.py)
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------------------
# Label schema — single source of truth for keypress → label.
#
# 2026-05-30: switched from the coarse 8-family shape scheme to the FULL 12-class
# VocalMat taxonomy (GRIMSLEY_12_CLASSES) so the γ human labels use the SAME label
# space as the v1-ResNet oracle. This makes A8 a direct human-vs-machine agreement
# (confusion matrix / κ on a shared 12-class axis), and the 12 can still be folded
# to coarse shape families for A6 (capture-finer, fold-later). 'unclear' is a human
# escape hatch (not a VocalMat class; excluded from A8 via --exclude-unclear).
#
# Step/jump family = number keys 1–4; everything else = mnemonic letter.
# Chevron = ∧ peak; Reverse Chevron = ∨ upside-down-U / valley; U = Up-FM sweep.
# ----------------------------------------------------------------------------
LABELS: Dict[str, str] = {
    "C": "Chevron",
    "R": "Reverse Chevron",
    "U": "Up-FM",
    "D": "Down-FM",
    "1": "Step up",
    "2": "Step down",
    "3": "Two steps",
    "4": "Multi-steps",
    "F": "Flat",
    "X": "Complex",
    "S": "Short",
    "N": "Noise",
    "?": "unclear",
}

# Drift guard: the 12 non-'unclear' labels MUST equal the canonical taxonomy the
# oracle predicts, or the A8 confusion matrix silently mis-aligns. Soft-checked so
# the lightweight GUI still runs if the (heavier) classifier module can't import.
_VOCALMAT_12 = (
    "Noise", "Step up", "Down-FM", "Short", "Chevron", "Up-FM", "Flat",
    "Two steps", "Step down", "Complex", "Reverse Chevron", "Multi-steps",
)
assert set(v for v in LABELS.values() if v != "unclear") == set(_VOCALMAT_12), (
    "γ LABELS drifted from the VocalMat 12-class taxonomy"
)
try:  # best-effort: assert against the live canonical tuple if importable
    from usv_spectrogram.classifier.dataset import GRIMSLEY_12_CLASSES as _G12
    assert set(_VOCALMAT_12) == set(_G12), (
        f"VocalMat taxonomy drift: tool={set(_VOCALMAT_12)} vs canonical={set(_G12)}"
    )
except ImportError:
    pass  # GUI runs on the hardcoded names; sync re-checked wherever src is importable

OUTPUT_COLUMNS = ["call_id", "cohort", "shape_label", "labeled_at_index"]

# Default cohort -> classified-detections CSV (relative to repo root).
# 5970 == classified_detections.csv (the original wild dyad export).
DEFAULT_COHORT_CSVS: Dict[str, str] = {
    "5970": "classified_detections.csv",
    "3452": "classified_detections_3452.csv",
    "9252": "classified_detections_9252.csv",
    "lab_131204": "classified_detections_lab_131204_clean.csv",
}

# Slope-sign buckets used for within-cohort stratification.
SLOPE_BUCKETS = ("down", "flat", "up")


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------
@dataclass
class Patch:
    """One labelable item."""

    call_id: str
    cohort: str
    path: Optional[str] = None          # PNG path (only when from a manifest)
    slope: Optional[float] = None
    slope_bucket: Optional[str] = None


@dataclass
class SampleResult:
    patches: List[Patch] = field(default_factory=list)
    per_cohort: Dict[str, int] = field(default_factory=dict)
    per_bucket: Dict[str, Dict[str, int]] = field(default_factory=dict)
    missing_cohorts: List[str] = field(default_factory=list)
    source: str = ""                     # "manifest" or "cohort_csv"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _resolve(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def slope_to_bucket(slope: Optional[float], flat_threshold: float) -> Optional[str]:
    if slope is None:
        return None
    if slope < -flat_threshold:
        return "down"
    if slope > flat_threshold:
        return "up"
    return "flat"


def infer_cohort_from_stem(stem: str) -> Optional[str]:
    """Best-effort cohort inference from a wav_stem / call_id string.

    Wild dyad files look like ``2024-09-30_11-18-17_0000001``; lab files look
    like ``131204_1400_m1fm1_chunk_022`` (and other lab dates 131205/131208…).
    We can't always tell 5970 vs 3452 vs 9252 apart from the filename alone, so
    this only confidently recognises the lab cohort and otherwise returns None
    (the caller treats None as "unknown / can't stratify by cohort").
    """
    if "chunk_" in stem and ("fm" in stem or "_m" in stem):
        return "lab_131204"
    return None


def make_call_id(row: Dict[str, str]) -> Optional[str]:
    """Build a stable, unique call_id from a classified-detections row.

    wav_stem is NOT unique (many detections share one file), so we disambiguate
    with det_index. Mirrors the join key A8 will reconstruct from the manifest.
    """
    stem = (row.get("wav_stem") or "").strip()
    if not stem:
        return None
    det_index = (row.get("det_index") or "").strip()
    if det_index:
        # det_index may be stored as a float-ish string ("0.0"); normalise.
        f = _parse_float(det_index)
        idx = str(int(f)) if f is not None else det_index
        return f"{stem}__{idx}"
    return stem


# ----------------------------------------------------------------------------
# Sampling — manifest path
# ----------------------------------------------------------------------------
def _read_manifest_rows(manifest_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with manifest_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = [dict(r) for r in reader]
    return rows, fieldnames


def sample_from_manifest(
    manifest_path: Path,
    n: int,
    per_cohort: int,
    seed: int,
    flat_threshold: float,
) -> SampleResult:
    rows, fieldnames = _read_manifest_rows(manifest_path)
    if "call_id" not in fieldnames:
        raise ValueError(
            f"Manifest {manifest_path} must have a 'call_id' column; got {fieldnames}"
        )
    if "path" not in fieldnames:
        raise ValueError(
            f"Manifest {manifest_path} must have a 'path' column; got {fieldnames}"
        )

    has_cohort = "cohort" in fieldnames
    has_slope = "slope" in fieldnames

    # Build candidate patches, deduplicated by call_id.
    by_cohort: Dict[str, List[Patch]] = {}
    seen: set = set()
    for row in rows:
        call_id = (row.get("call_id") or "").strip()
        if not call_id or call_id in seen:
            continue
        seen.add(call_id)
        cohort = (row.get("cohort") or "").strip() if has_cohort else ""
        if not cohort:
            cohort = infer_cohort_from_stem(call_id) or "unknown"
        slope = _parse_float(row.get("slope")) if has_slope else None
        bucket = slope_to_bucket(slope, flat_threshold)
        patch = Patch(
            call_id=call_id,
            cohort=cohort,
            path=(row.get("path") or "").strip(),
            slope=slope,
            slope_bucket=bucket,
        )
        by_cohort.setdefault(cohort, []).append(patch)

    return _stratified_select(
        by_cohort, n, per_cohort, seed, source="manifest", slope_aware=has_slope
    )


# ----------------------------------------------------------------------------
# Sampling — cohort-CSV fallback
# ----------------------------------------------------------------------------
def sample_from_cohort_csvs(
    cohort_csvs: Dict[str, str],
    n: int,
    per_cohort: int,
    seed: int,
    flat_threshold: float,
) -> SampleResult:
    by_cohort: Dict[str, List[Patch]] = {}
    missing: List[str] = []

    for cohort, csv_rel in cohort_csvs.items():
        csv_path = _resolve(csv_rel)
        if not csv_path.exists():
            missing.append(cohort)
            continue
        patches: List[Patch] = []
        seen: set = set()
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                call_id = make_call_id(row)
                if not call_id or call_id in seen:
                    continue
                seen.add(call_id)
                slope = _parse_float(row.get("slope"))
                patches.append(
                    Patch(
                        call_id=call_id,
                        cohort=cohort,
                        path=None,  # not rendered in fallback mode
                        slope=slope,
                        slope_bucket=slope_to_bucket(slope, flat_threshold),
                    )
                )
        if patches:
            by_cohort[cohort] = patches
        else:
            missing.append(cohort)

    return _stratified_select(
        by_cohort, n, per_cohort, seed, source="cohort_csv", slope_aware=True
    )


# ----------------------------------------------------------------------------
# Shared stratified selection
# ----------------------------------------------------------------------------
def _stratified_select(
    by_cohort: Dict[str, List[Patch]],
    n: int,
    per_cohort: int,
    seed: int,
    source: str,
    slope_aware: bool,
) -> SampleResult:
    """Select up to `per_cohort` patches per cohort, balanced across slope buckets.

    Robust to missing cohorts and to cohorts with fewer than `per_cohort`
    available patches: takes what's there and logs the shortfall. If the summed
    per-cohort target is below `n` (few cohorts available), redistributes the
    remaining budget across the cohorts that *do* have spare patches so we still
    approach `n` total.
    """
    rng = random.Random(seed)
    result = SampleResult(source=source)

    cohorts = sorted(by_cohort.keys())
    if not cohorts:
        return result

    # First pass: per_cohort target each, slope-balanced within cohort.
    selected: Dict[str, List[Patch]] = {}
    for cohort in cohorts:
        pool = list(by_cohort[cohort])
        rng.shuffle(pool)
        picked = _pick_balanced_by_bucket(pool, per_cohort, rng, slope_aware)
        selected[cohort] = picked

    # Second pass: if total < n and there's spare capacity, top up round-robin.
    total = sum(len(v) for v in selected.values())
    if total < n:
        # Build remaining pools (exclude already-picked call_ids).
        picked_ids = {p.call_id for v in selected.values() for p in v}
        remaining: Dict[str, List[Patch]] = {}
        for cohort in cohorts:
            extra = [p for p in by_cohort[cohort] if p.call_id not in picked_ids]
            rng.shuffle(extra)
            remaining[cohort] = extra
        # Round-robin across cohorts until we hit n or run dry.
        idx = 0
        while total < n and any(remaining[c] for c in cohorts):
            cohort = cohorts[idx % len(cohorts)]
            idx += 1
            if remaining[cohort]:
                selected[cohort].append(remaining[cohort].pop())
                total += 1

    # Flatten + record stats.
    for cohort in cohorts:
        items = selected[cohort]
        result.patches.extend(items)
        result.per_cohort[cohort] = len(items)
        bucket_counts = {b: 0 for b in SLOPE_BUCKETS}
        bucket_counts["unknown"] = 0
        for p in items:
            bucket_counts[p.slope_bucket if p.slope_bucket else "unknown"] += 1
        result.per_bucket[cohort] = bucket_counts

    # Deterministic global order: sort by (cohort, call_id) so resume is stable.
    result.patches.sort(key=lambda p: (p.cohort, p.call_id))
    return result


def _pick_balanced_by_bucket(
    pool: List[Patch],
    target: int,
    rng: random.Random,
    slope_aware: bool,
) -> List[Patch]:
    if target <= 0 or not pool:
        return []
    if not slope_aware:
        return pool[: min(target, len(pool))]

    # Group by slope bucket (None -> 'unknown').
    groups: Dict[str, List[Patch]] = {}
    for p in pool:
        key = p.slope_bucket if p.slope_bucket else "unknown"
        groups.setdefault(key, []).append(p)
    for g in groups.values():
        rng.shuffle(g)

    bucket_order = [b for b in SLOPE_BUCKETS if b in groups]
    if "unknown" in groups:
        bucket_order.append("unknown")
    if not bucket_order:
        return pool[: min(target, len(pool))]

    picked: List[Patch] = []
    # Round-robin draw across buckets for even representation.
    while len(picked) < target and any(groups[b] for b in bucket_order):
        for b in bucket_order:
            if len(picked) >= target:
                break
            if groups[b]:
                picked.append(groups[b].pop())
    return picked


# ----------------------------------------------------------------------------
# Output CSV handling (autosave + resume)
# ----------------------------------------------------------------------------
def read_existing_labels(output_path: Path) -> Dict[str, str]:
    """Return {call_id: shape_label} already present in the output CSV."""
    if not output_path.exists():
        return {}
    labeled: Dict[str, str] = {}
    with output_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = (row.get("call_id") or "").strip()
            if cid:
                labeled[cid] = (row.get("shape_label") or "").strip()
    return labeled


def write_scaffold(output_path: Path) -> None:
    """Create an empty (header-only) output CSV if it does not exist."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        with output_path.open("w", newline="") as f:
            csv.writer(f).writerow(OUTPUT_COLUMNS)


def append_label(
    output_path: Path,
    call_id: str,
    cohort: str,
    shape_label: str,
    labeled_at_index: int,
) -> None:
    """Append (or rewrite) a single label, autosaving immediately.

    If the call_id already exists (a re-label / correction), the file is
    rewritten with the updated value; otherwise the row is appended.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: List[Dict[str, str]] = []
    if output_path.exists():
        with output_path.open(newline="") as f:
            existing_rows = [dict(r) for r in csv.DictReader(f)]

    found = False
    for row in existing_rows:
        if (row.get("call_id") or "").strip() == call_id:
            row["cohort"] = cohort
            row["shape_label"] = shape_label
            row["labeled_at_index"] = str(labeled_at_index)
            found = True
            break

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({k: row.get(k, "") for k in OUTPUT_COLUMNS})
        if not found:
            writer.writerow(
                {
                    "call_id": call_id,
                    "cohort": cohort,
                    "shape_label": shape_label,
                    "labeled_at_index": str(labeled_at_index),
                }
            )


# ----------------------------------------------------------------------------
# Startup reporting
# ----------------------------------------------------------------------------
def print_startup(args: argparse.Namespace, sample: SampleResult) -> None:
    print("=" * 70)
    print("hand_label_200.py  —  γ hand-label tool (α₃-C Phase A7)")
    print("=" * 70)
    print("Parameters:")
    print(f"  manifest             : {args.manifest}")
    print(f"  n (target)           : {args.n}")
    print(f"  per_cohort           : {args.per_cohort}")
    print(f"  slope_flat_threshold : {args.slope_flat_threshold}")
    print(f"  seed                 : {args.seed}")
    print(f"  output               : {args.output}")
    print(f"  no_gui_selftest      : {args.no_gui_selftest}")
    print(f"  sample source        : {sample.source}")
    print("-" * 70)
    print(f"Sampled {len(sample.patches)} patches.")
    if sample.missing_cohorts:
        print(f"  MISSING cohorts (no CSV / no rows): {sample.missing_cohorts}")
    print("  Per-cohort counts:")
    for cohort in sorted(sample.per_cohort):
        buckets = sample.per_bucket.get(cohort, {})
        bucket_str = ", ".join(
            f"{b}={buckets.get(b, 0)}" for b in (*SLOPE_BUCKETS, "unknown")
        )
        print(f"    {cohort:<14} n={sample.per_cohort[cohort]:<4} [{bucket_str}]")
    print("=" * 70)
    sys.stdout.flush()


# ----------------------------------------------------------------------------
# GUI (PyQt6) — imported lazily so non-GUI paths work without a display.
# ----------------------------------------------------------------------------
def run_gui(sample: SampleResult, output_path: Path, args: argparse.Namespace) -> int:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QVBoxLayout,
        QWidget,
    )

    # Resume: filter out already-labeled call_ids.
    already = read_existing_labels(output_path)
    todo = [p for p in sample.patches if p.call_id not in already]
    n_done = len(sample.patches) - len(todo)

    if not todo:
        print(f"All {len(sample.patches)} patches already labeled in {output_path}.")
        return 0

    legend_text = "   ".join(f"[{k}] {v}" for k, v in LABELS.items())
    # Disambiguation guide for the 12-class VocalMat scheme: the valley shape is
    # its own class [R] Reverse Chevron (∨ / upside-down-U); [C] Chevron is the
    # ∧ peak only; [U] is the Up-FM sweep, NOT "U-shape"; the step family lives on
    # the number keys 1–4. (User switched 8→12 classes to match VocalMat for a
    # direct system-vs-system comparison, 2026-05-30.)
    legend_text += (
        "\n\n[C] Chevron = ∧ peak.    [R] Reverse Chevron = ∨ upside-down-U / "
        "valley.    [U] = Up-FM sweep (NOT 'U-shape').    Steps = number keys 1–4."
    )

    class LabelWindow(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.patches = todo
            self.total = len(sample.patches)
            self.base_done = n_done
            self.pos = 0  # index into self.patches
            self.setWindowTitle("γ hand-label — shape")
            self._build_ui()
            self._show_current()

        def _build_ui(self) -> None:
            root = QVBoxLayout(self)

            top = QHBoxLayout()
            self.progress = QLabel()
            self.progress.setStyleSheet("font-size: 18px; font-weight: bold;")
            self.cohort_lbl = QLabel()
            self.cohort_lbl.setStyleSheet("font-size: 16px; color: #555;")
            top.addWidget(self.progress)
            top.addStretch(1)
            top.addWidget(self.cohort_lbl)
            root.addLayout(top)

            self.image = QLabel("(no image)")
            self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image.setMinimumSize(700, 600)
            self.image.setStyleSheet("background:#111;")
            root.addWidget(self.image, stretch=1)

            self.legend = QLabel(legend_text)
            self.legend.setWordWrap(True)
            self.legend.setStyleSheet("font-size: 15px; padding: 6px;")
            root.addWidget(self.legend)

            self.hint = QLabel(
                "Press a label key to assign + autosave + advance.   "
                "← go back to fix.   Esc / Q to quit (progress saved)."
            )
            self.hint.setStyleSheet("font-size: 12px; color: #777;")
            root.addWidget(self.hint)

            self.resize(820, 780)

        def _current_patch(self) -> Optional[Patch]:
            if 0 <= self.pos < len(self.patches):
                return self.patches[self.pos]
            return None

        def _show_current(self) -> None:
            patch = self._current_patch()
            if patch is None:
                print("Done — all assigned patches labeled.")
                self.close()
                return
            done_count = self.base_done + self.pos
            self.progress.setText(f"{done_count + 1} / {self.total}")
            existing = read_existing_labels(output_path).get(patch.call_id, "")
            extra = f"   (current: {existing})" if existing else ""
            self.cohort_lbl.setText(f"{patch.cohort}   |   {patch.call_id}{extra}")

            if patch.path:
                img_path = _resolve(patch.path)
                pix = QPixmap(str(img_path))
                if pix.isNull():
                    self.image.setText(f"(could not load image)\n{img_path}")
                else:
                    self.image.setPixmap(
                        pix.scaled(
                            self.image.size(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
            else:
                self.image.setText(
                    f"(no rendered PNG for this patch)\n{patch.call_id}\n"
                    "Run with a --manifest pointing at rendered patches."
                )

        def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
            key = event.key()
            text = event.text().upper()

            if key in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
                print("Quit requested — progress is saved. Bye.")
                self.close()
                return
            if key == Qt.Key.Key_Left:
                if self.pos > 0:
                    self.pos -= 1
                    self._show_current()
                return
            if key == Qt.Key.Key_Right:
                if self.pos < len(self.patches) - 1:
                    self.pos += 1
                    self._show_current()
                return

            # '?' arrives as text on most layouts; map it explicitly too.
            if event.text() == "?":
                text = "?"
            if text in LABELS:
                patch = self._current_patch()
                if patch is None:
                    return
                done_count = self.base_done + self.pos
                append_label(
                    output_path,
                    call_id=patch.call_id,
                    cohort=patch.cohort,
                    shape_label=LABELS[text],
                    labeled_at_index=done_count,
                )
                # Advance.
                if self.pos < len(self.patches) - 1:
                    self.pos += 1
                    self._show_current()
                else:
                    n_labeled = len(read_existing_labels(output_path))
                    print(
                        f"Labeled the last patch. {n_labeled}/{self.total} done. "
                        f"Output: {output_path}"
                    )
                    self.close()

    app = QApplication.instance() or QApplication(sys.argv)
    win = LabelWindow()
    win.show()
    return app.exec()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="γ hand-label tool — hand-label ~200 spectrogram patches by shape.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Patch manifest CSV (call_id, path[, cohort, slope]). Preferred input; "
        "from the A3 render at data/alpha3_patches/manifest.csv.",
    )
    p.add_argument("--n", type=int, default=200, help="Total patches to sample.")
    p.add_argument(
        "--per-cohort",
        type=int,
        default=50,
        help="Patches per cohort (cohort stratification target).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="data/manual_shape_labels.csv",
        help="Output CSV (call_id, cohort, shape_label, labeled_at_index).",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="RNG seed (deterministic sampling)."
    )
    p.add_argument(
        "--slope-flat-threshold",
        type=float,
        default=100.0,
        help="|slope| <= threshold => 'flat' bucket; else up/down "
        "(units match the CSV 'slope' column).",
    )
    p.add_argument(
        "--cohort-csv",
        action="append",
        default=None,
        metavar="COHORT=PATH",
        help="Override/add a cohort source CSV (repeatable). "
        "Default mapping covers 5970/3452/9252/lab_131204.",
    )
    p.add_argument(
        "--no-gui-selftest",
        action="store_true",
        help="Build the sample + write a 0-row CSV scaffold, then exit "
        "(no QApplication). For headless verification.",
    )
    return p


def resolve_cohort_csvs(overrides: Optional[List[str]]) -> Dict[str, str]:
    mapping = dict(DEFAULT_COHORT_CSVS)
    if overrides:
        for item in overrides:
            if "=" not in item:
                raise SystemExit(f"--cohort-csv expects COHORT=PATH, got: {item!r}")
            cohort, path = item.split("=", 1)
            mapping[cohort.strip()] = path.strip()
    return mapping


def build_sample(args: argparse.Namespace) -> SampleResult:
    if args.manifest:
        manifest_path = _resolve(args.manifest)
        if not manifest_path.exists():
            raise SystemExit(
                f"--manifest not found: {manifest_path}\n"
                "Run the A3 render (scripts/experiments/render_vocalmat_style_patches.py) "
                "first, or omit --manifest to sample a plan from the cohort CSVs."
            )
        return sample_from_manifest(
            manifest_path,
            n=args.n,
            per_cohort=args.per_cohort,
            seed=args.seed,
            flat_threshold=args.slope_flat_threshold,
        )

    cohort_csvs = resolve_cohort_csvs(args.cohort_csv)
    sample = sample_from_cohort_csvs(
        cohort_csvs,
        n=args.n,
        per_cohort=args.per_cohort,
        seed=args.seed,
        flat_threshold=args.slope_flat_threshold,
    )
    # Record which configured cohorts produced nothing.
    produced = set(sample.per_cohort)
    sample.missing_cohorts = [c for c in cohort_csvs if c not in produced]
    return sample


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_path = _resolve(args.output)

    sample = build_sample(args)
    print_startup(args, sample)

    if not sample.patches:
        print("No patches sampled — nothing to label. Exiting.")
        return 1

    if args.no_gui_selftest:
        write_scaffold(output_path)
        print(f"[selftest] Wrote 0-row scaffold to {output_path}")
        print("[selftest] Non-GUI paths OK (sampling + scaffold). Exiting before GUI.")
        return 0

    # Manifest is required to actually display images.
    if sample.source != "manifest" or not any(p.path for p in sample.patches):
        print(
            "\nNo rendered patch images available (sampled from cohort CSVs, not a "
            "manifest).\nThe GUI needs PNGs — supply --manifest data/alpha3_patches/"
            "manifest.csv after the A3 render.\nA stratified sample PLAN was computed "
            "above; re-run with --no-gui-selftest to also write the CSV scaffold."
        )
        return 2

    write_scaffold(output_path)
    return run_gui(sample, output_path, args)


if __name__ == "__main__":
    raise SystemExit(main())
