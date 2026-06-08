#!/usr/bin/env python3
"""Cascade / labor-division simulation from the v1 ResNet confusion matrix.

Tests Shachar's proposal: Stage-1 classifier handles the crisp types
{Noise, Down-FM, Up-FM, Chevron, Reverse-Chevron, Short, Flat}; everything that
"doesn't fit" routes to Stage-2 (the step/complex family).

We do NOT have per-call softmax here, so this is a *grouping proxy*: we collapse
the existing 12-class predictions into the two groups and read off separability.
A dedicated 7-class Stage-1 model could do better (fewer classes to confuse) and
a real reject-option would re-route low-confidence calls — so treat these numbers
as a structural estimate, not the final cascade performance.

Source: results/lab_classifier_v1/metrics.json (held-out test set).
"""
from __future__ import annotations
import json
from pathlib import Path

METRICS = Path("results/lab_classifier_v1/metrics.json")

# Verified order (matches per_class_precision: idx0=Noise 0.984, idx9=Complex 0.596)
CLASSES = ["Noise", "Step up", "Down-FM", "Short", "Chevron", "Up-FM", "Flat",
           "Two steps", "Step down", "Complex", "Reverse Chevron", "Multi-steps"]

EASY = [0, 2, 3, 4, 5, 6, 10]          # Stage-1 (user's set): Noise, Down/Up-FM, Short, Chevron, Flat, Rev-Chevron
HARD = [1, 7, 8, 9, 11]                # Stage-2 residual: Step up/down, Two steps, Complex, Multi-steps


def main() -> None:
    C = json.loads(METRICS.read_text())["confusion_matrix"]  # C[true][pred]
    n = len(C)
    total = sum(sum(row) for row in C)
    easy_set, hard_set = set(EASY), set(HARD)

    print("=" * 72)
    print("CASCADE SIMULATION  (proxy from v1 12-class confusion matrix)")
    print("=" * 72)
    print(f"Source            : {METRICS}")
    print(f"Held-out calls    : {total}")
    print(f"Stage-1 (EASY)    : {[CLASSES[i] for i in EASY]}")
    print(f"Stage-2 (HARD)    : {[CLASSES[i] for i in HARD]}")
    print(f"NOTE              : grouping proxy, not a true reject-option cascade.")
    print()

    # --- 2x2 routing matrix: did the predicted GROUP match the true GROUP? ---
    EE = EH = HE = HH = 0
    for t in range(n):
        for p in range(n):
            c = C[t][p]
            t_easy, p_easy = t in easy_set, p in easy_set
            if t_easy and p_easy:   EE += c
            elif t_easy and not p_easy: EH += c
            elif not t_easy and p_easy: HE += c
            else: HH += c
    easy_true = EE + EH
    hard_true = HE + HH
    pred_easy = EE + HE
    pred_hard = EH + HH

    print("--- ROUTING (group-level): does a call land in the right STAGE? ---")
    print(f"  truly EASY -> routed EASY : {EE:4d} / {easy_true} ({EE/easy_true:.1%})  [Stage-1 keeps these]")
    print(f"  truly EASY -> routed HARD : {EH:4d} / {easy_true} ({EH/easy_true:.1%})  [easy call wrongly sent to Stage-2]")
    print(f"  truly HARD -> routed EASY : {HE:4d} / {hard_true} ({HE/hard_true:.1%})  [LEAK: never reaches Stage-2]")
    print(f"  truly HARD -> routed HARD : {HH:4d} / {hard_true} ({HH/hard_true:.1%})  [Stage-2 catches these]")
    print(f"  overall correct-stage routing : {(EE+HH)/total:.1%}")
    print(f"  Stage-1 bucket purity (pred EASY that is truly EASY): {EE/pred_easy:.1%}")
    print(f"  fraction of all calls falling through to Stage-2     : {pred_hard/total:.1%}")
    print()

    # --- Stage-1's real job: exact class among calls it keeps ---
    diag_easy = sum(C[i][i] for i in EASY)
    print("--- STAGE-1 internal accuracy (exact type, among calls kept by Stage-1) ---")
    print(f"  exact-class correct / kept-easy : {diag_easy}/{EE} = {diag_easy/EE:.1%}")
    print(f"  (recall to correct easy type, of all truly-easy: {diag_easy/easy_true:.1%})")
    print()

    # --- Which EASY types blur into each other (within-Stage-1 confusion) ---
    print("--- WITHIN-STAGE-1 confusions (true EASY -> pred EASY, off-diagonal >=3) ---")
    pairs = []
    for t in EASY:
        for p in EASY:
            if t != p and C[t][p] >= 3:
                pairs.append((C[t][p], CLASSES[t], CLASSES[p]))
    for c, a, b in sorted(pairs, reverse=True):
        print(f"    {a:>16s} -> {b:<16s} : {c}")
    if not pairs:
        print("    (none >=3 — Stage-1 types are crisply separated)")
    print()

    # --- Per-EASY-type recall/precision within the collapsed scheme ---
    print("--- PER STAGE-1 TYPE (collapsed scheme) ---")
    print(f"    {'type':>16s}  {'recall':>7s}  {'prec':>7s}  {'n_true':>6s}")
    for i in EASY:
        row_t = sum(C[i])                       # true count
        col_p = sum(C[t][i] for t in range(n))  # predicted count
        rec = C[i][i] / row_t if row_t else 0.0
        prec = C[i][i] / col_p if col_p else 0.0
        print(f"    {CLASSES[i]:>16s}  {rec:7.1%}  {prec:7.1%}  {row_t:6d}")
    print()

    # --- Stage-2 internal mess (justifies treating residual differently) ---
    print("--- STAGE-2 residual internal confusion (true HARD -> pred HARD, off-diag >=2) ---")
    hpairs = []
    for t in HARD:
        for p in HARD:
            if t != p and C[t][p] >= 2:
                hpairs.append((C[t][p], CLASSES[t], CLASSES[p]))
    for c, a, b in sorted(hpairs, reverse=True):
        print(f"    {a:>16s} -> {b:<16s} : {c}")
    diag_hard = sum(C[i][i] for i in HARD)
    print(f"  Stage-2 exact-class accuracy (if it had to name the type): "
          f"{diag_hard}/{hard_true} = {diag_hard/hard_true:.1%}")
    print()
    print("=" * 72)
    print("READ: high correct-stage routing + clean Stage-1 + messy Stage-2 =")
    print("      labor-division is sound; Stage-2 is where types blur (continuum).")
    print("=" * 72)


if __name__ == "__main__":
    main()
