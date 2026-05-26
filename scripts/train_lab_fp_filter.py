#!/usr/bin/env python3
"""Run 6: Post-hoc false-positive filter for lab data.

Trains a binary classifier on event-level features extracted by the
PRODUCTION CNN. Uses the 571 labeled lab events (joined to
results/batch_lab_131204_full/merged_events_with_filter.parquet) for
training, with recording-stratified split matching the CNN fine-tune
(holdout = sessions 131209_1000 + 131217_1400).

This is the "production model untouched + extra event-level filter"
strategy. The filter takes event features that production CNN
generated and reclassifies USV vs noise.

Output: a sklearn classifier pickle + a JSON sidecar with metrics.

Usage:
    .venv/bin/python scripts/train_lab_fp_filter.py \
        --output-dir models/lab_finetune_v1_run6_filter/
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

LABEL_AUDIT = REPO_ROOT / "data" / "lab_finetune_v1" / "labels_audit_72.csv"
LABEL_MINING = (
    REPO_ROOT / "data" / "lab_finetune_v1"
    / "mining_candidates_500" / "candidates_seed42.csv"
)
EVENTS_PARQUET = (
    REPO_ROOT / "results" / "batch_lab_131204_full"
    / "merged_events_with_filter.parquet"
)
HELD_OUT_SESSIONS = {"131209_1000", "131217_1400"}

FEATURES = [
    "max_probability",
    "mean_probability",
    "duration_s",
    "stationary_energy_fraction",
    "n_stationary_bins",
    "noise_band_overlap",
]


def session_of(filename: str) -> str:
    stem = filename.replace(".wav", "")
    return "_".join(stem.split("_")[:-1])


def load_labels() -> pd.DataFrame:
    audit = pd.read_csv(LABEL_AUDIT)[["chunk_stem", "event_idx", "verdict"]]
    mining = pd.read_csv(LABEL_MINING)[["chunk_stem", "event_idx", "verdict"]]
    df = pd.concat([audit, mining], ignore_index=True).drop_duplicates(
        subset=["chunk_stem", "event_idx"], keep="first"
    )
    df["label"] = df["verdict"].str.upper().map({"USV": 1, "NOISE": 0})
    if df["label"].isna().any():
        raise ValueError("Unmapped labels")
    return df[["chunk_stem", "event_idx", "label"]]


def join_to_features(labels: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_parquet(EVENTS_PARQUET)
    p = p.rename(columns={"chunk_detection_idx": "event_idx"})
    keep = ["chunk_stem", "event_idx", "original_filename"] + FEATURES
    p = p[keep]
    # bool → int
    p["noise_band_overlap"] = p["noise_band_overlap"].astype(int)
    merged = labels.merge(p, on=["chunk_stem", "event_idx"], how="left")
    if merged[FEATURES[0]].isna().any():
        missing = merged[merged[FEATURES[0]].isna()]
        raise ValueError(f"{len(missing)} labels could not be joined to features")
    merged["session"] = merged["original_filename"].apply(session_of)
    return merged


def metrics(y_true: np.ndarray, y_pred: np.ndarray, p: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    total = tp + fp + tn + fn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / total if total else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "n": int(total),
        "prob_mean": float(p.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--clf", choices=["logistic", "rf"], default="rf")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("RUN 6 — Post-hoc lab FP filter")
    print("=" * 64)
    print(f"Output:           {args.output_dir}")
    print(f"Classifier:       {args.clf}")
    print(f"Features:         {FEATURES}")
    print(f"Held-out sessions: {sorted(HELD_OUT_SESSIONS)}")

    labels = load_labels()
    df = join_to_features(labels)
    print(f"Joined rows:      {len(df)} "
          f"({(df['label']==1).sum()} USV / {(df['label']==0).sum()} noise)")

    train = df[~df["session"].isin(HELD_OUT_SESSIONS)].copy()
    holdout = df[df["session"].isin(HELD_OUT_SESSIONS)].copy()
    print(f"Train:            {len(train)} "
          f"({(train['label']==1).sum()} USV / {(train['label']==0).sum()} noise)")
    print(f"Held-out:         {len(holdout)} "
          f"({(holdout['label']==1).sum()} USV / {(holdout['label']==0).sum()} noise)")

    # Build classifier
    if args.clf == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, random_state=args.seed,
                                      class_weight="balanced")),
        ])
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=args.seed,
            class_weight="balanced",
            n_jobs=-1,
        )

    X_train = train[FEATURES].values
    y_train = train["label"].values
    X_hold = holdout[FEATURES].values
    y_hold = holdout["label"].values

    clf.fit(X_train, y_train)

    # Train metrics
    p_train = clf.predict_proba(X_train)[:, 1]
    y_pred_train = (p_train >= 0.5).astype(int)
    m_train = metrics(y_train, y_pred_train, p_train)

    # Held-out metrics
    p_hold = clf.predict_proba(X_hold)[:, 1]
    y_pred_hold = (p_hold >= 0.5).astype(int)
    m_hold = metrics(y_hold, y_pred_hold, p_hold)

    # Feature importances (RF) or coefficients (LR)
    importances = {}
    if hasattr(clf, "feature_importances_"):
        importances = dict(zip(FEATURES, clf.feature_importances_.tolist()))
    elif hasattr(clf, "named_steps") and hasattr(clf.named_steps.get("lr", None), "coef_"):
        coefs = clf.named_steps["lr"].coef_[0]
        importances = dict(zip(FEATURES, coefs.tolist()))

    result = {
        "clf_type": args.clf,
        "features": FEATURES,
        "feature_importances": importances,
        "train_metrics": m_train,
        "holdout_metrics": m_hold,
        "n_train_sessions": int(train["session"].nunique()),
        "n_holdout_sessions": int(holdout["session"].nunique()),
        "held_out_sessions": sorted(HELD_OUT_SESSIONS),
    }

    # Save artifacts
    with open(args.output_dir / "lab_fp_filter.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(args.output_dir / "lab_fp_filter.json", "w") as f:
        json.dump(result, f, indent=2)

    # Per-event predictions for downstream comparison
    holdout_out = holdout.copy()
    holdout_out["prob"] = p_hold
    holdout_out["pred"] = y_pred_hold
    holdout_out.to_csv(args.output_dir / "holdout_predictions.csv", index=False)

    # Report
    print("\nTrain set metrics:")
    for k, v in m_train.items(): print(f"  {k:>12}: {v}")
    print("\nHeld-out set metrics (RECORDING-STRATIFIED):")
    for k, v in m_hold.items(): print(f"  {k:>12}: {v}")
    print("\nFeature importances:")
    for k, v in importances.items(): print(f"  {k:>30}: {v:+.4f}")
    print(f"\nSaved: {args.output_dir/'lab_fp_filter.pkl'}")
    print(f"Saved: {args.output_dir/'lab_fp_filter.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
