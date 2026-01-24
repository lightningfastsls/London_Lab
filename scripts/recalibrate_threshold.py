"""Recalibrate threshold for fixed-padding predictions."""
import pandas as pd

test_df = pd.read_csv('splits/test.csv')
pred_df = pd.read_csv('test_predictions_fixed.csv')

merged = test_df.merge(pred_df[['candidate_id', 'probability']], on='candidate_id')
y_true = (merged['label'] == 'USV').astype(int).values
y_proba = merged['probability'].values

print("=" * 60)
print("THRESHOLD CALIBRATION (Fixed Padding to 512px)")
print("=" * 60)

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

for th in thresholds:
    y_pred = (y_proba >= th).astype(int)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    acc = (y_pred == y_true).mean()
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"Threshold {th:.2f}: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

print("=" * 60)
