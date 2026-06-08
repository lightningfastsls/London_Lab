"""
Empirical retrospective probe — score EVERY pre-computed shape-clustering label set on
BOTH the (circular) shape-eta2 metric AND non-circular chevron-separation metrics.

Goal: test the eval-critique hypothesis that shape-eta2 is near-tautological for
registration (it grades partitions in the exact space registration's KMeans optimizes),
and find out whether ANY representation actually makes chevrons neighbors of chevrons.

All inputs read-only. Pure CPU analysis on the 13 MB true_registered_ridges.npz.
Matches eta2() from archive/.../shape_registered_clustering.py exactly.
"""
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

NPZ = "/home/shachar/.claude/jobs/123a8338/tmp/shape_pilot/true_registered_ridges.npz"
SEED = 42
KNN_K = 10
KNN_SUBSAMPLE = 12000   # OOM-safe neighbor query on the box

print("="*90)
print("PARAMETERS")
print(f"  npz            = {NPZ}")
print(f"  seed           = {SEED}")
print(f"  kNN k          = {KNN_K}  (chevron-retrieval purity)")
print(f"  kNN subsample  = {KNN_SUBSAMPLE}  (box OOM guard)")
print("  eta2 def       = 1 - within_SS/total_SS on registered ridge Sh (matches repo)")
print("="*90)

d = np.load(NPZ, allow_pickle=True)
Sh   = d["shapes"].astype(np.float64)          # (N,50) registered ridge
pit  = d["pitch"].astype(np.float64)
dur  = d["duration"].astype(np.float64)
cv   = d["chevron_valley"].astype(str)         # 'chevron' / 'valley' / 'other'(?)
N = len(Sh)
print(f"[INFO] N = {N} registered ridges, dim = {Sh.shape[1]}")
print(f"[INFO] chevron_valley value counts: " +
      ", ".join(f"{v}={int((cv==v).sum())}" for v in np.unique(cv)))

# label sets present in the file (pre-computed clusterings)
LABEL_KEYS = [k for k in d.keys() if k.startswith("lab_")]
print(f"[INFO] pre-computed label sets found: {LABEL_KEYS}")


def eta2(values, labels):
    v = values if values.ndim == 2 else values[:, None]
    grand = v.mean(0)
    tot = float(((v - grand) ** 2).sum())
    within = 0.0
    for lab in np.unique(labels):
        m = labels == lab
        within += float(((v[m] - v[m].mean(0)) ** 2).sum())
    return 1.0 - within / tot if tot > 0 else 0.0


# ---- representation vectors for the NON-CIRCULAR kNN test --------------------
# These are the substrate spaces; we ask "are chevrons neighbours of chevrons HERE?"
def srvf(x):  # square-root velocity function: q = sign(f') sqrt(|f'|)
    g = np.gradient(x, axis=1)
    return np.sign(g) * np.sqrt(np.abs(g))

REPS = {
    "registered_shape": Sh,
    "derivative_dFdt":  np.diff(Sh, axis=1),
    "curvature_d2":     np.diff(Sh, n=2, axis=1),
    "srvf":             srvf(Sh),
}

# chevron binary target (substrate-independent heuristic from the file)
is_chev = (cv == "chevron").astype(int)
chev_rate = is_chev.mean()
print(f"[INFO] chevron base rate = {chev_rate:.4f}")

rng = np.random.default_rng(SEED)


def knn_chevron_purity(X, y, k=KNN_K, sub=KNN_SUBSAMPLE):
    """Leave-one-out: fraction of a point's k nearest neighbours sharing its chevron flag,
    plus balanced accuracy of majority-vote neighbour prediction. Higher purity than base
    rate => chevrons genuinely cluster together in this representation."""
    n = len(X)
    idx = rng.choice(n, size=min(sub, n), replace=False)
    Xs, ys = X[idx], y[idx]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xs)
    _, nbr = nn.kneighbors(Xs)
    nbr = nbr[:, 1:]                      # drop self
    same = (ys[nbr] == ys[:, None]).mean()
    vote = (ys[nbr].mean(1) >= 0.5).astype(int)
    bal = balanced_accuracy_score(ys, vote)
    # chevron-only retrieval precision: among neighbours of chevrons, frac that are chevron
    chev_mask = ys == 1
    chev_ret = ys[nbr[chev_mask]].mean() if chev_mask.any() else float("nan")
    return same, bal, chev_ret


print("\n" + "="*90)
print("PART 1 — CIRCULAR metric (shape eta2) + invariance, per pre-computed label set")
print("         shape eta2 high = compact in registered-ridge space (registration's own objective)")
print("         pitch/dur eta2 LOW = good pitch/duration invariance")
print("="*90)
print(f"{'label_set':<24}{'shape_eta2':>11}{'pitch_eta2':>11}{'dur_eta2':>10}{'chev_NMI':>10}{'n_clusters':>11}")
rows = {}
for k in LABEL_KEYS:
    lab = d[k]
    se = eta2(Sh, lab)
    pe = eta2(pit, lab)
    de = eta2(dur, lab)
    sel = cv != "other"
    cn = float(nmi(cv[sel], lab[sel])) if sel.sum() > 50 else float("nan")
    nc = len(np.unique(lab))
    rows[k] = dict(shape_eta2=se, pitch_eta2=pe, dur_eta2=de, chev_nmi=cn, n_clusters=nc)
    print(f"{k:<24}{se:>11.3f}{pe:>11.3f}{de:>10.3f}{cn:>10.3f}{nc:>11d}")

print("\n" + "="*90)
print("PART 2 — NON-CIRCULAR metric: chevron-retrieval kNN purity per REPRESENTATION space")
print("         (does the substrate itself put chevrons next to chevrons?)")
print(f"         base rate (random neighbour same-flag) ~= {chev_rate**2 + (1-chev_rate)**2:.3f};"
      f"  balanced-acc chance = 0.500")
print("="*90)
print(f"{'representation':<22}{'nbr_same_flag':>14}{'chev_balacc':>13}{'chev_retr_prec':>16}")
for name, X in REPS.items():
    same, bal, cr = knn_chevron_purity(X, is_chev)
    rows[f"rep::{name}"] = dict(nbr_same=same, chev_balacc=bal, chev_retr_prec=cr)
    print(f"{name:<22}{same:>14.3f}{bal:>13.3f}{cr:>16.3f}")

print("\n" + "="*90)
print("INTERPRETATION GUIDE")
print(" - If shape_eta2 is high for lab_shape but chev_NMI is ~0 for ALL label sets,")
print("   the win is in the circular metric, NOT in chevron organisation.")
print(" - If chev_balacc ~ 0.5 and chev_retr_prec ~ base rate across ALL representations,")
print("   then NO substrate (incl. SRVF/derivative) makes chevrons neighbours -> the")
print("   bottleneck is the label/representation, not the clustering algorithm.")
print(" - If SRVF/derivative chev_balacc clearly beats registered_shape, the elastic")
print("   substrate was the missed lever.")
print("="*90)

import json
out = "/home/shachar/.claude/jobs/123a8338/tmp/shape_pilot/scorecard.json"
with open(out, "w") as f:
    json.dump({"n": int(N), "chevron_rate": float(chev_rate), "rows": rows}, f, indent=2)
print(f"[OUT] {out}")
