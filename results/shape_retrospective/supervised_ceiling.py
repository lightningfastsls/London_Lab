"""
Supervised-ceiling probe: how much chevron signal is linearly/locally accessible on the
PROVEN substrate (registered shape, SRVF) that unsupervised KMeans-20 throws away?

Tests the leading 'what we could have done better' lever: a small labelled set + a
supervised/metric step on the registered ridge — instead of blind unsupervised KMeans —
would have 'actually achieved' chevron-vs-flat separation.

CAVEAT printed in output: chevron_valley is a heuristic derived FROM the registered shape,
so absolute numbers are inflated by construction. The clean signal is RELATIVE:
  (a) registered/SRVF vs the learned encoder's 0.517 (Phase 0a),
  (b) supervised ceiling vs unsupervised KMeans-20 NMI 0.17,
  (c) the few-label learning curve (how few labels suffice).
All read-only CPU.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

NPZ = "/home/shachar/.claude/jobs/123a8338/tmp/shape_pilot/true_registered_ridges.npz"
SEED = 42
rng = np.random.default_rng(SEED)

d = np.load(NPZ, allow_pickle=True)
Sh = d["shapes"].astype(np.float64)
cv = d["chevron_valley"].astype(str)
is_chev = (cv == "chevron").astype(int)

def srvf(x):
    g = np.gradient(x, axis=1)
    return np.sign(g) * np.sqrt(np.abs(g))

REPS = {"registered_shape": Sh, "srvf": srvf(Sh), "derivative": np.diff(Sh, axis=1)}

print("="*88)
print("PARAMETERS")
print(f"  npz={NPZ}  seed={SEED}")
print(f"  target=chevron (n={int(is_chev.sum())}, base rate={is_chev.mean():.3f})")
print("  CAVEAT: chevron_valley heuristic is derived FROM registered shape -> abs. numbers")
print("          inflated. Read RELATIVE vs learned-encoder 0.517 and vs KMeans NMI 0.17.")
print("="*88)

cvkf = StratifiedKFold(5, shuffle=True, random_state=SEED)

print("\n-- Supervised ceiling (5-fold balanced accuracy, FULL data) --")
print(f"{'representation':<20}{'logreg':>10}{'rf':>10}")
for name, X in REPS.items():
    lr = cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced"),
                         X, is_chev, cv=cvkf, scoring="balanced_accuracy").mean()
    rf = cross_val_score(RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=SEED,
                         class_weight="balanced"),
                         X, is_chev, cv=cvkf, scoring="balanced_accuracy").mean()
    print(f"{name:<20}{lr:>10.3f}{rf:>10.3f}")

print("\n-- Few-label learning curve on registered_shape (how few labels suffice?) --")
print("   train on N random labels, RF, test on a held-out 10k; balanced accuracy")
X = REPS["registered_shape"]
n = len(X)
test_idx = rng.choice(n, size=10000, replace=False)
test_mask = np.zeros(n, bool); test_mask[test_idx] = True
pool = np.where(~test_mask)[0]
print(f"{'n_labels':>10}{'bal_acc':>10}")
for nlab in [50, 100, 200, 500, 1000, 5000]:
    tr = rng.choice(pool, size=nlab, replace=False)
    clf = RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=SEED,
                                 class_weight="balanced").fit(X[tr], is_chev[tr])
    pred = clf.predict(X[test_idx])
    ba = balanced_accuracy_score(is_chev[test_idx], pred)
    print(f"{nlab:>10}{ba:>10.3f}")

print("\n[REFERENCE] learned Pathway-B encoder (Phase 0a) chevron balanced acc = 0.517 (~random 0.501)")
print("[REFERENCE] unsupervised KMeans-20 on registered shape: chevron NMI = 0.170")
print("="*88)
