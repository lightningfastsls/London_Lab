"""
DECISIVE de-circularized test: score the 1-D shape representations against the REAL
204-row human shape labels (data/manual_shape_labels.csv) — the test nobody ran while
the bake-off was graded against the circular chevron_valley heuristic.

Outputs, for {registered_shape, srvf, derivative}:
  - leave-one-out kNN purity per human class (chevron, jump-family, flat, complex)
  - NMI of registration's KMeans-20 (lab_shape) vs human labels
  - how good was the chevron_valley HEURISTIC vs humans? (confusion / precision-recall)
Comparison reference: learned encoder vs human (data/alpha3_a6/a6_gamma_binding.json).
All read-only CPU. Prints all params, join diagnostics, and N per class.
"""
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import normalized_mutual_info_score as nmi

TMP = "/home/shachar/.claude/jobs/123a8338/tmp/shape_pilot"
META = f"{TMP}/true_registered_ridges_meta.npz"
LAB  = f"{TMP}/true_registered_ridges.npz"        # has lab_shape KMeans labels + chevron_valley
HUM  = "/home/shachar/projects/mickey_london_lab/data/manual_shape_labels.csv"
SEED = 42; KNN = 10

print("="*92)
print("PARAMETERS")
print(f"  meta={META}\n  labels_npz={LAB}\n  human={HUM}\n  seed={SEED}  kNN_k={KNN}")
print("  metric = HUMAN-anchored kNN purity + NMI (NON-circular). chevron_valley used ONLY")
print("           to audit the heuristic itself, never as the target.")
print("="*92)

m = np.load(META, allow_pickle=True)
L = np.load(LAB, allow_pickle=True)
Sh = m["shapes"].astype(np.float64)
ws = m["wav_stem"].astype(str); cid = m["call_id"]
coh = m["cohort"].astype(str)
lab_shape = L["lab_shape"]; chev_val = L["chevron_valley"].astype(str)

# meta and lab npz must be row-aligned (same 'shapes'); verify
assert L["shapes"].shape == Sh.shape, "row mismatch"
print(f"[CHK] meta/lab row-aligned shapes match: {np.allclose(L['shapes'].astype(np.float64), Sh, atol=1e-4)}")

h = pd.read_csv(HUM)
hset = set(h["call_id"])

# resolve det offset: test {call_id} vs {call_id-1}
for off, tag in [(0, "__det{cid}"), (-1, "__det{cid-1}")]:
    comp = np.array([f"{ws[i]}__det{cid[i]+off}" for i in range(len(ws))])
    n_uniq_hit = len(set(comp) & hset)
    print(f"[JOIN] offset {off:>2} ({tag}): unique human ids hit = {n_uniq_hit}/{len(h)}")

# offset -1 wins (200/204 unique hits): human det is 0-indexed, meta call_id 1-indexed
OFF = -1
comp = np.array([f"{ws[i]}__det{cid[i]+OFF}" for i in range(len(ws))])
id2row = {}
for i, c in enumerate(comp):
    if c in hset and c not in id2row:
        id2row[c] = i
joined = h[h["call_id"].isin(id2row)].copy()
joined["row"] = joined["call_id"].map(id2row)
print(f"[JOIN] matched {len(joined)}/{len(h)} human labels to a unique registered ridge")
rows = joined["row"].to_numpy()
y_raw = joined["shape_label"].to_numpy()

# drop unclear/Short(n=3) for the class metrics; keep Noise as its own class
keep = ~np.isin(y_raw, ["unclear"])
rows, y = rows[keep], y_raw[keep]
print(f"[DATA] {len(y)} labels after dropping 'unclear'. class counts:")
for c, n in sorted(pd.Series(y).value_counts().items(), key=lambda x:-x[1]):
    print(f"        {c:<16} {n}")

# representations restricted to the labeled rows
def srvf(x):
    g = np.gradient(x, axis=1); return np.sign(g)*np.sqrt(np.abs(g))
REPS = {"registered_shape": Sh[rows], "srvf": srvf(Sh)[rows], "derivative": np.diff(Sh, axis=1)[rows]}

# group families (chevron = Chevron+Reverse Chevron; jump = Step up/down+Two steps+Multi-steps)
fam = {}
for lbl in y:
    if lbl in ("Chevron","Reverse Chevron"): fam[lbl]="chevron"
    elif lbl in ("Step up","Step down","Two steps","Multi-steps"): fam[lbl]="jump"
    elif lbl=="Flat": fam[lbl]="flat"
    elif lbl=="Complex": fam[lbl]="complex"
    else: fam[lbl]=lbl
yf = np.array([fam[v] for v in y])
print("\n[FAMILY] grouped counts:", dict(pd.Series(yf).value_counts()))

def loo_knn_purity(X, labels, target, k=KNN):
    """frac of each target-point's k nearest neighbours sharing the target family."""
    n=len(X); k=min(k, n-1)
    nn=NearestNeighbors(n_neighbors=k+1).fit(X)
    _,nbr=nn.kneighbors(X); nbr=nbr[:,1:]
    tmask = labels==target
    if tmask.sum()==0: return float("nan"), 0
    same = (labels[nbr[tmask]]==target).mean()
    return float(same), int(tmask.sum())

print("\n" + "="*92)
print("HUMAN-ANCHORED kNN purity (frac of a class-member's neighbours in the SAME class)")
print("   reference baselines from data/alpha3_a6 (2-D substrate, 185-anchor):")
print("     LEARNED encoder: chevron 0.124 jump 0.335 flat 0.291 complex 0.075 | NMI 0.287 probe 0.375")
print("     IDENTITY(2-D)  : chevron 0.068 jump 0.462 flat 0.300 complex 0.042 | NMI 0.245")
print("     RANDOM         : chevron 0.084 jump 0.311 flat 0.173 complex 0.092 | NMI 0.173")
print("="*92)
fams=["chevron","jump","flat","complex"]
print(f"{'representation':<18}" + "".join(f"{f:>11}" for f in fams) + f"{'NMI(reg KM)':>13}")
res={}
for name,X in REPS.items():
    pur={f: loo_knn_purity(X,yf,f)[0] for f in fams}
    # NMI: registration KMeans-20 labels (lab_shape) on these same rows vs human family
    km = lab_shape[rows]
    nm = float(nmi(yf, km))
    res[name]={**pur,"nmi_regKM":nm}
    print(f"{name:<18}" + "".join(f"{pur[f]:>11.3f}" for f in fams) + f"{nm:>13.3f}")

# base rates for context
print("\n[BASE RATES] " + "  ".join(f"{f}={ (yf==f).mean():.3f}" for f in fams))

print("\n" + "="*92)
print("HOW GOOD WAS THE chevron_valley HEURISTIC (the thing everything was graded against)?")
print("="*92)
cv = chev_val[rows]
human_chev = np.isin(y, ["Chevron","Reverse Chevron"])
heur_chev = (cv=="chevron")
tp=int((heur_chev&human_chev).sum()); fp=int((heur_chev&~human_chev).sum())
fn=int((~heur_chev&human_chev).sum())
prec = tp/(tp+fp) if tp+fp else float("nan")
rec  = tp/(tp+fn) if tp+fn else float("nan")
print(f"  heuristic 'chevron' vs human chevron-family: precision={prec:.3f} recall={rec:.3f} (tp={tp} fp={fp} fn={fn})")
print(f"  human chevron-family n={int(human_chev.sum())}; heuristic flags n={int(heur_chev.sum())}")
print("  -> if precision/recall are low, the metric the WHOLE bake-off optimized was a poor proxy.")

import json
json.dump({k:{kk:(None if (isinstance(vv,float) and vv!=vv) else vv) for kk,vv in v.items()} for k,v in res.items()},
          open(f"/home/shachar/projects/mickey_london_lab/results/shape_retrospective/human_anchored_scorecard.json","w"), indent=2)
print("\n[OUT] results/shape_retrospective/human_anchored_scorecard.json")
print("[CAVEAT] N is tiny per class (esp. chevron); treat as a validation anchor, not training. "
      "All 204 are lab cohort 131204 — wild cohorts unlabeled.")
