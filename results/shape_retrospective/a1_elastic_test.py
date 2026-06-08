"""
A1 lead-lever test: does per-pair ELASTIC time-warp alignment (DTW / soft-DTW) beat the
Euclidean distance registration uses, on the SAME registered ridges, scored against the
204 human labels? This is the one mechanism registration provably lacks (internal-landmark
warp). Small N (182 labeled) -> exact pairwise distance matrices, fast, OOM-safe.
"""
import numpy as np, pandas as pd
from tslearn.metrics import cdist_dtw, cdist_soft_dtw_normalized
from scipy.spatial.distance import cdist

TMP="/home/shachar/.claude/jobs/123a8338/tmp/shape_pilot"
m=np.load(f"{TMP}/true_registered_ridges_meta.npz",allow_pickle=True)
Sh=m["shapes"].astype(np.float64); ws=m["wav_stem"].astype(str); cid=m["call_id"]
h=pd.read_csv("/home/shachar/projects/mickey_london_lab/data/manual_shape_labels.csv")
hset=set(h["call_id"])
comp=np.array([f"{ws[i]}__det{cid[i]-1}" for i in range(len(ws))])  # offset -1
id2row={}
for i,c in enumerate(comp):
    if c in hset and c not in id2row: id2row[c]=i
j=h[h["call_id"].isin(id2row)].copy(); j["row"]=j["call_id"].map(id2row)
keep=~j["shape_label"].isin(["unclear"]); j=j[keep]
rows=j["row"].to_numpy(); y=j["shape_label"].to_numpy()
fam={}
for lbl in set(y):
    fam[lbl]=("chevron" if lbl in("Chevron","Reverse Chevron") else
              "jump" if lbl in("Step up","Step down","Two steps","Multi-steps") else
              "flat" if lbl=="Flat" else "complex" if lbl=="Complex" else lbl)
yf=np.array([fam[v] for v in y])
X=Sh[rows]; n=len(X)
print("="*84)
print(f"A1 ELASTIC TEST | n={n} human-labeled registered ridges | seed=42 | k=10")
print("  Euclidean = registration's metric;  DTW/softDTW = per-pair warp alignment (A1)")
print("="*84)
print("[FAMILY counts]", dict(pd.Series(yf).value_counts()))

K=10
def knn_purity_from_D(D, labels, k=K):
    np.fill_diagonal(D, np.inf)
    nbr=np.argsort(D,axis=1)[:,:k]
    out={}
    for f in ["chevron","jump","flat","complex"]:
        tm=labels==f
        out[f]=float((labels[nbr[tm]]==f).mean()) if tm.sum() else float("nan")
    return out

print("\n[INFO] computing pairwise distance matrices (Euclidean, DTW, soft-DTW)...")
D_euc=cdist(X,X,"euclidean")
Xt=X[:,:,None]  # tslearn wants (n, sz, d)
D_dtw=cdist_dtw(Xt)
D_sdtw=cdist_soft_dtw_normalized(Xt, gamma=1.0)

base={f:(yf==f).mean() for f in ["chevron","jump","flat","complex"]}
print("\n{:<16}{:>10}{:>10}{:>10}{:>10}".format("metric","chevron","jump","flat","complex"))
print("{:<16}{:>10.3f}{:>10.3f}{:>10.3f}{:>10.3f}".format("BASE RATE",*[base[f] for f in["chevron","jump","flat","complex"]]))
for name,D in [("euclidean(reg)",D_euc),("DTW(elastic)",D_dtw),("soft-DTW",D_sdtw)]:
    p=knn_purity_from_D(D.copy(),yf)
    print("{:<16}{:>10.3f}{:>10.3f}{:>10.3f}{:>10.3f}".format(name,p["chevron"],p["jump"],p["flat"],p["complex"]))
print("\n[READ] if DTW/soft-DTW > euclidean on chevron/jump, internal-warp alignment (A1) has merit.")
print("[CAVEAT] n=%d, chevron family ~25, all lab cohort 131204. Directional, not definitive."%n)
