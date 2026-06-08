"""Expanded whole-file noise test: matched_windows vs production (hard_neg) on a
LOT more data, RAW sliding scores (no FP-filter — the FP-filter masks the model
difference; the user remembers the raw model being noisy).

Data:
  - known-noise whole files  (manual_review all_noise  U  KNOWN_NOISE_SUFFIXES)
        -> airtight negatives: every flagged window is a false positive
  - USV-bearing files (manual_review usv events)  -> positives for the ROC
  - large RANDOM sample of 5970 files  -> aggregate noise-proneness at scale
        (most windows in random recordings are non-USV, so a higher flagged
         fraction = more noise found)

Outputs (presentation/figures/cnn_improvement/):
  cnn_matched_vs_prod_scaled.png
  _predictions/scaled_window_scores.csv   (file, time, category, y, p_matched, p_prod)
Progress -> $CLAUDE_JOB_DIR/scaled_run.log
"""
import sys, glob, json, os, random, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

REPO = Path("/home/shachar/projects/mickey_london_lab")
sys.path.insert(0, str(REPO / "src"))
from usv_spectrogram.app.core.sliding_inference import SlidingInference
from usv_spectrogram.app.core.audio_loader import AudioLoader

LOG = open(os.environ.get("CLAUDE_JOB_DIR", "/tmp") + "/scaled_run.log", "w", buffering=1)
def log(*a): print(*a, file=LOG); print(*a)

N_RANDOM = 200
TOL_S = 0.02
THR = 0.5
KNOWN_NOISE_SUFFIXES = {"0001960","0002431","0002522","0003502","0003503","0003781",
    "0003794","0005107","0005656","0006086","0000570","0000716","0000717","0003579",
    "0003825","0004706","0005108","0005647"}
MODELS = {"matched": "models/matched_windows/best_model.pt",
          "prod":    "models/hard_neg_retrain/best_model.pt"}


def resolve(stem, suffix=False):
    g = (glob.glob(str(REPO/f"5970 USV/*{stem}.wav")) if suffix else
         glob.glob(str(REPO/f"5970 USV/{stem}.wav")) + glob.glob(str(REPO/f"5970_manual_review_reviewed/{stem}.wav")))
    if not g:
        g = glob.glob(str(REPO/f"5970/**/*{stem}.wav"), recursive=True) if suffix else \
            glob.glob(str(REPO/f"5970/**/{stem}.wav"), recursive=True)
    return g[0] if g else None


def build_fileset():
    spec = importlib.util.spec_from_file_location("blab", REPO/"scripts/build_manual_review_labels.py")
    blab = importlib.util.module_from_spec(spec); spec.loader.exec_module(blab)
    rev = pd.read_csv(REPO/"results/batch_5970/manual_review_all_detections.csv"); rev["suffix"]=rev["stem"].str[-7:]
    noise, usv_iv = {}, {}
    for suf,(rule,data) in blab.ANNOTATIONS.items():
        if rule=="skip": continue
        sub = rev[rev["suffix"]==suf].sort_values("detection_idx")
        if sub.empty: continue
        stem = sub["stem"].iloc[0]; p = resolve(stem)
        if not p: continue
        labels = blab.apply_labels(len(sub), rule, data)
        if rule=="all_noise":
            noise[p]=stem
        ivs = [(r["start_time_s"],r["end_time_s"]) for i,(_,r) in enumerate(sub.iterrows()) if labels.get(i)=="usv"]
        if ivs: usv_iv[p]=ivs
    for suf in KNOWN_NOISE_SUFFIXES:
        p = resolve(suf, suffix=True)
        if p: noise.setdefault(p, suf)
    # random sample, excluding labeled files
    allwav = sorted(glob.glob(str(REPO/"5970/**/*.wav"), recursive=True))
    used = set(noise) | set(usv_iv)
    pool = [p for p in allwav if p not in used]
    random.Random(0).shuffle(pool)
    rand = pool[:N_RANDOM]
    return noise, usv_iv, rand


def main():
    noise, usv_iv, rand = build_fileset()
    log(f"[fileset] known-noise={len(noise)}  usv-bearing={len(usv_iv)}  random={len(rand)}  "
        f"TOTAL={len(set(noise)|set(usv_iv)|set(rand))}")
    si = {k: SlidingInference(str(REPO/v)) for k,v in MODELS.items()}
    al = AudioLoader()

    all_files = list(dict.fromkeys(list(noise)+list(usv_iv)+list(rand)))
    rows=[]
    for i,p in enumerate(all_files):
        cat = "noise" if p in noise else ("usv" if p in usv_iv else "random")
        try:
            a = al.load(p)
        except Exception as e:
            log(f"  skip {Path(p).name}: {e}"); continue
        sc={}
        for k in MODELS:
            sc[k]=si[k].infer(a.spectrogram_db, a.times)
        times = sc["matched"].times
        ivs = usv_iv.get(p, [])
        for j,t in enumerate(times):
            if cat=="noise": y=0
            elif cat=="usv": y=1 if any(s-TOL_S<=t<=e+TOL_S for s,e in ivs) else -1  # -1 = skip(ambig bg)
            else: y=-9  # random/unlabeled
            rows.append((Path(p).name, cat, float(t), y,
                         float(sc["matched"].probabilities[j]), float(sc["prod"].probabilities[j])))
        if (i+1)%20==0: log(f"  [{i+1}/{len(all_files)}] processed")
    df = pd.DataFrame(rows, columns=["file","category","time_s","y","p_matched","p_prod"])
    out_csv = REPO/"presentation/figures/cnn_improvement/_predictions/scaled_window_scores.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True); df.to_csv(out_csv, index=False)
    log(f"[scores] {len(df)} window-rows -> {out_csv}")

    # ---- ROC on labeled subset (pos=usv events, neg=known-noise windows) ----
    lab = df[df.y.isin([0,1])]
    res={}
    for k in ["matched","prod"]:
        y=lab.y.values; s=lab[f"p_{k}"].values
        auc=roc_auc_score(y,s); pra=average_precision_score(y,s)
        fpr,tpr,_=roc_curve(y,s); idx=np.argmax(tpr>=0.90)
        res[k]=dict(auc=auc,pra=pra,fpr=fpr,tpr=tpr,fpr90=fpr[idx])
        log(f"[ROC {k}] AUC={auc:.4f} PR-AUC={pra:.4f} FPR@90recall={fpr[idx]:.3f} "
            f"pos={int((y==1).sum())} neg={int((y==0).sum())}")
    # ---- noise-proneness at scale: frac windows > THR ----
    log("\n[scale] fraction of windows flagged > %.2f:" % THR)
    for cat in ["noise","random","usv"]:
        d=df[df.category==cat]
        if len(d)==0: continue
        log(f"   {cat:7s} (n_win={len(d):6d}, n_files={d.file.nunique():3d}): "
            f"matched={ (d.p_matched>THR).mean()*100:5.2f}%   prod={ (d.p_prod>THR).mean()*100:5.2f}%")
    # per-file FP detections on known-noise (windows>THR grouped loosely as count>THR)
    nd=df[df.category=="noise"]
    fp_m=int((nd.p_matched>THR).sum()); fp_p=int((nd.p_prod>THR).sum())
    log(f"[known-noise FP windows] matched={fp_m}  prod={fp_p}  ratio={fp_m/max(fp_p,1):.1f}x")

    # ---- figure ----
    fig,(axL,axR)=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
    for k,lab_,c in [("matched","matched_windows (old)","#5c8fb3"),("prod","production (hard-neg)","#26c6da")]:
        axL.plot(res[k]["fpr"],res[k]["tpr"],color=c,lw=2.4,label=f"{lab_}  (AUC {res[k]['auc']:.3f})")
    axL.plot([0,1],[0,1],ls=":",color="0.6",lw=1.0,label="chance")
    axL.set_xlabel("False-positive rate (noise flagged)"); axL.set_ylabel("True-positive rate (USV found)")
    axL.set_title(f"A.  Whole-file ROC ({lab.file.nunique()} labeled files, {int((lab.y==0).sum())} noise windows)",
                  fontsize=12,fontweight="bold",loc="left")
    axL.legend(loc="lower right",fontsize=9); axL.grid(alpha=0.25)
    cats=["noise","random"]; xm=[ (df[df.category==c].p_matched>THR).mean()*100 for c in cats]
    xp=[ (df[df.category==c].p_prod>THR).mean()*100 for c in cats]
    x=np.arange(len(cats)); w=0.38
    axR.bar(x-w/2,xm,w,label="matched_windows (old)",color="#5c8fb3")
    axR.bar(x+w/2,xp,w,label="production (hard-neg)",color="#26c6da")
    for xi,(a_,b_) in enumerate(zip(xm,xp)):
        axR.annotate(f"{a_:.1f}%",(xi-w/2,a_),ha="center",va="bottom",fontsize=9,fontweight="bold")
        axR.annotate(f"{b_:.1f}%",(xi+w/2,b_),ha="center",va="bottom",fontsize=9,fontweight="bold")
    axR.set_xticks(x); axR.set_xticklabels([f"known-noise\nfiles" if c=="noise" else f"random {len(rand)}\nfiles" for c in cats],fontsize=9)
    axR.set_ylabel(f"% of windows flagged as USV (>{THR})")
    axR.set_title("B.  Noise flagged at scale (raw model, no FP-filter)",fontsize=12,fontweight="bold",loc="left")
    axR.legend(fontsize=9); axR.grid(axis="y",alpha=0.25)
    fig.suptitle("matched_windows vs production on a lot more data (raw sliding scores)",fontsize=15,fontweight="bold")
    out=REPO/"presentation/figures/cnn_improvement/cnn_matched_vs_prod_scaled.png"
    fig.savefig(out,dpi=150,bbox_inches="tight"); plt.close(fig)
    log(f"wrote {out}")


if __name__=="__main__":
    main()
