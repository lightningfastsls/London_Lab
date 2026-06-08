"""Sanity: does the AUC change at the app's energy_threshold=0.35 (vs the 0.1
default used in the whole-file ROC)? Confirms the FP-filter/detection bug did NOT
touch the AUC (ROC uses raw per-window probs) and quantifies the energy-gate effect.

first CNN: per-window MAD + inferno/25-110, 150px (its faithful pipeline).
matched/hard_neg: magma/20-120, 100px, global MAD (their production inference).
"""
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd, librosa, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

REPO = Path("/home/shachar/projects/mickey_london_lab"); sys.path.insert(0, str(REPO/"src"))
from usv_spectrogram.app.core.sliding_inference import SlidingInference
from usv_spectrogram.app.core.audio_loader import AudioLoader
from usv_spectrogram.detection.extraction_config import ExtractionConfig
from usv_spectrogram.corpus import SAMPLE_RATE_HZ
_spec = importlib.util.spec_from_file_location("wfroc", REPO/"scripts"/"cnn_wholefile_roc.py")
wfroc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(wfroc)

SR = SAMPLE_RATE_HZ; HOP = 10
LEG = ExtractionConfig(freq_min_hz=25_000, freq_max_hz=110_000, colormap="inferno")
V1, V2 = LEG.mad_vmin_scale, LEG.mad_vmax_scale
TOL = wfroc.TOL_S

def pw_mad(c):
    m=np.median(c); d=np.median(np.abs(c-m)); lo,hi=m-V1*d,m+V2*d
    return (np.clip(c,lo,hi)-lo)/(hi-lo+1e-12)

def first_scores(si, sd, t, width, energy):
    n=sd.shape[1]; h=width//2; wt,wp=[],[]
    o=plt.get_cmap; plt.get_cmap=lambda *a,**k:o("inferno")
    try:
        for c in range(h, n-h, HOP):
            w=pw_mad(sd[:, c-h:c-h+width]); wt.append(float(t[c]))
            if float(w.max())<energy: wp.append(0.0); continue
            tt=si._prepare_batch([w])
            with torch.no_grad():
                wp.append(float(si.model.predict_proba(tt.to(si.device)).cpu().numpy().flatten()[0]))
    finally:
        plt.get_cmap=o
    return np.array(wt), np.array(wp)

usv,noise,alls = wfroc.load_ground_truth()
wavs={s:wfroc.resolve_wav(s) for s in alls}
def label(stem,tt):
    if stem in noise: return 0
    iv=usv.get(stem,[]); return 1 if any(s-TOL<=tt<=e+TOL for s,e in iv) else None

native=AudioLoader(config=LEG); default=AudioLoader()
print("AUC vs energy_threshold (33 files). first=per-win MAD/inferno/150px; matched/hard_neg=global/magma/100px")
print(f"{'energy':>7s}  {'first CNN':>10s} {'matched':>9s} {'hard_neg':>9s}")
for energy in (0.1, 0.35):
    rows={"first":[], "matched":[], "hard_neg":[]}
    si_f=SlidingInference(str(REPO/"models/production/best_model.pt"),window_width_px=150,hop_px=HOP,energy_threshold=energy,enable_per_window_norm=False)
    si_m=SlidingInference(str(REPO/"models/matched_windows/best_model.pt"),window_width_px=100,hop_px=HOP,energy_threshold=energy,enable_per_window_norm=False)
    si_h=SlidingInference(str(REPO/"models/hard_neg_retrain/best_model.pt"),window_width_px=100,hop_px=HOP,energy_threshold=energy,enable_per_window_norm=False)
    for stem in sorted(alls):
        wav=wavs[stem]
        if not wav: continue
        samp,_=librosa.load(str(wav),sr=SR,mono=True)
        au=default.load(str(wav)); sd_n,_,t_n=native._compute_spectrogram(samp,SR)
        ft,fp=first_scores(si_f,sd_n,t_n,150,energy)
        for tt,pp in zip(ft,fp):
            y=label(stem,tt)
            if y is not None: rows["first"].append((y,pp))
        for tag,si in (("matched",si_m),("hard_neg",si_h)):
            r=si.infer(au.spectrogram_db,au.times)
            for tt,pp in zip(r.times,r.probabilities):
                y=label(stem,tt)
                if y is not None: rows[tag].append((y,pp))
    aucs={}
    for k,v in rows.items():
        a=np.array(v); aucs[k]=roc_auc_score(a[:,0],a[:,1])
    print(f"{energy:>7.2f}  {aucs['first']:>10.4f} {aucs['matched']:>9.4f} {aucs['hard_neg']:>9.4f}")
