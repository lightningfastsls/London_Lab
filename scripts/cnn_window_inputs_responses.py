"""Show the ACTUAL images fed to each CNN iteration + the response (probability).

For a handful of illustrative sliding windows, reproduce the EXACT grayscale
tensor each model consumed (via the real SlidingInference._prepare_batch) and the
probability it returned. matched_windows + hard_neg share the magma/20-120 image;
the first CNN sees its native inferno/25-110 image. Three probs per window.

Window picks (from wholefile_window_scores_nativefirst.csv):
  A. clear USV — all agree (sanity)
  B. USV the FIRST CNN misses but production catches (recall win, 1st->3rd)
  C. noise matched_windows fires on but production rejects (FP suppression 2nd->3rd)
  D. noise the FIRST CNN (native) fires on — its honest weakness (worst FPR)

The displayed picture is the grayscale 256x100 input AFTER _prepare_batch
(colormap -> flip -> resize -> grayscale -> per-image norm) — literally what the
CNN saw. Probs printed should match the CSV (correctness check).

Authorized legacy override (first CNN only): ExtractionConfig(25_000,110_000,
colormap='inferno'). Defaults / corpus.py untouched (CNN-FREEZE).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/shachar/projects/mickey_london_lab")
sys.path.insert(0, str(REPO / "src"))
from usv_spectrogram.app.core.sliding_inference import SlidingInference  # noqa: E402
from usv_spectrogram.app.core.audio_loader import AudioLoader            # noqa: E402
from usv_spectrogram.detection.extraction_config import ExtractionConfig  # noqa: E402
from usv_spectrogram.corpus import SAMPLE_RATE_HZ                         # noqa: E402

_spec = importlib.util.spec_from_file_location("wfroc", REPO / "scripts" / "cnn_wholefile_roc.py")
wfroc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wfroc)

SR = SAMPLE_RATE_HZ
WIN = 100  # window_width_px (SlidingInference default)
LEG = ExtractionConfig(freq_min_hz=25_000, freq_max_hz=110_000, colormap="inferno")
SCORES = REPO / "presentation/figures/cnn_improvement/_predictions/wholefile_window_scores_nativefirst.csv"
OUT = REPO / "presentation/figures/cnn_improvement/cnn_window_inputs_responses.png"


def pivot_scores():
    df = pd.read_csv(SCORES)
    df["tr"] = df.time_s.round(4)
    wide = df.pivot_table(index=["stem", "tr"], columns="model", values="score").reset_index()
    yt = df.groupby(["stem", "tr"]).y_true.first().reset_index()
    wide = wide.merge(yt, on=["stem", "tr"])
    wide = wide.dropna(subset=["production", "matched_windows", "hard_neg_retrain"])
    return wide


def pick_examples(w):
    usv = w[w.y_true == 1].copy()
    noise = w[w.y_true == 0].copy()
    picks = []
    # A. clear USV — production confident, all see signal
    a = usv.sort_values("hard_neg_retrain", ascending=False).iloc[0]
    picks.append(("A. Clear USV — production confident", a))
    # B. USV first CNN misses but production catches — from a DIFFERENT recording than A
    usv["gap_1_3"] = usv.hard_neg_retrain - usv.production
    b_pool = usv[(usv.hard_neg_retrain >= 0.6) & (usv.stem != a["stem"])]
    b = b_pool.sort_values("gap_1_3", ascending=False).iloc[0]
    picks.append(("B. USV the FIRST CNN misses, production catches", b))
    # C. noise matched fires on, production rejects
    noise["gap_2_3"] = noise.matched_windows - noise.hard_neg_retrain
    c = noise.sort_values("gap_2_3", ascending=False).iloc[0]
    picks.append(("C. Noise matched_windows fires on, production rejects", c))
    # D. noise the first CNN fires on (its honest weakness)
    d = noise.sort_values("production", ascending=False).iloc[0]
    picks.append(("D. Noise the FIRST CNN (native) fires on", d))
    return picks


def window_input(loader, model_si, samples, center_time, native):
    """Return (grayscale_image_256xW, prob) — the exact CNN input + response."""
    if native:
        sd, _, times = loader._compute_spectrogram(samples, SR)
    else:
        sd, _, times = loader._compute_spectrogram(samples, SR)
    norm = model_si._apply_mad_normalization(sd)
    col = int(np.argmin(np.abs(times - center_time)))
    half = WIN // 2
    start = max(0, col - half)
    end = start + WIN
    if end > norm.shape[1]:
        end = norm.shape[1]; start = end - WIN
    window = norm[:, start:end]

    def run():
        t = model_si._prepare_batch([window])           # exact CNN input tensor
        img = t[0, 0].cpu().numpy()                      # (256, W) grayscale in [0,1]
        with torch.no_grad():
            p = float(model_si.model.predict_proba(t.to(model_si.device)).cpu().numpy().flatten()[0])
        return img, p

    if native:
        orig = plt.get_cmap
        plt.get_cmap = lambda *a, **k: orig("inferno")
        try:
            return run()
        finally:
            plt.get_cmap = orig
    return run()


def main():
    print("=" * 72)
    print("CNN window inputs + responses — exact _prepare_batch tensors")
    print(f"[params] window_width_px={WIN}; first-CNN extraction inferno/25000-110000; "
          f"matched+hard_neg magma/20000-120000 (defaults); SR={SR}")

    wide = pivot_scores()
    print(f"[data] {len(wide)} windows with all-3-model scores")
    picks = pick_examples(wide)

    # models + loaders
    si_first = SlidingInference(str(REPO / "models/production/best_model.pt"))
    si_matched = SlidingInference(str(REPO / "models/matched_windows/best_model.pt"))
    si_prod = SlidingInference(str(REPO / "models/hard_neg_retrain/best_model.pt"))
    ld_default = AudioLoader()       # magma / 20-120
    ld_native = AudioLoader(config=LEG)  # inferno / 25-110

    nrows = len(picks)
    fig, axes = plt.subplots(nrows, 2, figsize=(8.4, 3.0 * nrows))
    if nrows == 1:
        axes = axes[None, :]

    samples_cache = {}
    for r, (title, row) in enumerate(picks):
        stem = row["stem"]; t = float(row["tr"]); y = int(row["y_true"])
        wav = wfroc.resolve_wav(stem)
        if stem not in samples_cache:
            samples_cache[stem], _ = librosa.load(str(wav), sr=SR, mono=True)
        samples = samples_cache[stem]

        # magma image (matched + hard_neg) — same image, two models
        img_magma, p_matched = window_input(ld_default, si_matched, samples, t, native=False)
        _, p_prod = window_input(ld_default, si_prod, samples, t, native=False)
        # inferno-native image (first CNN)
        img_inferno, p_first = window_input(ld_native, si_first, samples, t, native=True)

        gt = "USV" if y == 1 else "noise"
        print(f"\n[{title}]")
        print(f"   {stem} @ {t:.3f}s   GT={gt}")
        print(f"   first-CNN(native inferno) p={p_first:.3f}  |  matched p={p_matched:.3f}  |  "
              f"production(hard_neg) p={p_prod:.3f}   (CSV: first={row['production']:.3f} "
              f"matched={row['matched_windows']:.3f} prod={row['hard_neg_retrain']:.3f})")

        axL, axR = axes[r]
        axL.imshow(img_inferno, cmap="gray", aspect="auto", origin="upper")
        axL.set_title(f"FIRST CNN input (inferno/25-110)\np = {p_first:.3f}",
                      fontsize=9, color=("#1a7d1a" if p_first >= 0.6 else "#b00"))
        axR.imshow(img_magma, cmap="gray", aspect="auto", origin="upper")
        axR.set_title(f"matched/hard_neg input (magma/20-120)\n"
                      f"matched p={p_matched:.3f}   production p={p_prod:.3f}",
                      fontsize=9)
        for ax in (axL, axR):
            ax.set_xticks([]); ax.set_yticks([])
        # row label on the far left
        axL.set_ylabel(f"{title}\nGT = {gt}  ({stem[-7:]} @ {t:.2f}s)",
                       fontsize=8.5, rotation=90, labelpad=8,
                       color=("#1a7d1a" if y == 1 else "#666"))

    fig.suptitle("What each CNN iteration was fed, and how it responded\n"
                 "(grayscale 256x100 = the literal CNN input; green prob = fired >=0.6)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] wrote {OUT}")


if __name__ == "__main__":
    main()
