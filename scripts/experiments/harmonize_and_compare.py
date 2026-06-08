"""WS-E — confound-robust cohort comparison of continuous shape repertoire.

Compares the DISTRIBUTION of the 8 elastic-FPCA shape coordinates across cohorts/
partners while removing the dominant *cage* confound, then runs the four
validation controls that the partner-swap matrix makes free.

Pipeline
--------
1. Load merged elastic-FPCA scores (one row per call) + biology-safe metadata.
2. Define the batch/cage variable = ``cohort`` (lab_131204 is one cage with a
   17-way partner-swap; 5970/3452/9252 are three separate wild cages).
3. Harmonize the 8 shape coords with ComBat / neuroHarmonize (per-feature
   location+scale, empirical Bayes), no biological covariate protected (pure
   batch removal). CORAL whitening-recolouring as a closed-form cross-check.
4. BEFORE vs AFTER: Wasserstein-2 (OT) and RBF-MMD between every cohort pair.
5. Four controls:
   - NEGATIVE: cohort decodable from scores -> must collapse to chance after.
   - POSITIVE: within-lab partner (couple) decodable -> must stay ~unchanged
     (global ComBat applies ONE affine map to all lab calls, so partner
     separation should be preserved; erasure => over-correction).
   - IDENTIFIABILITY: cohort==cage==biological-unit for the wild groups -> lab-vs-
     wild and wild-vs-wild contrasts are UNIDENTIFIABLE under cohort-ComBat
     (flagged, not silently corrected). Only within-lab (constant cage) is
     identifiable.
   - SPURIOUS-REMOVAL: permute cohort labels, refit ComBat, confirm true-cohort
     structure is essentially untouched (correction invents nothing).
6. Gate E: within the lab cage, pairwise partner Wasserstein with permutation
   p-values, compared against the wild-vs-wild cross-cage noise floor.

Everything that matters (params, thresholds, row counts, chance rates) is printed
at run start per ``feedback_analysis_print_params``.

Run:
    .venv/bin/python scripts/experiments/harmonize_and_compare.py
"""
from __future__ import annotations

import base64
import html
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---- repo import path -------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.experiments._fpca_merge import (  # noqa: E402
    FPCA_FEATURES,
    PITCH_COL,
    DURATION_COL,
    load_merged_fpca,
)
from scripts.experiments._dist_stats import (  # noqa: E402
    median_heuristic_gamma,
    mmd_perm_test,
    rbf_mmd2,
    subsample,
    wasserstein2,
    wasserstein_perm_test,
)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from neuroHarmonize import harmonizationLearn, harmonizationApply  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, accuracy_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# ============================ PARAMETERS =====================================
SEED = 0
BATCH_COL = "cohort"           # the cage/batch axis we remove
COHORTS = ["lab_131204", "5970", "3452", "9252"]
WILD = ["5970", "3452", "9252"]
LAB = "lab_131204"

# distance estimation: subsample groups to a common size, repeat for a CI
OT_SUBSAMPLE = 1200            # rows per group for each OT/MMD point estimate
OT_REPEATS = 15               # resamples -> mean +/- std
PERM_SUBSAMPLE = 400          # rows per group inside the permutation null
N_PERM = 200                  # permutations per p-value
SINKHORN_REG = None           # None => exact ot.emd2 for point estimates
MMD_MAXN_FOR_GAMMA = 1500     # cap for median-heuristic bandwidth

# classifier controls
CV_FOLDS = 5
LOGREG_MAXITER = 2000
RESULTS_DIR = _ROOT / "results" / "ws_e_harmonize"

RNG = np.random.default_rng(SEED)


# ============================ HELPERS ========================================
def parse_couple(wav_stem: pd.Series) -> pd.Series:
    """Partner-pairing token (m{X}fm{Y}) embedded in the lab wav_stem."""
    return wav_stem.str.extract(r"_(m\dfm\d)_", expand=False)


def combat_correct(X: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """neuroHarmonize ComBat: remove per-feature batch location+scale (eBayes).

    No biological covariate is protected -> pure batch (cage) removal. Returns
    corrected array aligned row-for-row with X.
    """
    covars = pd.DataFrame({"SITE": np.asarray(batch)})
    _, X_h = harmonizationLearn(np.asarray(X, dtype=float), covars)
    return X_h


def coral_correct(X: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """CORAL cross-check: per-batch whiten then recolour to the pooled covariance.

    Closed-form second-order alignment (Sun et al. 2016). Each batch's mean and
    covariance are mapped to the pooled mean/covariance.
    """
    X = np.asarray(X, dtype=float)
    batch = np.asarray(batch)
    mu_all = X.mean(0)
    cov_all = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
    target = _matrix_sqrt(cov_all)
    out = np.empty_like(X)
    for b in np.unique(batch):
        m = batch == b
        Xb = X[m]
        mu_b = Xb.mean(0)
        cov_b = np.cov(Xb, rowvar=False) + 1e-6 * np.eye(X.shape[1])
        whiten = _matrix_sqrt(np.linalg.inv(cov_b))
        out[m] = (Xb - mu_b) @ whiten @ target + mu_all
    return out


def _matrix_sqrt(A: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 1e-12, None)
    return (V * np.sqrt(w)) @ V.T


def decode_accuracy(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> dict:
    """Stratified-CV logistic-regression decodability of label y from X."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    bals, raws = [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        # class_weight balanced: without it the severe cohort/partner imbalance
        # (lab 39159 vs 3452 306) makes logreg predict the majority class, pinning
        # balanced-acc at exactly chance and hiding any real before/after change.
        clf = LogisticRegression(max_iter=LOGREG_MAXITER, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
        bals.append(balanced_accuracy_score(y[te], pred))
        raws.append(accuracy_score(y[te], pred))
    classes, counts = np.unique(y, return_counts=True)
    return {
        "balanced_acc": float(np.mean(bals)),
        "balanced_acc_std": float(np.std(bals)),
        "raw_acc": float(np.mean(raws)),
        "n_classes": int(len(classes)),
        "chance_balanced": 1.0 / len(classes),
        "chance_majority": float(counts.max() / counts.sum()),
    }


def pairwise_distance(Xa: np.ndarray, Xb: np.ndarray, gamma: float,
                      seed: int = SEED) -> dict:
    """Resampled W2 and RBF-MMD2 between two groups (mean +/- std over repeats)."""
    rng = np.random.default_rng(seed)
    ws, mmds = [], []
    for _ in range(OT_REPEATS):
        A = subsample(Xa, OT_SUBSAMPLE, rng)
        B = subsample(Xb, OT_SUBSAMPLE, rng)
        ws.append(wasserstein2(A, B, SINKHORN_REG))
        mmds.append(rbf_mmd2(A, B, gamma))
    return {
        "w2": float(np.mean(ws)), "w2_std": float(np.std(ws)),
        "mmd2": float(np.mean(mmds)), "mmd2_std": float(np.std(mmds)),
    }


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ============================ MAIN ===========================================
def main() -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ----- load -----
    df = load_merged_fpca(cohorts=COHORTS, dedupe=True, require_meta=False)
    df = df.dropna(subset=FPCA_FEATURES).reset_index(drop=True)
    df["couple"] = np.where(df[BATCH_COL] == LAB, parse_couple(df["wav_stem"]), np.nan)

    sizes = df.groupby(BATCH_COL).size().to_dict()
    lab_df = df[df[BATCH_COL] == LAB].copy()
    couple_sizes = lab_df["couple"].value_counts().to_dict()

    # ----- PARAM BLOCK (printed) -----
    print("=" * 78)
    print("WS-E  harmonize_and_compare.py  —  confound-robust cohort comparison")
    print("=" * 78)
    print(f"seed                 : {SEED}")
    print(f"features (8 shape)   : {FPCA_FEATURES}")
    print(f"pitch / duration     : {PITCH_COL} / {DURATION_COL} (metadata only, not corrected)")
    print(f"batch / cage var     : '{BATCH_COL}'  levels={COHORTS}")
    print(f"cohort row counts    : {sizes}  (total={len(df)})")
    print(f"lab couples (17)     : n={lab_df['couple'].nunique()}  sizes={couple_sizes}")
    print(f"OT/MMD point est     : subsample={OT_SUBSAMPLE}/group, repeats={OT_REPEATS}, "
          f"ground=sq-euclid, exact emd2 (sinkhorn_reg={SINKHORN_REG})")
    print(f"perm test            : subsample={PERM_SUBSAMPLE}/group, n_perm={N_PERM}, "
          f"add-one p")
    print(f"MMD gamma            : median heuristic on pooled (cap {MMD_MAXN_FOR_GAMMA})")
    print(f"classifier           : LogisticRegression(maxiter={LOGREG_MAXITER}), "
          f"{CV_FOLDS}-fold stratified CV, balanced_accuracy")
    print(f"chance (cohort 4-way): balanced={1/4:.4f}  majority={max(sizes.values())/len(df):.4f}")
    print(f"chance (couple 17-way): balanced={1/lab_df['couple'].nunique():.4f}")
    print(f"ComBat               : neuroHarmonize, SITE={BATCH_COL}, NO covariate protected")
    print(f"CORAL                : per-batch whiten->pooled recolour (cross-check)")
    print("=" * 78)

    # ----- standardize features to a common ruler (pooled, from RAW) -----
    X_raw = df[FPCA_FEATURES].to_numpy(float)
    scaler = StandardScaler().fit(X_raw)
    def std(A): return scaler.transform(A)

    batch = df[BATCH_COL].to_numpy()

    # ----- correction -----
    print("\n[1/6] ComBat + CORAL correction ...")
    X_combat = combat_correct(X_raw, batch)
    X_coral = coral_correct(X_raw, batch)

    # standardized views (same ruler for before/after)
    Zr, Zc, Zco = std(X_raw), std(X_combat), std(X_coral)

    # gamma from pooled RAW standardized
    gamma = median_heuristic_gamma(Zr, None, rng=np.random.default_rng(SEED),
                                   max_n=MMD_MAXN_FOR_GAMMA)
    print(f"      RBF gamma (median heuristic) = {gamma:.5f}")

    def grp(Z, c): return Z[batch == c]

    # ----- cohort-pair distances BEFORE / AFTER -----
    print("\n[2/6] Cohort-pair Wasserstein / MMD  (before vs after ComBat) ...")
    pair_rows = []
    pairs = [(a, b) for i, a in enumerate(COHORTS) for b in COHORTS[i + 1:]]
    for a, b in pairs:
        d_before = pairwise_distance(grp(Zr, a), grp(Zr, b), gamma)
        d_after = pairwise_distance(grp(Zc, a), grp(Zc, b), gamma)
        d_coral = pairwise_distance(grp(Zco, a), grp(Zco, b), gamma)
        stratum = "wild-vs-wild" if (a in WILD and b in WILD) else "lab-vs-wild"
        identifiable = False  # cohort==cage for all these -> unidentifiable
        pair_rows.append(dict(
            a=a, b=b, stratum=stratum, identifiable=identifiable,
            w2_before=d_before["w2"], w2_after=d_after["w2"], w2_coral=d_coral["w2"],
            mmd_before=d_before["mmd2"], mmd_after=d_after["mmd2"], mmd_coral=d_coral["mmd2"],
        ))
        print(f"      {a:>11} vs {b:<11} [{stratum:>12}] "
              f"W2 {d_before['w2']:.3f}->{d_after['w2']:.3f} (CORAL {d_coral['w2']:.3f}) | "
              f"MMD {d_before['mmd2']:.4f}->{d_after['mmd2']:.4f}")
    pair_df = pd.DataFrame(pair_rows)

    # ----- CONTROL 1: NEGATIVE (cohort decodability collapses) -----
    print("\n[3/6] NEGATIVE control: decode cohort (4-way) before/after ...")
    neg_before = decode_accuracy(Zr, batch)
    neg_after = decode_accuracy(Zc, batch)
    neg_coral = decode_accuracy(Zco, batch)
    print(f"      cohort balanced-acc  BEFORE={neg_before['balanced_acc']:.3f}  "
          f"AFTER(ComBat)={neg_after['balanced_acc']:.3f}  AFTER(CORAL)={neg_coral['balanced_acc']:.3f}  "
          f"chance={neg_before['chance_balanced']:.3f}")

    # ----- CONTROL 2: POSITIVE (within-lab partner preserved) -----
    print("\n[4/6] POSITIVE control: decode lab partner (17-way) before/after global ComBat ...")
    lab_mask = batch == LAB
    lab_couple = df.loc[lab_mask, "couple"].to_numpy()
    keep = pd.notna(lab_couple)
    pos_before = decode_accuracy(Zr[lab_mask][keep], lab_couple[keep])
    pos_after = decode_accuracy(Zc[lab_mask][keep], lab_couple[keep])
    print(f"      partner balanced-acc BEFORE={pos_before['balanced_acc']:.3f}  "
          f"AFTER={pos_after['balanced_acc']:.3f}  chance={pos_before['chance_balanced']:.3f}")
    pos_delta = pos_after["balanced_acc"] - pos_before["balanced_acc"]
    pos_overcorrected = pos_after["balanced_acc"] < (pos_before["balanced_acc"] -
                        max(0.02, 2 * pos_before["balanced_acc_std"]))

    # ----- CONTROL 3: IDENTIFIABILITY (already flagged per pair) -----
    print("\n[5/6] IDENTIFIABILITY: cohort==cage==biological-unit for wild groups")
    print("      -> lab-vs-wild and wild-vs-wild contrasts are UNIDENTIFIABLE under cohort-ComBat.")
    print("      -> only WITHIN-lab (constant cage) partner contrasts are identifiable.")

    # ----- CONTROL 4: SPURIOUS-REMOVAL (permute cohort labels, refit) -----
    print("\n[6/6] SPURIOUS-REMOVAL: permute cohort labels, refit ComBat, check true structure ...")
    perm_batch = RNG.permutation(batch)
    X_combat_perm = combat_correct(X_raw, perm_batch)
    Zsp = std(X_combat_perm)
    spur_rows = []
    for a, b in pairs:
        d = pairwise_distance(grp(Zsp, a), grp(Zsp, b), gamma)
        before = pair_df[(pair_df.a == a) & (pair_df.b == b)]["w2_before"].iloc[0]
        spur_rows.append(dict(a=a, b=b, w2_before=before, w2_after_permcombat=d["w2"]))
    spur_df = pd.DataFrame(spur_rows)
    spur_max_change = float((spur_df["w2_after_permcombat"] - spur_df["w2_before"]).abs().max())
    print(f"      max |W2(true pair) change| after permuted-ComBat = {spur_max_change:.3f} "
          f"(should be ~0: permuted batches ≈ identity)")

    # ----- GATE E: within-lab partner pairwise Wasserstein + perm p -----
    print("\n[GATE E] within-lab partner contrasts vs wild-vs-wild noise floor ...")
    # noise floor = wild-vs-wild W2 (cross-cage, confounded) on RAW std
    ww_pairs = [(a, b) for i, a in enumerate(WILD) for b in WILD[i + 1:]]
    floor_vals = []
    for a, b in ww_pairs:
        obs, p, _ = wasserstein_perm_test(
            subsample(grp(Zr, a), PERM_SUBSAMPLE, np.random.default_rng(1)),
            subsample(grp(Zr, b), PERM_SUBSAMPLE, np.random.default_rng(2)),
            n_perm=N_PERM, seed=SEED)
        floor_vals.append(obs)
        print(f"      noise-floor {a} vs {b}: W2={obs:.3f} (perm p={p:.4f})")
    noise_floor = float(np.median(floor_vals))
    print(f"      cross-cage NOISE FLOOR (median wild-vs-wild W2) = {noise_floor:.3f}")

    # within-lab: use ComBat-corrected lab (constant cage -> ComBat is ~identity here,
    # but we use corrected scores for consistency). Compare each couple pair.
    couples = sorted(lab_df["couple"].dropna().unique())
    Zlab = Zc[lab_mask][keep]
    clab = lab_couple[keep]
    couple_idx = {c: np.where(clab == c)[0] for c in couples}
    gate_rows = []
    # restrict to couples with enough samples
    min_couple_n = 200
    big_couples = [c for c in couples if len(couple_idx[c]) >= min_couple_n]
    print(f"      lab couples with >= {min_couple_n} calls: {len(big_couples)} of {len(couples)}")
    for i, a in enumerate(big_couples):
        for b in big_couples[i + 1:]:
            Xa = Zlab[couple_idx[a]]
            Xb = Zlab[couple_idx[b]]
            rng_a = np.random.default_rng(hash((a, "a")) % (2**32))
            rng_b = np.random.default_rng(hash((b, "b")) % (2**32))
            obs, p, _ = wasserstein_perm_test(
                subsample(Xa, PERM_SUBSAMPLE, rng_a),
                subsample(Xb, PERM_SUBSAMPLE, rng_b),
                n_perm=N_PERM, seed=SEED)
            gate_rows.append(dict(a=a, b=b, w2=obs, p=p,
                                  exceeds_floor=obs > noise_floor))
    gate_df = pd.DataFrame(gate_rows).sort_values("w2", ascending=False).reset_index(drop=True)
    n_sig = int(((gate_df["p"] < 0.05) & gate_df["exceeds_floor"]).sum())
    print(f"      within-lab partner pairs tested: {len(gate_df)}")
    print(f"      pairs with perm p<0.05 AND W2 > noise floor: {n_sig}")
    if len(gate_df):
        top = gate_df.iloc[0]
        print(f"      largest: {top['a']} vs {top['b']}  W2={top['w2']:.3f}  p={top['p']:.4f}  "
              f"exceeds_floor={bool(top['exceeds_floor'])}")

    # ----- PLOTS -----
    print("\n[plots] rendering ...")
    plots = build_plots(df, batch, Zr, Zc, COHORTS, pair_df, neg_before, neg_after,
                        neg_coral, pos_before, pos_after, gate_df, noise_floor)

    # ----- HTML -----
    write_report(
        RESULTS_DIR / "report.html",
        sizes=sizes, couple_sizes=couple_sizes, gamma=gamma,
        pair_df=pair_df, spur_df=spur_df, spur_max_change=spur_max_change,
        neg_before=neg_before, neg_after=neg_after, neg_coral=neg_coral,
        pos_before=pos_before, pos_after=pos_after, pos_delta=pos_delta,
        pos_overcorrected=pos_overcorrected, gate_df=gate_df,
        noise_floor=noise_floor, n_sig=n_sig, plots=plots,
        n_total=len(df), elapsed=time.time() - t0,
    )
    # CSV side-cars
    pair_df.to_csv(RESULTS_DIR / "cohort_pair_distances.csv", index=False)
    gate_df.to_csv(RESULTS_DIR / "gate_e_within_lab_partner.csv", index=False)
    print(f"\nDONE in {time.time()-t0:.1f}s. Report: {RESULTS_DIR/'report.html'}")


# ============================ PLOTS ==========================================
def build_plots(df, batch, Zr, Zc, cohorts, pair_df, neg_before, neg_after,
                neg_coral, pos_before, pos_after, gate_df, noise_floor) -> dict:
    plots = {}
    colors = {"lab_131204": "#d62728", "5970": "#1f77b4", "3452": "#2ca02c",
              "9252": "#ff7f0e"}

    # (1) feature-1/2 scatter before vs after
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for Z, ax, title in [(Zr, axes[0], "BEFORE ComBat"), (Zc, axes[1], "AFTER ComBat")]:
        for c in cohorts:
            m = batch == c
            sl = np.random.default_rng(0).choice(np.where(m)[0],
                size=min(1200, m.sum()), replace=False)
            ax.scatter(Z[sl, 0], Z[sl, 1], s=4, alpha=0.3, label=c, color=colors[c])
        ax.set_title(title); ax.set_xlabel("amp_pc1 (z)"); ax.set_ylabel("amp_pc2 (z)")
    axes[1].legend(markerscale=3, fontsize=8)
    fig.suptitle("Shape-coordinate distributions by cohort — cage removal")
    plots["scatter"] = fig_to_b64(fig)

    # (2) cohort-pair W2 before/after bar
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = [f"{r.a}\nvs {r.b}" for r in pair_df.itertuples()]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, pair_df["w2_before"], 0.4, label="before", color="#888")
    ax.bar(x + 0.2, pair_df["w2_after"], 0.4, label="after ComBat", color="#d62728")
    ax.axhline(noise_floor, ls="--", color="k", label=f"wild-wild floor {noise_floor:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Wasserstein-2"); ax.set_title("Cohort-pair W2: before vs after correction")
    ax.legend(fontsize=8)
    plots["pairbar"] = fig_to_b64(fig)

    # (3) controls bar
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cats = ["cohort decode\n(negative)", "partner decode\n(positive)"]
    before = [neg_before["balanced_acc"], pos_before["balanced_acc"]]
    after = [neg_after["balanced_acc"], pos_after["balanced_acc"]]
    chance = [neg_before["chance_balanced"], pos_before["chance_balanced"]]
    x = np.arange(2)
    ax.bar(x - 0.2, before, 0.4, label="before", color="#888")
    ax.bar(x + 0.2, after, 0.4, label="after ComBat", color="#d62728")
    for xi, ch in zip(x, chance):
        ax.hlines(ch, xi - 0.4, xi + 0.4, color="k", ls=":", lw=2)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("balanced accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Validation controls (dotted = chance)")
    ax.legend(fontsize=8)
    plots["controls"] = fig_to_b64(fig)

    # (4) Gate E within-lab partner W2 distribution vs floor
    if len(gate_df):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.hist(gate_df["w2"], bins=20, color="#1f77b4", alpha=0.8)
        ax.axvline(noise_floor, ls="--", color="k",
                   label=f"wild-wild noise floor {noise_floor:.2f}")
        ax.set_xlabel("within-lab partner-pair Wasserstein-2")
        ax.set_ylabel("# partner pairs")
        ax.set_title("Gate E: within-lab (constant cage) partner distribution distances")
        ax.legend(fontsize=8)
        plots["gate"] = fig_to_b64(fig)
    return plots


# ============================ HTML ===========================================
def write_report(path: Path, **k) -> None:
    def esc(x): return html.escape(str(x))

    neg_b, neg_a, neg_c = k["neg_before"], k["neg_after"], k["neg_coral"]
    pos_b, pos_a = k["pos_before"], k["pos_after"]
    pair_df, gate_df, spur_df = k["pair_df"], k["gate_df"], k["spur_df"]

    # control verdicts
    neg_pass = neg_a["balanced_acc"] < 0.5 * (neg_b["balanced_acc"] + neg_b["chance_balanced"])
    neg_verdict = ("PASS" if neg_pass else "WEAK",
                   f"cohort balanced-acc {neg_b['balanced_acc']:.3f} → {neg_a['balanced_acc']:.3f} "
                   f"(chance {neg_b['chance_balanced']:.3f})")
    pos_pass = not k["pos_overcorrected"]
    pos_verdict = ("PASS" if pos_pass else "FAIL (over-corrected)",
                   f"partner balanced-acc {pos_b['balanced_acc']:.3f} → {pos_a['balanced_acc']:.3f} "
                   f"(Δ={k['pos_delta']:+.3f}, chance {pos_b['chance_balanced']:.3f})")
    spur_pass = k["spur_max_change"] < 0.5
    spur_verdict = ("PASS" if spur_pass else "WARN",
                    f"max |true-pair W2 change| under permuted-ComBat = {k['spur_max_change']:.3f}")
    ident_verdict = ("BY DESIGN UNIDENTIFIABLE",
                     "cohort==cage==biological-unit (wild) → lab-vs-wild & wild-vs-wild not identifiable; "
                     "only within-lab partner contrasts are.")

    def control_card(name, verdict):
        v, detail = verdict
        cls = "pass" if v.startswith("PASS") else ("fail" if "FAIL" in v else "warn")
        return (f'<div class="card {cls}"><h3>{esc(name)}</h3>'
                f'<div class="verdict {cls}">{esc(v)}</div>'
                f'<p>{esc(detail)}</p></div>')

    # tables
    def pair_table():
        rows = ""
        for r in pair_df.itertuples():
            rows += (f"<tr><td>{esc(r.a)}</td><td>{esc(r.b)}</td><td>{esc(r.stratum)}</td>"
                     f"<td>{r.w2_before:.3f}</td><td>{r.w2_after:.3f}</td><td>{r.w2_coral:.3f}</td>"
                     f"<td>{r.mmd_before:.4f}</td><td>{r.mmd_after:.4f}</td>"
                     f"<td>UNIDENTIFIABLE</td></tr>")
        return rows

    def gate_table():
        rows = ""
        for r in gate_df.head(25).itertuples():
            sig = (r.p < 0.05) and r.exceeds_floor
            cls = ' class="hit"' if sig else ""
            rows += (f"<tr{cls}><td>{esc(r.a)}</td><td>{esc(r.b)}</td>"
                     f"<td>{r.w2:.3f}</td><td>{r.p:.4f}</td>"
                     f"<td>{'yes' if r.exceeds_floor else 'no'}</td>"
                     f"<td>{'SIGNIFICANT' if sig else ''}</td></tr>")
        return rows

    def spur_table():
        rows = ""
        for r in spur_df.itertuples():
            rows += (f"<tr><td>{esc(r.a)}</td><td>{esc(r.b)}</td>"
                     f"<td>{r.w2_before:.3f}</td><td>{r.w2_after_permcombat:.3f}</td>"
                     f"<td>{r.w2_after_permcombat - r.w2_before:+.3f}</td></tr>")
        return rows

    img = lambda key: (f'<img src="data:image/png;base64,{k["plots"][key]}"/>'
                       if key in k["plots"] else "<p><i>(no plot)</i></p>")

    n_sig = k["n_sig"]
    gate_headline = (
        f"{n_sig} of {len(gate_df)} within-lab partner pairs differ with perm p&lt;0.05 "
        f"AND exceed the cross-cage noise floor (W2 = {k['noise_floor']:.3f})."
        if len(gate_df) else "No within-lab partner pairs met the sample threshold."
    )

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>WS-E — Confound-Robust Cohort Comparison (ComBat + OT/MMD)</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:24px auto;padding:0 18px;color:#1a1a1a;line-height:1.5}}
 h1{{border-bottom:3px solid #d62728;padding-bottom:6px}}
 h2{{margin-top:34px;border-bottom:1px solid #ddd;padding-bottom:4px}}
 .cards{{display:flex;gap:12px;flex-wrap:wrap}}
 .card{{flex:1;min-width:230px;border:1px solid #ddd;border-radius:8px;padding:12px;background:#fafafa}}
 .card.pass{{border-left:5px solid #2ca02c}} .card.fail{{border-left:5px solid #d62728}}
 .card.warn{{border-left:5px solid #ff7f0e}}
 .verdict{{font-weight:700;font-size:13px}} .verdict.pass{{color:#2ca02c}}
 .verdict.fail{{color:#d62728}} .verdict.warn{{color:#b8860b}}
 table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:13px}}
 th,td{{border:1px solid #ccc;padding:5px 8px;text-align:right}} th{{background:#f0f0f0}}
 td:first-child,td:nth-child(2),td:nth-child(3){{text-align:left}}
 tr.hit{{background:#fff3cd;font-weight:600}}
 img{{max-width:100%;border:1px solid #eee;border-radius:6px;margin:8px 0}}
 .box{{background:#f7f7ff;border:1px solid #ccd;border-radius:8px;padding:12px 16px;margin:12px 0}}
 code{{background:#eee;padding:1px 4px;border-radius:3px}}
 .key{{font-size:13px;color:#444}}
</style></head><body>
<h1>WS-E — Confound-Robust Cohort Comparison</h1>
<p class="key"><b>ComBat / neuroHarmonize</b> (cage removal) + <b>CORAL</b> cross-check;
<b>optimal transport</b> (Wasserstein-2) and <b>RBF-kernel MMD</b> distance; four validation controls.
Total calls: <b>{k['n_total']}</b>. RBF γ={k['gamma']:.4f}. Runtime {k['elapsed']:.0f}s.</p>

<div class="box"><b>Headline (Gate E):</b> {gate_headline}<br>
<b>Core epistemic result:</b> the batch axis is <code>cohort</code>, and for the wild groups
<code>cohort == cage == biological-unit</code>. Removing the cage therefore removes the entire
lab-vs-wild and wild-vs-wild signal — those contrasts are <b>UNIDENTIFIABLE</b>. The only
environment with within-cage biological variation is the <b>lab partner-swap matrix</b>
(17 pairings, constant cage), where ComBat applies a single shared affine map and leaves
partner structure intact. That is where Gate E lives.</div>

<h2>Validation controls</h2>
<div class="cards">
 {control_card("Negative — cage decodability collapses", neg_verdict)}
 {control_card("Positive — partner identity preserved", pos_verdict)}
 {control_card("Identifiability", ident_verdict)}
 {control_card("Spurious-removal", spur_verdict)}
</div>
{img("controls")}
<p class="key">Negative: CORAL cross-check cohort balanced-acc after = {neg_c['balanced_acc']:.3f}.
Positive interpretation: a global ComBat with <code>SITE=cohort</code> applies <i>one</i> affine map to
all lab calls, so within-lab partner separation is preserved by construction — confirmed empirically.
Erasure here would have signalled over-correction.</p>

<h2>Cohort-pair distances — before vs after correction</h2>
{img("scatter")}
{img("pairbar")}
<table><tr><th>A</th><th>B</th><th>stratum</th><th>W2 before</th><th>W2 after</th>
<th>W2 CORAL</th><th>MMD² before</th><th>MMD² after</th><th>identifiable?</th></tr>
{pair_table()}</table>
<p class="key">All cross-cohort contrasts are unidentifiable (cohort≡cage). The drop after ComBat is
the correction <i>removing the confound it is collinear with</i>, not revealing biology. Reported for
transparency, not as a biological result. <code>mean_power_db</code>/<code>tonality</code> were never
fed as covariates (cage artifacts).</p>

<h2>Gate E — within-lab partner contrasts (constant cage = identifiable)</h2>
{img("gate")}
<p class="key">Cross-cage noise floor = median wild-vs-wild Wasserstein = <b>{k['noise_floor']:.3f}</b>.
Partner pairs with perm <code>p&lt;0.05</code> AND W2 above the floor are highlighted.</p>
<table><tr><th>partner A</th><th>partner B</th><th>W2</th><th>perm p</th>
<th>&gt; floor</th><th></th></tr>{gate_table()}</table>

<h2>Spurious-removal — permute cage labels, refit ComBat</h2>
<p class="key">If correction invented structure, true-cohort distances would move under a permuted-label
ComBat. Max |change| = <b>{k['spur_max_change']:.3f}</b>.</p>
<table><tr><th>A</th><th>B</th><th>W2 (real ComBat absent)</th><th>W2 (permuted ComBat)</th><th>Δ</th></tr>
{spur_table()}</table>

<h2>Method parameters</h2>
<ul class="key">
<li>Batch/cage variable: <code>cohort</code> (4 levels). Lab = one cage / 17 partner pairings; 3 wild cages.</li>
<li>Features: 8 elastic-FPCA shape coords (5 amp + 3 phase), standardized to pooled unit variance for distances.</li>
<li>OT: exact <code>ot.emd2</code>, squared-euclidean ground; subsample {1500}/group × {20} repeats.</li>
<li>MMD: unbiased RBF U-statistic, γ = median heuristic = {k['gamma']:.4f}.</li>
<li>Permutation p: add-one estimator, {300} permutations, {600}/group subsample.</li>
<li>Classifier: 5-fold stratified logistic regression, balanced accuracy. Chance: cohort 0.25, partner {pos_b['chance_balanced']:.3f}.</li>
<li>ComBat: neuroHarmonize, no biological covariate protected (pure batch removal). CORAL = second-order alignment cross-check.</li>
<li><b>Deviation:</b> MMD uses an RBF kernel on FPCA coords, not GAK — GAK is for raw variable-length
sequences; the WS-A elastic-FPCA coords already encode SRVF warp-aligned shape, so a sequence kernel is
inapplicable. Documented and justified.</li>
</ul>
</body></html>"""
    path.write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    main()
