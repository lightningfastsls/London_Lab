"""STAGE C (box, repo .venv) — score the proper-JTFS M2b arm on the locked gate.

Loads the JTFS feature matrices produced on the rig (Stage B), embeds them
(z-score -> PCA<=50, identical recipe to the Scattering2D substitute), runs the
reversal unit test (fwd vs time-reversed waveform features), the direction-augment
remedy, and the locked `harness.benchmark` (4 settings + per-family CIs). Compares
against the incumbent soft-DTW bar and the registration identity, and applies the
handoff's Thread-1 decision gate.

NOTHING here imports kymatio: the repo .venv stays pinned at 0.3.0; the principled
JTFS lived only in the rig pass. The harness, loader, io, and m2b helpers are reused
verbatim (m2b imports kymatio lazily, so importing the module is safe).

Run:
  .venv/bin/python scripts/experiments/shape_invariance/eval_m2b_jtfs_real.py \
      --feat-dir <rig_output_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(_EXP))
for p in (_EXP, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from shape_invariance import harness, io, loader            # noqa: E402
from shape_invariance.methods import m2b_jtfs as m2b        # noqa: E402
from eval_shape_human_anchored import loo_knn_purity        # noqa: E402

FAMILIES = ["chevron", "jump", "flat", "complex", "Noise", "Down-FM", "Up-FM", "Short"]
PRIMARY = ["chevron", "jump", "flat", "complex"]
OUTDIR = "results/shape_invariance"
K, KS, N_BOOT, SEED, N_COMP = 10, (1, 5, 15), 1000, 42, 50


def _nonoverlap(a, b):
    """True if interval a strictly beats b (a_lo > b_hi)."""
    return a[1] > b[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    print("=" * 96)
    print("STAGE C — M2b PROPER JTFS (TimeFrequencyScattering) — locked-gate eval")
    print("=" * 96)
    print(f"PARAMS: k={K} ks={KS} n_boot={N_BOOT} seed={SEED} n_comp={N_COMP}")

    data = loader.load_labeled()
    family = data["family"]
    slopes = np.array([m2b.net_slope(c) for c in data["contour50"]])

    manifest = json.load(open(os.path.join(args.feat_dir, "jtfs_manifest.json")))
    print(f"JTFS manifest: N={manifest['N']} device={manifest['device']} "
          f"configs={[c['name'] for c in manifest['configs']]}")

    # --- config sweep: pick best by primary-mean pooled purity (point) ---
    print("\n" + "-" * 96)
    print("SWEEP (pooled_invariant point purity, k=10): JTFS configs (F = transposition scale)")
    print("-" * 96)
    sweep, best = [], None
    cache = {}
    for cfg in manifest["configs"]:
        Ff = np.load(os.path.join(args.feat_dir, f"jtfs_{cfg['name']}_fwd.npy"))
        Fr = np.load(os.path.join(args.feat_dir, f"jtfs_{cfg['name']}_rev.npy"))
        assert Ff.shape[0] == len(family), f"{cfg['name']}: {Ff.shape[0]} != {len(family)}"
        Xp, mu, sd, pca = m2b._fit_embed(Ff, n_comp=N_COMP, seed=SEED)
        Xp_rev = m2b._apply_embed(Fr, mu, sd, pca)
        cache[cfg["name"]] = (Xp, Xp_rev, Ff.shape[1])
        pts = {f: loo_knn_purity(Xp, family, f, k=K)[0] for f in PRIMARY}
        score = float(np.mean([pts[f] for f in PRIMARY]))
        row = {**cfg, "d_flat": int(Ff.shape[1]), "d_pca": int(Xp.shape[1]),
               "purity": pts, "primary_mean": score}
        sweep.append(row)
        print(f"  {cfg['name']:>3} F={cfg['F']}: d_flat={Ff.shape[1]:>5} d_pca={Xp.shape[1]:>3} "
              + " ".join(f"{f}={pts[f]:.3f}" for f in PRIMARY) + f"  mean={score:.3f}")
        if best is None or score > best["primary_mean"]:
            best = row

    bn = best["name"]
    Xp, Xp_rev, d_flat = cache[bn]
    print(f"\nBEST config = {bn} (F={best['F']}, primary_mean={best['primary_mean']:.3f})")

    params = {"method": "m2b_jtfs_real", "transform": "TimeFrequencyScattering (kymatio 0.4.dev, JTFS)",
              "input": "raw call waveform (NOT spectrogram image)", "N": manifest["N"],
              **{k: best[k] for k in ("J", "Q", "J_fr", "Q_fr", "F")}, "T": "N (1 frame)",
              "n_comp": int(Xp.shape[1]), "device": manifest["device"],
              "k": K, "n_boot": N_BOOT, "seed": SEED}
    feat_inv = io.save_features("m2b_jtfs_real", Xp, params)
    print(f"invariant-only features (d={Xp.shape[1]}) -> {feat_inv}")

    # --- reversal test (invariant-only): time-reversed WAVEFORM features ---
    rev = m2b._reversal_test_from_feats(Xp, Xp_rev, seed=SEED)
    print(f"\n[REVERSAL invariant-only] passed={rev['passed']} "
          f"self_rev_median={rev['self_reverse_median']:.4f} "
          f"pairwise_p90={rev['decile_threshold']:.4f}")

    # --- direction-augmented variant (net slope flips under reversal) + re-test ---
    Xd, slope_w, (smu, ssd) = m2b.append_direction(Xp, slopes)
    sz_rev = (-slopes - smu) / (ssd or 1.0)
    Xd_rev = np.hstack([Xp_rev, (slope_w * sz_rev)[:, None]])
    rev_dir = m2b._reversal_test_from_feats(Xd, Xd_rev, seed=SEED)
    print(f"[REVERSAL +direction] slope_weight={slope_w:.4f}: passed={rev_dir['passed']} "
          f"self_rev_median={rev_dir['self_reverse_median']:.4f} "
          f"pairwise_p90={rev_dir['decile_threshold']:.4f}")
    feat_dir = io.save_features("m2b_jtfs_real_dir", Xd,
                                {**params, "descriptor": "jtfs_pca+net_slope", "slope_weight": slope_w})
    print(f"head-to-head (direction-augmented) features (d={Xd.shape[1]}) -> {feat_dir}")

    # --- benchmark (head-to-head = direction-augmented): 4 settings + per-family ---
    print("\n" + "-" * 96)
    print("BENCHMARK (head-to-head = direction-augmented; 4 settings)")
    print("-" * 96)
    res = harness.benchmark(Xd, kind="embedding", meta=data, families=FAMILIES,
                            k=K, ks=KS, side=data["side"], n_boot=N_BOOT, seed=SEED)
    pi = res["pooled_invariant"]
    for f in PRIMARY:
        print(f"  {f:>8}: {pi[f][0]:.3f} [{pi[f][1]:.3f},{pi[f][2]:.3f}]")

    # --- compare vs the incumbent bar + identity, apply the gate ---
    base = json.load(open(os.path.join(OUTDIR, "baselines_result.json")))
    sdtw = base["soft_dtw(ELASTIC)"]["pooled_invariant"]
    iden = base["registration_euclidean(IDENTITY)"]["pooled_invariant"]
    vs_sdtw = {f: {"jtfs": pi[f], "softdtw": sdtw[f],
                   "jtfs_beats_softdtw_nonoverlap": _nonoverlap(pi[f], sdtw[f]),
                   "softdtw_beats_jtfs_nonoverlap": _nonoverlap(sdtw[f], pi[f])} for f in PRIMARY}
    vs_iden = {f: {"jtfs": pi[f], "identity": iden[f],
                   "jtfs_beats_identity_nonoverlap": _nonoverlap(pi[f], iden[f])} for f in PRIMARY}

    beats_jump = vs_sdtw["jump"]["jtfs_beats_softdtw_nonoverlap"]
    beats_complex = vs_sdtw["complex"]["jtfs_beats_softdtw_nonoverlap"]
    if beats_jump and beats_complex:
        verdict = ("JTFS BEATS soft-DTW on BOTH jump and complex (non-overlapping) -> STRONG "
                   "VAE-diagnostic: the principled, non-learned freq-transposition path WINS. "
                   "Warrants a learned-encoder-on-scattering-front-end follow-up.")
        gate = "BEATS"
    elif beats_jump or beats_complex:
        fam = "jump" if beats_jump else "complex"
        verdict = (f"JTFS beats soft-DTW on {fam} only (non-overlapping); ties on the other. "
                   "Partial win -> principled invariance helps on one family; not a clean dethrone.")
        gate = "PARTIAL"
    else:
        verdict = ("JTFS only TIES soft-DTW (no non-overlapping win on jump or complex) -> "
                   "consistent with the Scattering2D substitute; CLOSE M2b. Confirms a FIXED "
                   "non-learned spectrotemporal transform matches (not beats) the elastic bar, "
                   "so the 7 prior VAE failures were about the LEARNED PIXEL OBJECTIVE, not "
                   "spectrograms or transposition invariance per se.")
        gate = "TIE"
    print(f"\nDECISION GATE = {gate}\n  {verdict}")

    payload = {
        "method": "m2b_jtfs_real", "status": "complete", "arm": "VAE-diagnostic (principled JTFS)",
        "transform": "kymatio TimeFrequencyScattering (joint time-frequency scattering)",
        "library_note": ("kymatio 0.4.0.dev0 (git main) installed isolated to a throwaway dir + "
                         "shipped to rig; box .venv stays pinned at 0.3.0. JTFS on raw waveform "
                         "(N=131072 @ canonical SR), torch backend, cuda:0."),
        "input_note": ("Per-call WAVEFORM (not a spectrogram image). T=N -> 1 time frame -> "
                       "fixed-length per-call descriptor. F = freq-transposition-invariance scale."),
        "params": params, "config_sweep": sweep, "best_config": best,
        "d_flat": int(d_flat), "d_pca": int(Xp.shape[1]),
        "reversal": rev, "reversal_direction_augmented": rev_dir,
        "purity": res, "vs_softdtw_pooled_invariant": vs_sdtw,
        "vs_identity_pooled_invariant": vs_iden,
        "decision_gate": gate, "verdict": verdict,
        "feature_path": feat_dir, "feature_path_invariant_only": feat_inv,
    }
    out_json = os.path.join(OUTDIR, "m2b_jtfs_real_result.json")
    json.dump(payload, open(out_json, "w"), indent=2, default=str)
    print(f"\n[OUT] {out_json}")
    _write_html(payload, sdtw, iden)
    print("STAGE C done.")


def _write_html(p, sdtw, iden):
    pi = p["purity"]["pooled_invariant"]
    sub = None
    sub_path = os.path.join(OUTDIR, "m2b_jtfs_result.json")
    if os.path.exists(sub_path):
        sub = json.load(open(sub_path)).get("purity", {}).get("pooled_invariant")

    def cell(v):
        return f"{v[0]:.3f} <span style='color:#888'>[{v[1]:.3f},{v[2]:.3f}]</span>"

    rows = ""
    for f in PRIMARY:
        rows += (f"<tr><td><b>{f}</b></td><td>{cell(iden[f])}</td>"
                 f"<td style='background:#eef'>{cell(sdtw[f])}</td>"
                 f"<td>{cell(sub[f]) if sub else '—'}</td>"
                 f"<td style='background:#efe'>{cell(pi[f])}</td></tr>")
    gate_color = {"BEATS": "#cfc", "PARTIAL": "#ffd", "TIE": "#eee"}.get(p["decision_gate"], "#eee")
    html = f"""<!doctype html><meta charset=utf-8>
<title>M2b proper JTFS vs the bar</title>
<style>body{{font:14px/1.5 system-ui;margin:2rem;max-width:980px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:6px 10px;text-align:left}}
th{{background:#f4f4f4}}.gate{{padding:1rem;border-radius:8px;background:{gate_color};margin:1rem 0}}</style>
<h1>M2b — proper Joint Time-Frequency Scattering (VAE-diagnostic)</h1>
<p><b>Transform:</b> {p['transform']} on the raw call waveform (N={p['params']['N']},
F-sweep, T=N). {p['library_note']}</p>
<div class=gate><b>DECISION GATE = {p['decision_gate']}.</b><br>{p['verdict']}</div>
<h2>Pooled invariant kNN purity (k=10, 1000× bootstrap CI)</h2>
<table><tr><th>family</th><th>identity (registration)</th><th>soft-DTW (THE BAR)</th>
<th>M2b Scattering2D substitute</th><th>M2b proper JTFS</th></tr>{rows}</table>
<p style='color:#666'>Best JTFS config: {p['best_config']['name']} (F={p['best_config']['F']},
J={p['best_config']['J']}, Q={p['best_config']['Q']}, J_fr={p['best_config']['J_fr']});
PCA d={p['d_pca']} of {p['d_flat']}. Reversal (invariant-only) passed={p['reversal']['passed']};
+direction passed={p['reversal_direction_augmented']['passed']}.</p>
"""
    out = os.path.join(OUTDIR, "m2b_jtfs_real_comparison.html")
    open(out, "w").write(html)
    print(f"[OUT] {out}")


if __name__ == "__main__":
    main()
