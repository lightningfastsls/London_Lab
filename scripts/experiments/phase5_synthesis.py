#!/usr/bin/env python
"""Phase 5 synthesis: unify the shape-invariance benchmark results.

Reads results/shape_invariance/*_result.json (baselines + M2a/M2b/M3/M4/M5),
emits a self-contained HTML comparison report + machine-readable synthesis.json.

Truth-finding, NOT crowning a winner: maps which invariance helps which shape family.
All decisions are made on NON-overlapping 95% bootstrap CIs.
"""
import json
import os
from datetime import date

ROOT = "/home/shachar/projects/mickey_london_lab"
RDIR = os.path.join(ROOT, "results", "shape_invariance")

PRIMARY = ["chevron", "jump", "flat", "complex"]
SECONDARY = ["Noise", "Down-FM", "Up-FM", "Short"]
SETTINGS = ["pooled_invariant", "pooled_sidechannel",
            "withinstratum_invariant", "withinstratum_sidechannel"]
SETTING_LABEL = {
    "pooled_invariant": "pooled · invariant-only",
    "pooled_sidechannel": "pooled · + side-channels",
    "withinstratum_invariant": "within-stratum · invariant-only",
    "withinstratum_sidechannel": "within-stratum · + side-channels",
}
FAM_N = {"chevron": 44, "jump": 204, "flat": 125, "complex": 67,
         "Noise": 66, "Down-FM": 54, "Up-FM": 27, "Short": 24}
N_LABELS = 611


def load(name):
    with open(os.path.join(RDIR, name)) as f:
        return json.load(f)


base = load("baselines_result.json")
identity = base["registration_euclidean(IDENTITY)"]
softdtw = base["soft_dtw(ELASTIC)"]

methods = {
    "soft-DTW (BAR)": {"purity": softdtw, "d": "dist", "kind": "bar"},
    "registration-Euclidean (IDENTITY)": {"purity": identity, "d": 50, "kind": "identity"},
    "M5 turning-fn": load("m5_result.json"),
    "M4 RQA": load("m4_rqa_result.json"),
    "M3 persistence": load("m3_persist_result.json"),
    "M2a Scattering1D": load("m2a_scatter1d_result.json"),
    "M2b Scattering2D": load("m2b_jtfs_result.json"),
}


def fam_ci(purity, setting, fam):
    """Return [point, lo, hi] or None."""
    blk = purity.get(setting, {})
    if not isinstance(blk, dict) or fam not in blk:
        return None
    v = blk[fam]
    if not isinstance(v, list):
        return None
    return v


def overall_weighted(purity, setting, fams=PRIMARY):
    """N-weighted point estimate across families (CI not bootstrapped jointly)."""
    num = 0.0
    den = 0.0
    for fam in fams:
        v = fam_ci(purity, setting, fam)
        if v is None:
            return None
        num += v[0] * FAM_N[fam]
        den += FAM_N[fam]
    return num / den if den else None


def decide(a, b):
    """Non-overlapping-CI decision: a vs b. Returns 'beats'/'loses'/'ties'/None."""
    if a is None or b is None:
        return None
    if a[1] > b[2]:
        return "beats"
    if a[2] < b[1]:
        return "loses"
    return "ties"


# Random-control base rates (prevalence in the LOO pool ~ n_f / N)
base_rate = {fam: FAM_N[fam] / N_LABELS for fam in PRIMARY + SECONDARY}
overall_base = sum((FAM_N[f] / sum(FAM_N[p] for p in PRIMARY)) * base_rate[f]
                   for f in PRIMARY)

# ---------------------------------------------------------------- synthesis.json
synth = {
    "generated": str(date.today()),
    "n_labels": N_LABELS,
    "family_counts": FAM_N,
    "cohort_counts": base["cohort_counts"],
    "metric": "LOO kNN retrieval purity vs human shape family, k=10, 1000x bootstrap 95% CI",
    "decision_rule": "non-overlapping 95% CIs (never point estimates)",
    "bar": "soft-DTW (gamma=1.0, distance-native)",
    "identity": "registration-Euclidean on mean-subtracted 50-pt contour",
    "random_control_base_rate": {**{f: round(base_rate[f], 4) for f in PRIMARY + SECONDARY},
                                 "overall_primary_weighted": round(overall_base, 4)},
    "data_deviation": base["data_deviation_note"],
    "baseline_validation": base["validation"],
    "table": {},
    "per_family_decisions": {},
    "predictions": {},
    "reversal": {},
}

for mname, m in methods.items():
    purity = m["purity"]
    synth["table"][mname] = {}
    for s in SETTINGS:
        row = {"overall_primary": overall_weighted(purity, s)}
        for fam in PRIMARY + SECONDARY:
            row[fam] = fam_ci(purity, s, fam)
        synth["table"][mname][s] = row
    # decisions vs bar & identity on pooled_invariant
    if m.get("kind") not in ("bar",):
        dvs = {"vs_softdtw": {}, "vs_identity": {}}
        for fam in PRIMARY + SECONDARY:
            a = fam_ci(purity, "pooled_invariant", fam)
            dvs["vs_softdtw"][fam] = decide(a, fam_ci(softdtw, "pooled_invariant", fam))
            dvs["vs_identity"][fam] = decide(a, fam_ci(identity, "pooled_invariant", fam))
        synth["per_family_decisions"][mname] = dvs
    if isinstance(m, dict) and "prediction_held" in m:
        synth["predictions"][mname] = m["prediction_held"]
    if isinstance(m, dict) and "reversal" in m:
        r = m["reversal"]
        synth["reversal"][mname] = {
            "passed": r.get("passed"),
            "passed_after_direction": r.get("passed_after_direction"),
            "self_reverse_median": r.get("self_reverse_median"),
            "decile_threshold": r.get("decile_threshold"),
            "direction_feature_appended": r.get("direction_feature_appended"),
        }

with open(os.path.join(RDIR, "synthesis.json"), "w") as f:
    json.dump(synth, f, indent=2)

# ---------------------------------------------------------------- HTML
def ci_cell(v, ref=None, bold=False):
    if v is None:
        return '<td class="na">—</td>'
    pt, lo, hi = v
    cls = "num"
    mark = ""
    if ref is not None:
        d = decide(v, ref)
        if d == "beats":
            cls = "num win"
            mark = " ▲"
        elif d == "loses":
            cls = "num lose"
            mark = " ▼"
    inner = f'<b>{pt:.3f}</b>' if bold else f'{pt:.3f}'
    return (f'<td class="{cls}">{inner}{mark}'
            f'<span class="ci">[{lo:.3f},{hi:.3f}]</span></td>')


def overall_cell(p, bold=False):
    if p is None:
        return '<td class="na">—</td>'
    inner = f'<b>{p:.3f}</b>' if bold else f'{p:.3f}'
    return f'<td class="num">{inner}<span class="ci">N-wt</span></td>'


# Build main table rows: each method x 4 settings; ref = soft-DTW same setting per family
def softdtw_ref(setting, fam):
    return fam_ci(softdtw, setting, fam)


rows_html = []
method_order = ["soft-DTW (BAR)", "registration-Euclidean (IDENTITY)",
                "M5 turning-fn", "M4 RQA", "M3 persistence",
                "M2a Scattering1D", "M2b Scattering2D"]
for mname in method_order:
    purity = methods[mname]["purity"]
    is_bar = methods[mname].get("kind") == "bar"
    is_id = methods[mname].get("kind") == "identity"
    rowspan = len(SETTINGS)
    for i, s in enumerate(SETTINGS):
        cells = []
        ov = overall_weighted(purity, s)
        cells.append(overall_cell(ov, bold=is_bar))
        for fam in PRIMARY:
            ref = None if (is_bar) else softdtw_ref(s, fam)
            cells.append(ci_cell(fam_ci(purity, s, fam), ref=ref, bold=is_bar))
        method_td = ""
        if i == 0:
            cls = "mname bar" if is_bar else ("mname id" if is_id else "mname")
            dd = methods[mname].get("d", "")
            method_td = f'<td class="{cls}" rowspan="{rowspan}">{mname}<br><span class="d">d={dd}</span></td>'
        setting_td = f'<td class="setting">{SETTING_LABEL[s]}</td>'
        rowcls = "barrow" if is_bar else ("idrow" if is_id else "")
        rows_html.append(f'<tr class="{rowcls}">{method_td}{setting_td}{"".join(cells)}</tr>')

# random control row
rc_cells = [overall_cell(overall_base)]
for fam in PRIMARY:
    rc_cells.append(f'<td class="num rc">{base_rate[fam]:.3f}</td>')
rows_html.append(f'<tr class="rcrow"><td class="mname rc">random control<br><span class="d">base rate</span></td>'
                 f'<td class="setting">prevalence n_f/N</td>{"".join(rc_cells)}</tr>')

# Secondary-family table (context: direction story lives here) - pooled_invariant only
sec_rows = []
for mname in method_order:
    purity = methods[mname]["purity"]
    is_bar = methods[mname].get("kind") == "bar"
    cells = []
    for fam in SECONDARY:
        ref = None if is_bar else fam_ci(softdtw, "pooled_invariant", fam)
        cells.append(ci_cell(fam_ci(purity, "pooled_invariant", fam), ref=ref, bold=is_bar))
    sec_rows.append(f'<tr class="{"barrow" if is_bar else ""}"><td class="mname">{mname}</td>{"".join(cells)}</tr>')
sec_rc = "".join(f'<td class="num rc">{base_rate[f]:.3f}</td>' for f in SECONDARY)
sec_rows.append(f'<tr class="rcrow"><td class="mname rc">random control</td>{sec_rc}</tr>')

# Per-family verdict boxes
PRED_VERDICT = {
    "M5 turning-fn": ("PARTIAL",
        "Turning function ties soft-DTW on the CLEAN families (flat 0.402[.365,.441] vs bar 0.396[.362,.433]; "
        "chevron 0.214[.166,.264] vs 0.214[.168,.261]) but LOSES on jump (0.362[.332,.394] vs 0.522[.480,.570]) "
        "and complex (0.104[.079,.131] vs 0.243[.199,.284]). Exactly the predicted dissociation: shape itself is "
        "low-dimensional; <b>warp alignment is the entire value of soft-DTW</b>. M5 isolates that lever by lacking it."),
    "M4 RQA": ("HELD",
        "Relational self-distance + RQA ties soft-DTW on ALL four primaries (chevron/jump/flat/complex, overlapping CIs) "
        "and BEATS the identity incumbent on jump (0.504[.467,.545] vs 0.415[.377,.453], non-overlapping) at d=10, a "
        "fraction of soft-DTW's cost. A cheap relational stand-in for the elastic distance, as predicted. Honest caveat: "
        "pure invariant-only RQA loses to identity on flat (0.332 vs 0.419) — relational encoding discards the absolute "
        "trajectory cue flat needs; the appended direction block recovers a soft-DTW tie."),
    "M3 persistence": ("HELD",
        "Pure sublevel/superlevel persistence ties soft-DTW on all four primaries and BEATS identity on jump "
        "(0.545[.509,.585] vs 0.415[.377,.453], non-overlapping) — the multi-extrema family is topology's home run. "
        "As predicted it is sweep-DIRECTION blind: Up-FM 0.085 / Down-FM 0.176 collapse (NOT 'flat', which is zero-slope); "
        "appending the antisymmetric-slope vector LIFTS them (Up-FM→0.148, Down-FM→0.276 non-overlapping). "
        "<b>Extrema-configuration and direction cleanly separate as two orthogonal shape factors.</b>"),
    "M2a Scattering1D": ("PARTIAL",
        "A fixed deformation-stable front-end ties soft-DTW on every primary + Noise (0/5 lose) and BEATS identity on "
        "jump (0.580[.540,.625] vs 0.415) and complex (0.306[.260,.351] vs 0.194) non-overlapping. But the handoff's "
        "specific prediction of a WIN over soft-DTW on the noisy/oscillatory pocket and warped jump is NOT met — both only "
        "TIE. Deformation-stability buys PARITY with the elastic bar, not superiority."),
    "M2b Scattering2D": ("PARTIAL",
        "A FIXED (non-learned) spectrogram scattering ties soft-DTW on all four primaries and beats identity on jump "
        "(0.587[.548,.629]); the within-stratum jump win survives (0.598) so it is NOT cage leakage. "
        "<b>Spectrograms are not the blocker</b> — consistent with the 7 prior VAE failures being about the LEARNED PIXEL "
        "OBJECTIVE, not spectrograms per se. Stops short of a clean BEAT. Caveat: kymatio 0.3.0 has no "
        "TimeFrequencyScattering1D — this is isotropic Scattering2D as a fixed TF-invariant SUBSTITUTE, not true JTFS."),
}
VCLR = {"HELD": "#1b7a3d", "PARTIAL": "#9a6a00", "FALSIFIED": "#a01b1b"}
pred_boxes = []
for mname in ["M5 turning-fn", "M4 RQA", "M3 persistence", "M2a Scattering1D", "M2b Scattering2D"]:
    v, txt = PRED_VERDICT[mname]
    pred_boxes.append(
        f'<div class="vbox"><div class="vhdr"><span class="mtag">{mname}</span>'
        f'<span class="verdict" style="background:{VCLR[v]}">{v}</span></div>'
        f'<div class="vtxt">{txt}</div></div>')

# Reversal summary rows
rev_rows = []
for mname in ["M5 turning-fn", "M4 RQA", "M3 persistence", "M2a Scattering1D", "M2b Scattering2D"]:
    r = methods[mname]["reversal"]
    sr = r.get("self_reverse_median")
    th = r.get("decile_threshold")
    appended = r.get("direction_feature_appended")
    rev_rows.append(
        f'<tr><td class="mname">{mname}</td>'
        f'<td class="num">{"PASS" if r.get("passed") else "FAIL"}</td>'
        f'<td class="num">{sr:.2f}</td><td class="num">{th:.2f}</td>'
        f'<td class="num">{"yes" if appended else "no"}</td>'
        f'<td class="num">{"PASS" if r.get("passed_after_direction") else "FAIL"}</td></tr>')

prim_header = "".join(f'<th>{f}<br><span class="d">n={FAM_N[f]}</span></th>' for f in PRIMARY)
sec_header = "".join(f'<th>{f}<br><span class="d">n={FAM_N[f]}</span></th>' for f in SECONDARY)

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Shape-invariance benchmark — Phase 5 synthesis</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#f4f5f7;color:#1c2430;}}
 .wrap{{max-width:1180px;margin:0 auto;padding:28px 22px 70px;}}
 h1{{font-size:23px;margin:0 0 4px;}} h2{{font-size:18px;margin:30px 0 10px;border-bottom:2px solid #d8dce2;padding-bottom:5px;}}
 .sub{{color:#5b6573;font-size:13px;margin-bottom:6px;}}
 table{{border-collapse:collapse;width:100%;background:#fff;font-size:12.5px;box-shadow:0 1px 3px rgba(0,0,0,.08);}}
 th,td{{border:1px solid #e1e5ea;padding:5px 7px;text-align:center;}}
 th{{background:#2f3a4a;color:#fff;font-weight:600;font-size:12px;}}
 td.num{{font-variant-numeric:tabular-nums;}}
 td.mname{{text-align:left;font-weight:600;background:#f0f2f5;}}
 td.setting{{text-align:left;color:#46505e;font-size:11.5px;background:#fafbfc;}}
 .ci{{display:block;color:#8a93a0;font-size:10px;font-weight:400;}}
 .d{{color:#9aa3b0;font-size:10px;font-weight:400;}}
 .win{{color:#1b7a3d;font-weight:700;}} .lose{{color:#a01b1b;}}
 tr.barrow td{{background:#fff7e6;}} td.mname.bar{{background:#ffe9bf;}}
 tr.idrow td{{background:#eef3f8;}} td.mname.id{{background:#dbe6f1;}}
 tr.rcrow td{{background:#f3f3f3;color:#7a828d;font-style:italic;}}
 td.rc{{color:#8a929d;}}
 .na{{color:#c2c7cd;}}
 .vbox{{background:#fff;border:1px solid #e1e5ea;border-left:5px solid #2f3a4a;border-radius:4px;padding:12px 15px;margin:11px 0;box-shadow:0 1px 2px rgba(0,0,0,.05);}}
 .vhdr{{display:flex;align-items:center;gap:12px;margin-bottom:6px;}}
 .mtag{{font-weight:700;font-size:14px;}}
 .verdict{{color:#fff;font-size:11px;font-weight:700;padding:2px 9px;border-radius:10px;letter-spacing:.4px;}}
 .vtxt{{font-size:13px;line-height:1.5;color:#2a323d;}}
 .read{{background:#fff;border:1px solid #e1e5ea;border-radius:5px;padding:16px 20px;line-height:1.6;font-size:13.5px;}}
 .read li{{margin:6px 0;}}
 .key{{font-weight:700;color:#11305e;}}
 .caveat{{background:#fff8f4;border:1px solid #f0d9c8;border-left:5px solid #c8702a;border-radius:4px;padding:12px 16px;font-size:12.8px;line-height:1.55;}}
 .legend{{font-size:11.5px;color:#5b6573;margin:8px 0 2px;}}
 code{{background:#eef1f4;padding:1px 5px;border-radius:3px;font-size:11.5px;}}
</style></head><body><div class="wrap">

<h1>By-construction time/frequency-invariant USV shape representations — Phase 5 synthesis</h1>
<div class="sub">Generated {date.today()} · N={N_LABELS} human-labeled calls · metric = leave-one-out kNN retrieval purity vs human shape family, k=10, 1000× bootstrap 95% CI · <b>all decisions on NON-overlapping CIs</b> · bar = soft-DTW (γ=1.0).</div>
<div class="sub">Goal is truth-finding, not crowning a winner: map <i>which invariance buys which shape factor</i>. A method that loses overall but wins a family is a KEPT result.</div>

<h2>1 · Unified comparison — primary families</h2>
<div class="legend">Rows = method × 4 settings. <span class="win">▲ green</span> = beats soft-DTW (this setting, non-overlapping CI); <span class="lose">▼ red</span> = loses to soft-DTW. <b>soft-DTW row bolded = THE BAR</b>; registration-Euclidean = IDENTITY incumbent; random control = family prevalence. Overall = N-weighted point over the 4 primaries (no joint bootstrap → no CI).</div>
<table>
<tr><th>method</th><th>setting</th><th>overall<br><span class="d">primary, N-wt</span></th>{prim_header}</tr>
{''.join(rows_html)}
</table>

<h2>2 · Context families — direction & noise (pooled · invariant-only)</h2>
<div class="legend">The sweep-DIRECTION story (Up-FM / Down-FM) and the noisy pocket live here. ▲/▼ vs soft-DTW.</div>
<table>
<tr><th>method</th>{sec_header}</tr>
{''.join(sec_rows)}
</table>

<h2>3 · Per-family verdicts — did each logged prediction hold?</h2>
{''.join(pred_boxes)}

<h2>4 · Which invariance bought what — the written read</h2>
<div class="read">
<p><span class="key">The headline.</span> Five by-construction invariant representations were benchmarked against the elastic soft-DTW bar. <b>No method beats soft-DTW on any primary family</b> — the elastic distance is not dethroned. But four of the five <b>tie it across the board</b> while costing a fraction as much, and all four <b>beat the registration-Euclidean incumbent on jump</b> (the multi-extrema family) with non-overlapping CIs. The benchmark cleanly decomposes "shape" into separable factors:</p>
<ul>
<li><span class="key">Warp alignment is soft-DTW's entire edge.</span> M5 (turning function), which is translation/scale-invariant but has <i>no warp alignment</i>, ties the bar on the CLEAN families (flat, chevron) and loses precisely on jump (0.362 vs 0.522) and complex (0.104 vs 0.243). Removing the one lever soft-DTW has, and watching only jump/complex collapse, localizes the elastic gain to <b>temporal warp on multi-segment calls</b> — exactly the handoff's prediction.</li>
<li><span class="key">Relational / topological structure recovers most of that edge cheaply.</span> M4 (recurrence-RQA, d=10) and M3 (persistence) both encode <i>relational</i> structure rather than absolute trajectory; both tie soft-DTW on all four primaries and both beat identity on jump. M3 is the multi-extrema home run (jump 0.545). This is the cheap stand-in the handoff predicted.</li>
<li><span class="key">Extrema-configuration ⟂ direction.</span> Pure persistence is reversal/direction-blind by construction → it collapses the directed sweeps (Up-FM 0.085, Down-FM 0.176). Appending the antisymmetric-slope vector lifts them (Up-FM→0.148, Down-FM→0.276, non-overlapping) <i>without disturbing the extrema families</i>. Configuration-of-extrema and sweep-direction are two orthogonal shape axes, and we can dial them independently.</li>
<li><span class="key">Spectrograms are not the blocker.</span> M2b — a FIXED (non-learned) spectrogram scattering — ties soft-DTW on all four primaries and beats identity on jump, and the jump win survives within-stratum (0.598, not cage leakage). The seven prior VAE failures were about the <b>learned pixel objective</b>, not spectrograms per se. M2a (Scattering1D on the contour) confirms deformation-stability buys parity, not superiority.</li>
<li><span class="key">Side-channels barely move the Hz-scale methods.</span> Duration / freq-range / freq-std side-channels are dominated by the Hz-scale feature geometry (identity, scattering) and add ~nothing; they only participate for the radian-scale turning function (M5 jump 0.362→0.412 pooled). Modulation depth is kept as signal (<code>scale_invariant=False</code>).</li>
</ul>
<p><span class="key">Recommendation for the downstream manifold / sequence / biology pipelines.</span> Carry <b>two complementary representations, not one</b>:</p>
<ul>
<li><b>soft-DTW</b> (or its cheap stand-in <b>M4-RQA d=10 + direction block</b>) as the <b>labeled-set metric / jump-and-complex workhorse</b> — it owns the warp-sensitive families. M4 is the deployable surrogate when 67k×67k elastic distances are infeasible.</li>
<li><b>M3 persistence-image (+ antisymmetric slope)</b> as the <b>interpretable factor basis</b> for the manifold/grammar work: it decomposes calls into extrema-configuration ⊕ direction, the two axes the biology questions (chevron vs jump counts; up- vs down-FM) actually ask about.</li>
<li>Keep <b>M2b scattering</b> as the <b>standing VAE diagnostic</b>: a fixed front-end already matches the elastic bar, so any learned encoder must be evaluated <i>on top of</i> a scattering front-end, not against raw pixels.</li>
</ul>
</div>

<h2>5 · Reversal unit test (cross-cutting rule 1: never time-reversal invariant)</h2>
<div class="legend">Strict bar: median dist(x, reverse(x)) must exceed the 90th percentile of the pairwise distance distribution. Direction feature = signed net slope / antisymmetric turning part.</div>
<table>
<tr><th>method</th><th>native</th><th>self-reverse median</th><th>decile threshold</th><th>dir. feature appended</th><th>after direction</th></tr>
{''.join(rev_rows)}
</table>
<div class="legend" style="margin-top:8px">Every method FAILS the strict decile bar even after a signed direction feature — but this is a <b>structural property of the harness</b>, not unhandled reversal: on mean-subtracted contours even the raw-contour IDENTITY (maximally direction-sensitive) fails it (ratio 0.376), because a curve and its mirror differ by 2·‖slope‖ while the 90th-percentile inter-call separation exceeds 2·median‖slope‖. The direction signal IS present and effective — evidenced by the Up-FM/Down-FM purity LIFT when it is appended (M3). The test verdict is recorded honestly either way per the contract.</div>

<h2>6 · Honest caveats</h2>
<div class="caveat">
<ul style="margin:0;padding-left:18px;line-height:1.6">
<li><b>Cohort / cage stratification — corrected from the handoff.</b> The synthesis prompt asserted all 611 labels are lab_131204 with wild unlabeled (→ vacuous cross-cohort stratification). The actual b619c2bb data spans FOUR cohorts (lab_131204:182, 5970:204, 9252:140, 3452:85). "within-stratum" here is therefore <b>real cross-cohort (cage) stratification</b>, not the within-pairing proxy. Reassuring: M2b's jump win survives within-stratum (0.598), so the wins are shape, not cage leakage.</li>
<li><b>Baseline reproduction warning.</b> Identity jump = 0.415 reproduces the handoff's ~0.41–0.42 exactly, but soft-DTW jump = 0.522, NOT the handoff's "~0.45". The "~0.45" is a STALE point estimate from the original 204-label lab-only set; this is the expanded 611-label 4-cohort set. The canonical SPEC harness <code>eval_shape_human_anchored.py</code> on the SAME data returns IDENTICAL numbers (identity 0.415[.377,.453], soft-DTW 0.522[.480,.570], GATE-1=PROCEED) → harness validated to the digit. The QUALITATIVE incumbent result (soft-DTW ≫ identity on jump, non-overlapping) reproduces strongly.</li>
<li><b>M2a 256-pt support is a spline upsample of 50 knots</b> — no new temporal information beyond the registered contour; scattering needs &gt;50 samples but cannot recover detail the 50-pt downsample discarded.</li>
<li><b>M2b is a SUBSTITUTE, not true JTFS.</b> kymatio 0.3.0 ships no <code>TimeFrequencyScattering1D</code>; M2b uses isotropic Scattering2D (J=3, L=8) on per-call narrowband spectrograms as a fixed time+freq-translation-invariant stand-in. It lacks the separable frequential wavelet and ties the time/freq invariance scales. The "spectrograms are not the blocker" conclusion carries this caveat; it is the exploratory VAE-diagnostic arm, not a production representation.</li>
<li><b>Overall column has no CI.</b> It is an N-weighted point over the four primaries; bootstrap CIs were computed per-family only. Decisions in the table are made per-family on those CIs, never on the overall point.</li>
<li><b>"Tie" ≠ "equal".</b> Overlapping CIs at N=611 mean the benchmark cannot distinguish the methods on that family, not that they are identical. Several point estimates differ (e.g. M2a jump 0.580 vs bar 0.522) but CIs overlap.</li>
</ul>
</div>

<div class="sub" style="margin-top:26px">Sources: <code>results/shape_invariance/{{baselines,m5,m4_rqa,m3_persist,m2a_scatter1d,m2b_jtfs}}_result.json</code> · machine-readable synthesis: <code>results/shape_invariance/synthesis.json</code></div>
</div></body></html>"""

out = os.path.join(RDIR, "shape_invariance_comparison.html")
with open(out, "w") as f:
    f.write(html)

print("=== Phase 5 synthesis params ===")
print(f"N labels={N_LABELS}  primaries={ {f:FAM_N[f] for f in PRIMARY} }")
print(f"cohorts={base['cohort_counts']}")
print(f"settings={SETTINGS}")
print(f"decision rule = non-overlapping 95% bootstrap CI; bar=soft-DTW gamma=1.0; k=10")
print(f"overall random base rate (primary, N-wt) = {overall_base:.4f}")
print("wrote:", out)
print("wrote:", os.path.join(RDIR, "synthesis.json"))
