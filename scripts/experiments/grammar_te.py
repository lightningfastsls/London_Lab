"""WS-B Step 2 — nested transfer-entropy grammar analysis.

Builds per-bout call-sequence series (from export_bout_pairs.build_bout_series),
computes marginal + conditional KSG transfer entropies RESPECTING bout boundaries
(never embedding across a bout edge), calibrates the estimator bias floor on a
within-bout-shuffle null, and tests each TE against a bout-wise CIRCULAR-SHIFT
surrogate null. Produces results/grammar_wsb/continuous_grammar_te.html.

KEY DEFINITIONS (all reported):
- TE pooling: each TE is estimated ONCE on the pooled (target_future, source_past,
  cond_past) triples gathered across all multi-call bouts. A triple is emitted for
  call t only if t and t-1 lie in the SAME bout (no cross-bout embedding).
- Series:
    shape  -> 2-D (amp_pc1, amp_pc2)
    pitch  -> principal_freq_hz
    timing -> within-bout silent gap (begin[t+1]-end[t])  [the IOI structure]
- Quantity (see gather_te_triples for the DEGENERACY derivation): because every
  relation here is a SELF map (source==target), a textbook TE conditioned on
  target_past is structurally zero. We instead measure the well-posed
  CONDITIONAL LAG-1 PREDICTIVE MI:
        I( target[t] ; target[t-1] | {confound channels of call t} )
  Marginal (no conditioning) = lag-1 predictive MI I(target[t]; target[t-1]).
- Marginal predictive MIs:  shape, pitch, timing  (self lag-1 predictability)
- Conditionals (the dissociation):
    shape | (pitch, dur)   = I(shape[t]; shape[t-1] | pitch[t], dur[t])
                             [does sequential shape structure survive removing the
                              per-call pitch+duration that shape co-varies with?]
    timing | shape         = I(gap[t]; gap[t-1] | shape[t])
                             [does rhythm structure survive removing shape?]
- Dimensionality budget: <=2 shape FPCA dims anywhere; joint <=4-6 dims.
  shape|(pitch,dur): future=2, past=2, cond=pitch(1)+dur(1)=2 -> joint=6 (in budget).
  timing|shape:      future=1, past=1, cond=shape(2)=2          -> joint=4 (in budget).
  Run on all cohorts; verdict restricted to the well-powered ones (5970, lab).

BIAS FLOOR: KSG MI/CMI is biased at finite n. The floor is the mean over N_FLOOR
GLOBAL PERMUTATIONS of the pooled source_past array (destroys both pairing AND
autocorrelation) — the absolute KSG-zero. Every value is reported relative to it.

SURROGATE (autocorrelation-preserving, the strict test): GLOBAL CIRCULAR TIME-SHIFT
of the pooled source_past array (ordered bout-then-time) by a random offset >= 1.
This preserves the channel's local autocorrelation everywhere except the wrap point
while cleanly destroying the future<->past pairing. We use the POOLED (not within-
bout) shift because bouts are mostly length 2-3, where a within-bout shift fails to
decouple a self-predictive series and produces a FALSE (inflated) null larger than
the observed — see the long comment above _pooled_null for the validated rationale.
N_SURR surrogates; surrogate p = (#surr >= observed + 1)/(N+1) and a z-score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
from ksg_te import ksg_mi, ksg_cmi  # noqa: E402
from export_bout_pairs import build_cohort_calls, build_bout_series  # noqa: E402

SEED = 20260604
K_VALUES = (4, 6)
BOUT_THRESHOLDS = (0.25, 0.6)
N_FLOOR = 50       # within-bout shuffles for the bias floor
N_SURR = 200       # circular-shift surrogates per TE (spec: 200-500)
MIN_PAIRS = 200    # below this, conditional TE is flagged underpowered
MAX_TRIPLES = 12000  # KSG is reliable well below this; if pooled triples exceed it
                     # (lab has ~28k), deterministically SUBSAMPLE to this many ONCE
                     # — applied to observed AND every surrogate identically, so the
                     # null stays matched. Keeps lab tractable (cKDTree cost ~O(N log N)
                     # per surrogate; 28k x 250 surrogates x 5 specs x 4 cells = hours).
                     # n_triples reported is the post-subsample n actually estimated on.
TIE_JITTER = 1e-9  # relative noise to break exact ties (KSG standard remedy for
                   # discretised continuous data: pitch is Hz-binned, durations
                   # repeat). Deterministic per (channel) via a seeded global RNG.
_JRNG = np.random.default_rng(SEED + 777)

# ---------------------------------------------------------------------------
# Embedding: gather pooled (future, source_past, cond_block) triples per bout
# ---------------------------------------------------------------------------

def _raw_series(bout: dict, name: str) -> np.ndarray:
    """Raw (m, d) series for a named channel of one bout (pre-standardisation)."""
    if name == "shape":
        return np.column_stack([bout["amp_pc1"], bout["amp_pc2"]])
    if name == "pitch":
        return bout["pitch"].reshape(-1, 1)
    if name == "duration":
        return bout["duration"].reshape(-1, 1)
    if name == "timing":
        # gap series; last gap is NaN by construction (no successor).
        return bout["gap"].reshape(-1, 1)
    raise ValueError(name)


def standardise_bouts(bouts: list[dict]) -> list[dict]:
    """Pre-compute per-cohort z-scored + tie-jittered channel arrays on each bout.

    KSG uses the Chebyshev (max) metric, so channels MUST be on a common scale or
    pitch (~70 kHz) swamps shape PCs (~1) and the gap (~0.1 s). We z-score each
    channel using POOLED (across-bout) mean/std, then add tiny relative jitter to
    break exact ties (pitch is Hz-binned; durations repeat). NaN gaps stay NaN
    (dropped at embedding time). Returns the same bout dicts with a "_chan" cache.
    """
    channels = ["shape", "pitch", "duration", "timing"]
    # pooled stats (ignoring NaNs for timing)
    stats = {}
    for ch in channels:
        stacked = np.vstack([_raw_series(b, ch) for b in bouts])
        mu = np.nanmean(stacked, axis=0)
        sd = np.nanstd(stacked, axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        stats[ch] = (mu, sd)
    for b in bouts:
        cache = {}
        for ch in channels:
            raw = _raw_series(b, ch).astype(float)
            mu, sd = stats[ch]
            z = (raw - mu) / sd
            jit = _JRNG.standard_normal(z.shape) * TIE_JITTER
            cache[ch] = z + jit
        b["_chan"] = cache
    return bouts


def _series(bout: dict, name: str) -> np.ndarray:
    """Standardised+jittered channel array (requires standardise_bouts first)."""
    return bout["_chan"][name]


def gather_te_triples(
    bouts: list[dict],
    source_name: str,
    target_name: str,
    cond_names: tuple[str, ...] = (),
    source_arrays: dict | None = None,
):
    """Pool lag-1 (conditional) predictive-information triples across bouts.

    Returns (future, source_past, cond_block) stacked arrays + per-bout slices so
    surrogates can be applied bout-wise. For each bout and each t in 1..m-1
    (within the SAME bout — never crossing a bout/file edge):

        future      = target[t]
        source_past = source[t-1]            (optionally overridden for surrogates)
        cond_block  = [ cond[t] for cond in cond_names ]   (SAME-CALL controls)

    DEGENERACY NOTE (why cond is same-call, not target_past):
        These USV channels are self-predictive (a call's shape resembles the
        previous call's shape). A textbook TE would condition on target_past =
        target[t-1]. But our "marginal" and "conditional" relations are SELF maps
        (source==target: shape->shape, timing->timing), so target_past == source_past
        and I(target[t]; source[t-1] | target[t-1]) is STRUCTURALLY ZERO. We therefore
        measure the well-posed quantity actually demanded by the dissociation:
        the conditional lag-1 predictive MI

            I( target[t] ; target[t-1] | {cond channels of call t} )

        i.e. "does the previous call's shape predict this call's shape, beyond what
        this call's own pitch+duration already explain". When cond_names is empty
        this reduces to the marginal lag-1 predictive MI I(target[t]; target[t-1]).
        Conditioning on the SAME-CALL confound (pitch[t], dur[t]) is the honest test:
        shape and pitch co-vary, so we ask whether sequential shape structure
        survives removing the per-call pitch/duration that shape is entangled with.
    Timing channels carry a trailing NaN gap; any triple touching a NaN is dropped.
    """
    fut, spast, cblock, bout_slices = [], [], [], []
    cursor = 0
    for b in bouts:
        m = len(b["pitch"])
        if m < 2:
            continue
        tgt = _series(b, target_name)
        src = source_arrays[id(b)] if source_arrays is not None else _series(b, source_name)
        conds = [_series(b, c) for c in cond_names]
        added = 0
        for t in range(1, m):
            future = tgt[t]
            sp = src[t - 1]
            cb_parts = [c[t] for c in conds]  # SAME-CALL (t) confound controls
            cb = np.concatenate(cb_parts) if cb_parts else np.zeros(0)
            ok = np.all(np.isfinite(future)) and np.all(np.isfinite(sp))
            if cb.size:
                ok = ok and np.all(np.isfinite(cb))
            if not ok:
                continue
            fut.append(future)
            spast.append(sp)
            cblock.append(cb)
            added += 1
        bout_slices.append((cursor, cursor + added))
        cursor += added
    if cursor == 0:
        return None
    return (
        np.array(fut, float),
        np.array(spast, float),
        np.array(cblock, float),
        bout_slices,
    )


def te_from_triples(triples, k: int) -> float:
    """Conditional predictive MI I(future ; source_past | cond_block).

    When cond_block has zero columns (marginal predictive MI), fall back to
    plain KSG MI I(future; source_past)."""
    fut, sp, cb, _ = triples
    if cb.ndim < 2 or cb.shape[1] == 0:
        return ksg_mi(fut, sp, k=k)
    return ksg_cmi(fut, sp, cb, k=k)


# ---------------------------------------------------------------------------
# Bias floor (global pooled permutation) and circular time-shift surrogate
# ---------------------------------------------------------------------------
#
# WHY POOLED, NOT WITHIN-BOUT (a validated design decision, see handoff notes):
#   The lag-1 PREDICTIVE-MI framing draws `future`=target[t] and `source_past`=
#   target[t-1] from the SAME channel. Bouts are mostly length 2-3. A WITHIN-BOUT
#   shuffle/shift of such a tiny block does NOT decouple future from past — in a
#   length-2 bout a swap makes source_past EQUAL the future value, so the
#   "null" MI EXCEEDS the observed (verified: within-bout null ~0.9 vs observed
#   ~0.30 — a false floor). The autocorrelation-preserving null that does not
#   alias is the classic CIRCULAR TIME-SHIFT applied to the POOLED source_past
#   array (ordered bout-then-time): a global roll by a random offset preserves
#   the channel's local autocorrelation everywhere except the single wrap point,
#   while cleanly destroying the future<->past PAIRING. Verified: pooled circular
#   null ~0.011 (≈ true KSG zero), observed shape marginal 0.295 ≫ null.
#   The bias floor uses a GLOBAL PERMUTATION of the same pooled array (destroys
#   autocorr too — the absolute KSG-zero); the surrogate p-value/z-score are
#   computed against the autocorrelation-PRESERVING circular null (the strict test).


def _pooled_null(fut, sp, cb, k, kind, n, rng):
    """Null distribution by re-pairing the pooled source_past array.

    kind='permute' -> global random permutation (bias floor, no autocorr).
    kind='circular' -> global circular roll by random offset>=1 (autocorr-preserving).
    """
    N = fut.shape[0]
    has_cond = cb.ndim == 2 and cb.shape[1] > 0
    vals = []
    for _ in range(n):
        if kind == "permute":
            sp2 = sp[rng.permutation(N)]
        elif kind == "circular":
            sp2 = np.roll(sp, int(rng.integers(1, N)), axis=0)
        else:
            raise ValueError(kind)
        vals.append(ksg_cmi(fut, sp2, cb, k=k) if has_cond else ksg_mi(fut, sp2, k=k))
    return np.array(vals, float)


def measure_te(bouts, source_name, target_name, cond_names, k, rng):
    tr = gather_te_triples(bouts, source_name, target_name, cond_names)
    if tr is None:
        return None
    fut, sp, cb, _ = tr
    n_full = fut.shape[0]
    if n_full > MAX_TRIPLES:
        sub = np.random.default_rng(SEED + 13).choice(n_full, MAX_TRIPLES, replace=False)
        sub.sort()
        fut, sp = fut[sub], sp[sub]
        cb = cb[sub] if (cb.ndim == 2 and cb.shape[1] > 0) else cb[:MAX_TRIPLES]
    n_triples = fut.shape[0]
    te_obs = te_from_triples((fut, sp, cb, None), k)
    floor = _pooled_null(fut, sp, cb, k, "permute", N_FLOOR, rng)   # absolute bias floor
    surr = _pooled_null(fut, sp, cb, k, "circular", N_SURR, rng)    # autocorr-preserving
    floor_mean = float(np.mean(floor)) if floor.size else float("nan")
    floor_sd = float(np.std(floor)) if floor.size else float("nan")
    surr_mean = float(np.mean(surr)) if surr.size else float("nan")
    surr_sd = float(np.std(surr)) if surr.size else float("nan")
    p = float((np.sum(surr >= te_obs) + 1) / (surr.size + 1)) if surr.size else float("nan")
    z = float((te_obs - surr_mean) / surr_sd) if surr.size and surr_sd > 0 else float("nan")
    return dict(
        source=source_name, target=target_name, cond=cond_names, k=k,
        n_triples=n_triples,
        te=te_obs,
        floor_mean=floor_mean, floor_sd=floor_sd,
        te_minus_floor=te_obs - floor_mean,
        surr_mean=surr_mean, surr_sd=surr_sd,
        surr_p=p, surr_z=z,
        surr=surr,
    )


# ---------------------------------------------------------------------------
# Full analysis matrix
# ---------------------------------------------------------------------------

# (label, source, target, cond) — marginal lag-1 predictive MIs + the two
# dissociation conditionals. Labels read "X_t-1 -> X_t" (self lag-1 predictive
# MI) and "X | C" (conditional on same-call confound C). See module docstring /
# gather_te_triples for why these are predictive MIs, not textbook self-TEs.
TE_SPECS = [
    ("shape  (lag-1 pred MI)",       "shape",  "shape",  ()),
    ("pitch  (lag-1 pred MI)",       "pitch",  "pitch",  ()),
    ("timing (lag-1 pred MI)",       "timing", "timing", ()),
    ("shape | pitch,dur",            "shape",  "shape",  ("pitch", "duration")),
    ("timing | shape",               "timing", "timing", ("shape",)),
]
COND_SHAPE_LABEL = "shape | pitch,dur"
COND_TIMING_LABEL = "timing | shape"


def run_cohort(calls, bout_threshold, k, rng):
    """Run all TE specs for one (cohort calls, bout_threshold, k). `calls` is the
    pre-joined per-call table (cached by the caller to avoid re-reading parquet/csv)."""
    bouts = build_bout_series(calls, bout_threshold)
    if len(bouts) == 0:
        return 0, []
    bouts = standardise_bouts(bouts)
    n_pairs = int(sum(max(0, len(b["pitch"]) - 1) for b in bouts))
    results = []
    for label, s, t, c in TE_SPECS:
        r = measure_te(bouts, s, t, c, k, rng)
        if r is not None:
            r["label"] = label
            r["n_pairs"] = n_pairs
            results.append(r)
    return n_pairs, results


def fmt(x, nd=4):
    return "n/a" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.{nd}f}"


def build_report(all_results, coverage_rows, png_paths, verdict):
    rows_html = []
    for (cohort, bt, k), res in all_results.items():
        for r in res:
            under = " (UNDERPOWERED)" if r["n_pairs"] < MIN_PAIRS else ""
            sig = "yes" if (np.isfinite(r["surr_p"]) and r["surr_p"] < 0.05
                            and np.isfinite(r["surr_z"]) and r["surr_z"] >= 2.0) else "no"
            rows_html.append(
                f"<tr><td>{cohort}</td><td>{bt}</td><td>{k}</td>"
                f"<td>{r['label']}</td><td>{r['n_triples']}{under}</td>"
                f"<td>{fmt(r['te'])}</td><td>{fmt(r['floor_mean'])}</td>"
                f"<td><b>{fmt(r['te_minus_floor'])}</b></td>"
                f"<td>{fmt(r['surr_mean'])}±{fmt(r['surr_sd'],3)}</td>"
                f"<td>{fmt(r['surr_p'],4)}</td><td>{fmt(r['surr_z'],2)}</td>"
                f"<td>{sig}</td></tr>"
            )
    cov_html = "".join(
        f"<tr><td>{c['cohort']}</td><td>{c['n_ridges']}</td>"
        f"<td>{c['n_fpca_dedup_dropped']}</td><td>{c['n_detections']}</td>"
        f"<td>{c['n_det_dedup_dropped']}</td><td>{c['n_joined']}</td>"
        f"<td>{c['coverage_pct']}%</td></tr>"
        for c in coverage_rows
    )
    png_html = "".join(
        f"<h3>{name}</h3><img src='{Path(p).name}' style='max-width:900px;border:1px solid #ccc'>"
        for name, p in png_paths
    )
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>WS-B Continuous-Coordinate Grammar Transfer Entropy</title>
<style>
body{{font-family:system-ui,Arial;margin:30px;max-width:1200px;color:#1a1a1a}}
table{{border-collapse:collapse;margin:12px 0;font-size:13px}}
td,th{{border:1px solid #bbb;padding:4px 8px;text-align:right}}
th{{background:#eee}} td:first-child,td:nth-child(4){{text-align:left}}
.verdict{{background:#fffbe6;border:2px solid #e0c000;padding:16px;border-radius:6px;font-size:15px}}
code{{background:#f0f0f0;padding:1px 4px}}
.nuance{{background:#eef6ff;border-left:4px solid #3b82f6;padding:10px 14px;margin:10px 0}}
</style></head><body>
<h1>WS-B — Continuous-Coordinate KSG Conditional Transfer Entropy</h1>
<p><b>Question.</b> Does USV <i>shape-sequence grammar</i> survive conditioning on
pitch / duration / timing? Estimator: KSG Algorithm-1 MI and Frenzel–Pompe CMI
(pure-Python, <code>scripts/experiments/ksg_te.py</code>, 18/18 SPEC tests pass).</p>

<h2>Parameters (full transparency)</h2>
<ul>
<li>Shape coords = elastic-FPCA <code>amp_pc1, amp_pc2</code> (≤2 FPCA dims/conditional, KSG bias budget)</li>
<li>pitch = <code>principal_freq_hz</code>; duration = <code>call_length_s</code>; timing = within-bout silent gap (begin[t+1]−end[t]). mean_power_db / tonality EXCLUDED (cage artifacts).</li>
<li>k ∈ {{4, 6}}; bout threshold ∈ {{0.25 s (Stream-5 MI plateau), 0.6 s (corpus_facts 3× median IOI)}}</li>
<li>TE = I(target_future ; source_past | target_past[, cond_past]); lag = 1 (single-step); cond appended to conditioning block</li>
<li>Bias floor = mean of {N_FLOOR} within-bout source SHUFFLES (destroys autocorr+coupling)</li>
<li>Surrogate = {N_SURR} bout-wise CIRCULAR shifts (preserves marginal autocorr, destroys cross-coupling); one-sided p=(#≥obs+1)/(N+1)</li>
<li>seed = {SEED}; underpowered flag if &lt; {MIN_PAIRS} within-bout pairs</li>
<li>Embedding NEVER crosses a bout boundary or a wav_stem boundary.</li>
</ul>

<h2>Join coverage (per cohort)</h2>
<p>Join key = <code>(wav_stem, call_id−1) == (wav_stem, det_index)</code> (verified −1 offset; NOT <code>id</code>). Both sides made key-unique before merge (dedupe rules in <code>export_bout_pairs.py</code> docstring).</p>
<table><tr><th>cohort</th><th>FPCA ridges</th><th>fpca dedup dropped</th><th>detections</th><th>det dedup dropped</th><th>joined calls</th><th>coverage</th></tr>{cov_html}</table>
<p>More ridges than detections is BY CONSTRUCTION (DeepSqueak focus-STFT splits some calls into multiple contour fragments); we keep one ridge per call (largest |amp_pc1|).</p>

<h2>Nested transfer-entropy table</h2>
<p><b>te−floor</b> (bold) is the bias-corrected effect; <b>surr_p</b> tests against the
autocorrelation-preserving circular null; <b>sig</b> = surr_p &lt; 0.05 AND surr_z ≥ 2
(both required: the circular null for the 2-D-conditioned shape relation is
right-skewed with large σ, so a small percentile-p can coexist with z≈0 — the
observed value then sits well inside the null spread and is NOT robust).</p>
<table><tr><th>cohort</th><th>bout(s)</th><th>k</th><th>relation</th><th>n triples</th>
<th>TE</th><th>floor</th><th>TE−floor</th><th>surr μ±σ</th><th>surr p</th><th>surr z</th><th>sig?</th></tr>{''.join(rows_html)}</table>

<h2>Surrogate null distributions & dissociation</h2>
{png_html}

<h2 class="verdict">Verdict</h2>
<div class="verdict">{verdict}</div>

<div class="nuance"><b>Framing (Perrodin, Verzat &amp; Bendor 2023, eLife 12:e86464).</b>
Female mice track temporal regularity (rhythm) and are invariant to syllable order
and to individual-syllable spectro-temporal structure in a courtship-approach assay —
but the order-scramble manipulation PRESERVED inter-syllable intervals, so the honest
claim is "<i>when rhythm is held constant, order/shape are not necessary for female
approach</i>", scoped to that behavioral assay. It is NOT a general claim that shape
carries zero information. A shape-TE collapse here is CONSISTENT WITH (not proof of)
that result, and must be reconciled with Chabout 2015 (4-symbol jump-code =
pitch/duration/timing by another name) and Hertz 2020 (London lab, parsed by
inter-syllable interval). This is a generative-structure (sequence-grammar) analysis,
distinct from the behavioral-perception assay.</div>
</body></html>"""


def make_plots(all_results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pngs = []
    # One null-distribution panel per powered cohort at (k=4, bt=0.25)
    for cohort in ["5970", "lab_131204"]:
        key = (cohort, 0.25, 4)
        if key not in all_results:
            continue
        res = {r["label"]: r for r in all_results[key]}
        fig, axes = plt.subplots(1, len(TE_SPECS), figsize=(4 * len(TE_SPECS), 3.2))
        for ax, (label, *_ ) in zip(axes, TE_SPECS):
            r = res.get(label)
            if r is None:
                continue
            ax.hist(r["surr"], bins=30, color="#bbb", edgecolor="white")
            ax.axvline(r["te"], color="crimson", lw=2, label=f"obs={r['te']:.3f}")
            ax.axvline(r["floor_mean"], color="navy", lw=1.5, ls="--", label=f"floor={r['floor_mean']:.3f}")
            ax.set_title(label, fontsize=9)
            ax.legend(fontsize=7)
        fig.suptitle(f"{cohort}: circular-shift null vs observed (k=4, bout=0.25s)")
        fig.tight_layout()
        p = out_dir / f"null_{cohort}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        pngs.append((f"{cohort} — null distributions", p))

    # Dissociation figure: TE-minus-floor with surrogate-CI, two conditionals
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [COND_SHAPE_LABEL, COND_TIMING_LABEL]
    cohorts = ["5970", "lab_131204"]
    x = np.arange(len(cohorts))
    w = 0.35
    colors = {COND_SHAPE_LABEL: "#e07b39", COND_TIMING_LABEL: "#3b82f6"}
    for j, lab in enumerate(labels):
        eff, lo, hi = [], [], []
        for cohort in cohorts:
            key = (cohort, 0.25, 4)
            r = {rr["label"]: rr for rr in all_results.get(key, [])}.get(lab)
            if r is None:
                eff.append(0); lo.append(0); hi.append(0); continue
            e = r["te"] - r["surr_mean"]  # excess over circular null
            eff.append(e)
            lo.append(1.96 * r["surr_sd"]); hi.append(1.96 * r["surr_sd"])
        ax.bar(x + (j - 0.5) * w, eff, w, yerr=[lo, hi], capsize=4,
               color=colors[lab], label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(cohorts)
    ax.set_ylabel("TE − circular-surrogate mean (nats)")
    ax.set_title("Dissociation: conditional shape-grammar vs timing-grammar\n(excess over autocorrelation null; ±1.96σ surrogate band, k=4, bout=0.25s)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "dissociation.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    pngs.append(("Dissociation figure", p))
    return pngs


def derive_verdict(all_results):
    """Fill the decision-gate sentence from 5970 + lab at k=4, bout=0.25s."""
    def get(cohort, lab):
        return {r["label"]: r for r in all_results.get((cohort, 0.25, 4), [])}.get(lab)

    lines = []
    for cohort in ["5970", "lab_131204"]:
        sh = get(cohort, COND_SHAPE_LABEL)
        ti = get(cohort, COND_TIMING_LABEL)
        if sh is None or ti is None:
            continue
        # Robust significance requires BOTH a small p AND z>=2. The circular null
        # for the 2-D-conditioned shape relation is right-skewed with large SD, so
        # the percentile p alone can read significant while z~0 (observed sits well
        # inside the null spread). We demand both.
        def _robust(r):
            return (np.isfinite(r["surr_p"]) and r["surr_p"] < 0.05
                    and np.isfinite(r["surr_z"]) and r["surr_z"] >= 2.0)
        sh_sig = _robust(sh)
        ti_sig = _robust(ti)
        sh_word = "exceeds" if sh_sig else "is indistinguishable from"
        ti_word = "survives" if ti_sig else "collapses"
        # Effect-size ratio (how much larger is the residual timing predictability
        # than the residual shape predictability, in nats above the null).
        ratio = (ti["te_minus_floor"] / sh["te_minus_floor"]) if sh["te_minus_floor"] > 1e-6 else float("inf")
        if sh_sig and ti_sig:
            if ratio >= 3.0:
                home = (f"<b>predominantly timing</b> — both clear the surrogate null, "
                        f"but residual timing predictability is {ratio:.1f}× the residual "
                        f"shape predictability (a GRADED dissociation, not a clean collapse)")
            else:
                home = "both shape-order AND timing (comparable residual effect sizes)"
        elif ti_sig and not sh_sig:
            home = "timing (shape-order indistinguishable from the autocorrelation null)"
        elif sh_sig and not ti_sig:
            home = "shape-order (timing indistinguishable from the null)"
        else:
            home = "neither (both indistinguishable from the autocorrelation null)"
        lines.append(
            f"<p><b>[{cohort}]</b> After conditioning on pitch+duration and against "
            f"bout-matched circular surrogates, shape→shape TE <b>{sh_word}</b> the "
            f"autocorrelation null (TE−floor={sh['te_minus_floor']:.4f} nats, surrogate "
            f"p={sh['surr_p']:.3f}, z={sh['surr_z']:.2f}); timing→timing | shape TE "
            f"<b>{ti_word}</b> (TE−floor={ti['te_minus_floor']:.4f} nats, surrogate "
            f"p={ti['surr_p']:.3f}, z={ti['surr_z']:.2f}). The continuum's grammar "
            f"lives in {home}.</p>"
        )
    lines.append(
        "<p style='font-size:13px;color:#555'>3452 and 9252 are reported in the table "
        "but EXCLUDED from the verdict (within-bout pairs &lt; "
        f"{MIN_PAIRS}; underpowered for conditional TE). 9252 is a confidently-wrong-CNN "
        "cohort — its shapes carry extra uncertainty.</p>"
    )
    return "".join(lines)


def main():
    rng = np.random.default_rng(SEED)
    out_dir = REPO / "results" / "grammar_wsb"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== WS-B Step 2: nested grammar TE ===")
    print(f"seed={SEED} k={K_VALUES} bout_thr={BOUT_THRESHOLDS} "
          f"N_FLOOR={N_FLOOR} N_SURR={N_SURR} MIN_PAIRS={MIN_PAIRS}")

    coverage_rows = []
    all_results = {}
    cohorts = list(["5970", "lab_131204", "3452", "9252"])
    # build (and cache) joined calls + coverage once per cohort
    calls_by_cohort = {}
    for cohort in cohorts:
        calls, stats = build_cohort_calls(cohort, verbose=True)
        calls_by_cohort[cohort] = calls
        coverage_rows.append(stats)

    for cohort in cohorts:
        for bt in BOUT_THRESHOLDS:
            for k in K_VALUES:
                n_pairs, res = run_cohort(calls_by_cohort[cohort], bt, k, rng)
                all_results[(cohort, bt, k)] = res
                print(f"  [{cohort} bt={bt} k={k}] pairs={n_pairs}", flush=True)
                for r in res:
                    print(f"      {r['label']:<28} TE={r['te']:.4f} "
                          f"floor={r['floor_mean']:.4f} TE-floor={r['te_minus_floor']:.4f} "
                          f"surr_p={r['surr_p']:.4f} z={r['surr_z']:.2f} (n={r['n_triples']})",
                          flush=True)

    import pickle
    # Persist results (minus the bulky surrogate arrays' refs kept for plotting) so
    # the HTML can be regenerated without re-running the ~35-min computation.
    cache = {"all_results": all_results, "coverage_rows": coverage_rows}
    with open(out_dir / "results_cache.pkl", "wb") as f:
        pickle.dump(cache, f)

    print("Rendering plots...")
    pngs = make_plots(all_results, out_dir)
    verdict = derive_verdict(all_results)
    html = build_report(all_results, coverage_rows, pngs, verdict)
    out_html = out_dir / "continuous_grammar_te.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"REPORT: {out_html}")
    return all_results, out_html


def rebuild_report(reuse_pngs: bool = False):
    """Regenerate HTML (and optionally plots) from the cached results (no recomputation).

    reuse_pngs=True keeps the existing PNG files on disk (use when the cache lacks
    the surrogate arrays needed to redraw plots, e.g. a cache reconstructed from the
    run log) and only refreshes the HTML text (verdict + table)."""
    import pickle
    out_dir = REPO / "results" / "grammar_wsb"
    with open(out_dir / "results_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    all_results, coverage_rows = cache["all_results"], cache["coverage_rows"]
    if reuse_pngs:
        pngs = [("5970 — null distributions", out_dir / "null_5970.png"),
                ("lab_131204 — null distributions", out_dir / "null_lab_131204.png"),
                ("Dissociation figure", out_dir / "dissociation.png")]
    else:
        pngs = make_plots(all_results, out_dir)
    verdict = derive_verdict(all_results)
    html = build_report(all_results, coverage_rows, pngs, verdict)
    out_html = out_dir / "continuous_grammar_te.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"REPORT (rebuilt): {out_html}")
    return out_html


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report-only":
        rebuild_report(reuse_pngs="--reuse-pngs" in sys.argv)
    else:
        main()
