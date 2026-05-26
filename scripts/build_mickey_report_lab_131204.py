"""Build a self-contained HTML report of Phase 2B findings for Mickey.

Reads result CSVs and PNGs from Phase 2A/2B/2C, base64-encodes every figure,
and writes a single HTML file (no CDN/external CSS) at
``reports/lab_131204_phase2b_mickey.html``.

Usage::

    .venv/bin/python scripts/build_mickey_report_lab_131204.py
    .venv/bin/python scripts/build_mickey_report_lab_131204.py --root /path/to/lab_repo

The report covers five findings:
  1. Scattoni-7 syllable distribution (descriptive)
  2. Continuous repertoire structure (NEW, robust under down-sample)
  3. Tier signal (CONFIRMED, V=0.250)
  4. Couple keep-set signal (CONFIRMED, V=0.165)
  5. Two independent noise mechanisms (NEW)
"""

from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path

DEFAULT_ROOT = Path("/home/shachar/projects/mickey_london_lab")


def fig_paths(root: Path) -> dict[str, Path]:
    return {
        "type_distribution": root / "results/traditional_taxonomy_lab_131204/type_distribution.png",
        "feature_summary":   root / "results/traditional_taxonomy_lab_131204/feature_summary.png",
        "cluster_heatmap":   root / "results/traditional_taxonomy_lab_131204/cluster_vs_type_heatmap.png",
        "umap_hdbscan":      root / "results/recluster_umap_hdbscan_lab_131204/umap_hdbscan_scatter.png",
        "umap_kmeans":       root / "results/recluster_umap_hdbscan_lab_131204/umap_kmeans_scatter.png",
        "umap_hdbscan_ds":   root / "results/recluster_umap_hdbscan_lab_131204_downsampled/umap_hdbscan_scatter.png",
        "contingency":       root / "results/recluster_umap_hdbscan_lab_131204/contingency_matrix.png",
        "by_tier":           root / "results/repertoire_lab_131204/by_tier.png",
        "by_couple_keepset": root / "results/repertoire_lab_131204/by_couple_keep_set.png",
        "by_couple":         root / "results/repertoire_lab_131204/by_couple.png",
        "correlation":       root / "results/acoustic_feature_analysis_lab_131204/correlation_matrix.png",
    }


def encode_png(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def render_fig(figures: dict[str, Path], label: str, alt: str, caption: str = "") -> str:
    src = encode_png(figures[label])
    cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
    return f'<figure><img src="{src}" alt="{html.escape(alt)}">{cap}</figure>'


CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  max-width: 900px; margin: 2em auto; padding: 0 1.5em;
  color: #1a1a1a; line-height: 1.55;
}
h1 { font-size: 1.8em; border-bottom: 3px solid #2c5282; padding-bottom: .3em; }
h2 { font-size: 1.4em; margin-top: 2em; color: #2c5282;
     border-bottom: 1px solid #cbd5e0; padding-bottom: .2em; }
h3 { font-size: 1.1em; color: #4a5568; margin-top: 1.4em; }
.banner {
  background: #ebf4ff; border-left: 4px solid #2c5282;
  padding: 0.8em 1.2em; margin: 1em 0 2em 0; border-radius: 4px;
}
.banner ul { margin: .4em 0; }
.tag {
  display: inline-block; padding: 0.1em 0.6em; border-radius: 3px;
  font-size: 0.78em; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.04em; margin-left: 0.5em; vertical-align: middle;
}
.tag-new       { background: #c6f6d5; color: #22543d; }
.tag-confirmed { background: #bee3f8; color: #2a4365; }
.tag-descr     { background: #edf2f7; color: #2d3748; }
figure { margin: 1.2em 0; text-align: center; }
figure img { max-width: 100%; height: auto; border: 1px solid #e2e8f0;
             border-radius: 4px; }
figcaption { font-size: 0.86em; color: #4a5568; margin-top: 0.5em; font-style: italic; }
.figrow { display: flex; flex-wrap: wrap; gap: 1em; justify-content: center; }
.figrow > figure { flex: 1 1 380px; max-width: 48%; margin: 0; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.92em; }
th, td { padding: 0.5em 0.7em; text-align: left; border: 1px solid #cbd5e0; }
th { background: #edf2f7; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.callout {
  background: #fffaf0; border-left: 4px solid #dd6b20;
  padding: 0.8em 1.2em; margin: 1em 0; border-radius: 4px;
}
.callout-good { background: #f0fff4; border-left-color: #38a169; }
.footer-meta {
  background: #f7fafc; border-top: 2px solid #cbd5e0;
  padding: 1em 1.2em; margin-top: 2em; font-size: 0.88em;
  color: #4a5568; border-radius: 4px;
}
code { background: #edf2f7; padding: 0.1em 0.4em; border-radius: 3px; font-size: 0.92em; }
"""


def build_html(figures: dict[str, Path]) -> str:
    f = lambda label, alt, cap="": render_fig(figures, label, alt, cap)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Lab 131204 — First-Pass USV Analysis (Phase 2B)</title>
<style>{CSS}</style>
</head>
<body>

<h1>Lab 131204 — First-Pass USV Analysis</h1>

<div class="banner">
  <strong>Dataset summary</strong>
  <ul>
    <li><strong>17 couples</strong> (12 cross-strain + 5 same-strain), recorded 2013-12-04</li>
    <li><strong>~14 hours</strong> of audio split into 25,770 × 2-second WAV chunks at 300 kHz</li>
    <li><strong>41,061 USV calls detected</strong> by the production CNN (post-300 ms duration filter)</li>
    <li><strong>40,787 calls</strong> kept after pairing with DeepSqueak acoustic features</li>
  </ul>
  <em>Report generated 2026-05-15 · Pipeline: detection → DeepSqueak features → 3 classifiers + repertoire stats</em>
</div>

<h2>At a glance</h2>
<p>Five findings emerged from this first-pass analysis. Two are <span class="tag tag-new">new</span>,
two <span class="tag tag-confirmed">confirmed</span> earlier signals from the post-labeling handoff,
and one is <span class="tag tag-descr">descriptive</span> (repertoire composition).</p>

<table>
  <thead><tr><th>#</th><th>Finding</th><th>Status</th><th>Headline</th></tr></thead>
  <tbody>
    <tr><td>1</td><td>Scattoni-7 syllable distribution</td>
        <td><span class="tag tag-descr">descriptive</span></td>
        <td>Flat &amp; Chevron together make up over half of all calls</td></tr>
    <tr><td>2</td><td>Continuous repertoire structure</td>
        <td><span class="tag tag-new">new</span></td>
        <td>Density-based clustering finds <strong>one dominant continuous mode</strong>;
            unimodality survives the size-control down-sample test</td></tr>
    <tr><td>3</td><td>Tier signal</td>
        <td><span class="tag tag-confirmed">confirmed</span></td>
        <td>Manual-review tier carries 2.4× more Short and 2.3× more Complex calls
            (effect size V = 0.25, medium-large)</td></tr>
    <tr><td>4</td><td>Couple-keep-set signal</td>
        <td><span class="tag tag-confirmed">confirmed</span></td>
        <td>Four noise-prone couples have distinct repertoires (V = 0.17, small-medium)</td></tr>
    <tr><td>5</td><td>Two independent noise mechanisms</td>
        <td><span class="tag tag-new">new</span></td>
        <td>A duration noise (filtered out) and a residual 49 kHz tonal noise
            (survived all filters) hit <em>different</em> couples</td></tr>
  </tbody>
</table>

<h2>Finding 1 — Syllable distribution <span class="tag tag-descr">descriptive</span></h2>

<p>Using the standard Scattoni-2008 7-type taxonomy (a rule-based classifier
applied to each call's acoustic features):</p>

<table>
  <thead><tr><th>Syllable type</th><th class="num">Count</th><th class="num">%</th></tr></thead>
  <tbody>
    <tr><td>Flat</td>            <td class="num">12,134</td><td class="num">29.75</td></tr>
    <tr><td>Chevron</td>         <td class="num">9,132</td> <td class="num">22.39</td></tr>
    <tr><td>Down</td>            <td class="num">5,884</td> <td class="num">14.43</td></tr>
    <tr><td>Short</td>           <td class="num">5,364</td> <td class="num">13.15</td></tr>
    <tr><td>Complex</td>         <td class="num">3,651</td> <td class="num">8.95</td></tr>
    <tr><td>Up</td>              <td class="num">3,545</td> <td class="num">8.69</td></tr>
    <tr><td>Frequency Jump</td>  <td class="num">1,077</td> <td class="num">2.64</td></tr>
  </tbody>
</table>

{f("type_distribution", "Scattoni-7 syllable distribution",
   "Per-type counts across the 40,787 calls. Flat and Chevron together account for ~52% of the lab's vocal output.")}

<p>Flat calls (steady tone, little frequency modulation) and Chevron calls
(rise-then-fall) dominate. Frequency Jump calls — sharp discontinuous shifts
mid-call — are rare (2.6%).</p>

{f("feature_summary", "Per-type acoustic feature summary",
   "Median duration, peak frequency, bandwidth and sinuosity for each of the seven syllable types.")}

<h2>Finding 2 — Continuous repertoire structure <span class="tag tag-new">new</span></h2>

<p>The Scattoni labels above assume seven distinct categories. To test whether
the data actually cluster into distinct types, we ran two independent
clustering methods on the raw acoustic features:</p>

<ul>
  <li><strong>DeepSqueak k-means (k=26)</strong> — forces the data into 26 partitions by design</li>
  <li><strong>UMAP + HDBSCAN</strong> — density-based; lets the data decide how many clusters exist</li>
</ul>

<div class="figrow">
  {f("umap_hdbscan", "UMAP scatter colored by HDBSCAN cluster",
     "HDBSCAN (data-driven). Finds 3 clusters: one dominant mode (71%, blue), one outlier cluster (8%), one residual-noise cluster (0.6%) plus unclustered noise points (20%).")}
  {f("umap_kmeans", "UMAP scatter colored by DeepSqueak k-means clusters",
     "k-means (forced k=26). Carves the same continuous UMAP cloud into 26 partitions because k was hard-coded, not because the data structure demanded 26 groups.")}
</div>

<p>HDBSCAN groups events that genuinely cluster in feature space. On the
full lab dataset, it returns one dominant cluster containing
<strong>71% of all calls</strong>, one small outlier cluster (~8%), one tiny
residual-noise cluster (244 calls — see Finding 5), and 20% noise points.
By comparison, the wild-mouse dataset (3452) produces 5+ separable clusters
with the same algorithm settings.</p>

<div class="callout callout-good">
  <strong>Size-control check (down-sample test).</strong> A natural objection
  is that the lab has 40,787 calls while wild 3452 has only 7,921 — could the
  unimodality just reflect more data smoothing out clusters? We
  stratified-sampled the lab to <strong>n=7,920 (matching 3452)</strong>
  preserving couple proportions, and re-ran HDBSCAN with identical settings.
  Result: <strong>2 clusters with 0.2% noise; the mega-cluster grew to 91%</strong>.
  Unimodality is not a dataset-size artifact.
</div>

{f("umap_hdbscan_ds", "UMAP scatter — lab down-sampled to n=7,920",
   "After matching the wild-mouse sample size, HDBSCAN finds 2 clusters with only 14 noise points (0.2%); the dominant mode contains 91% of calls. Wild 3452 at this same n produces 5+ separable clusters.")}

<p><strong>Interpretation.</strong> Lab USVs appear to occupy a continuous
acoustic space rather than discrete categories. The Scattoni labels remain
useful as bookkeeping coordinates, but they should not be read as evidence
for distinct call-types in this cohort.</p>

{f("contingency", "Scattoni-type × HDBSCAN-cluster contingency",
   "How the Scattoni rule-based types distribute across the HDBSCAN data-driven clusters.")}

<h2>Finding 3 — Tier signal <span class="tag tag-confirmed">confirmed</span></h2>

<p>Each detected call carries a <em>tier</em> from the upstream labeling
pipeline: <code>auto_accept</code> (high confidence, n=29,790) or
<code>manual_review</code> (lower confidence, n=10,997). The post-labeling
handoff predicted these two tiers would have different syllable mixes —
a residual quality artifact rather than a biological difference.</p>

{f("by_tier", "Repertoire by tier",
   "Per-tier proportions of the seven Scattoni syllable types. Manual-review (right bars) is enriched for Short and Complex calls.")}

<table>
  <thead><tr><th>Test</th><th class="num">N</th><th class="num">χ²</th><th class="num">p</th><th class="num">Cramér's V</th></tr></thead>
  <tbody><tr>
    <td>tier (auto_accept vs manual_review)</td>
    <td class="num">40,787</td><td class="num">2,551.3</td><td class="num">≈ 0</td><td class="num">0.250</td>
  </tr></tbody>
</table>

<p><strong>Effect size V = 0.25</strong> sits between "medium" (0.3) and "small"
(0.1) — meaningful, not subtle. Manual-review carries
<strong>2.4× more Short</strong> and <strong>2.3× more Complex</strong> calls
than auto_accept.</p>

<div class="callout">
  <strong>Implication for Phase 3.</strong> Primary wild-vs-lab statistics
  should be computed on <code>auto_accept</code>-tier calls only, to avoid
  contaminating biological comparisons with detection-quality artifacts.
</div>

<h2>Finding 4 — Couple keep-set signal <span class="tag tag-confirmed">confirmed</span></h2>

<p>Four couples (<code>m1fm1</code>, <code>m1fm2</code>, <code>m1fm4</code>,
<code>m3fm3</code>) were flagged in upstream labeling as having
noise-prone recordings. Comparing their pooled repertoire to the other
13 couples:</p>

{f("by_couple_keepset", "Repertoire by couple-keep-set",
   "Four flagged couples (left) vs the remaining 13 (right). Flagged couples show ~2.4× more Complex calls.")}

<table>
  <thead><tr><th>Test</th><th class="num">N</th><th class="num">χ²</th><th class="num">p</th><th class="num">Cramér's V</th></tr></thead>
  <tbody><tr>
    <td>couple_keep_set (flagged 4 vs other 13)</td>
    <td class="num">40,787</td><td class="num">1,108.0</td><td class="num">3.9e-236</td><td class="num">0.165</td>
  </tr></tbody>
</table>

<p>The full per-couple breakdown:</p>

{f("by_couple", "Repertoire by individual couple (all 17)",
   "Each bar group is one couple. Overall heterogeneity is real (V = 0.114) but smaller than the keep-set contrast above, suggesting the four flagged couples drive most of the inter-couple variability.")}

<h2>Finding 5 — Two independent noise mechanisms <span class="tag tag-new">new</span></h2>

<p>While inspecting the small HDBSCAN clusters, we found that one of them
(<strong>244 events</strong>) has a very specific acoustic signature that
does not look like a real USV:</p>

<table>
  <thead><tr><th>Feature</th><th>Cluster 1 median</th><th>Real USV typical range</th></tr></thead>
  <tbody>
    <tr><td>Peak frequency</td><td>49 kHz</td><td>60–80 kHz</td></tr>
    <tr><td>Bandwidth</td>     <td>9.8 kHz</td>  <td>30–60 kHz</td></tr>
    <tr><td>Sinuosity</td>     <td>1.04</td>     <td>1.5–2.0 (FM curvature)</td></tr>
    <tr><td>Tier</td>          <td>89.8% auto_accept</td><td>—</td></tr>
  </tbody>
</table>

<p>That profile — low-frequency, narrow-band, no curvature, but
high-confidence — describes a <strong>sustained tonal artifact</strong> the
CNN was confident-wrong about.</p>

<div class="callout">
  <strong>The mechanism is independent from the duration-noise cohort.</strong>
  The four duration-noise-prone couples (Finding 4) were <code>m1fm1</code>,
  <code>m1fm2</code>, <code>m1fm4</code>, <code>m3fm3</code>. The residual
  tonal-noise events concentrate in a different set:
  <strong><code>m5fm5</code> (54), <code>m4fm4</code> (47), <code>m3fm1</code>
  (33), <code>m4fm2</code> (31)</strong> — these four couples account for
  68% of the residual-noise cluster.
</div>

<p>So at least two distinct noise mechanisms are operating in this dataset:</p>

<table>
  <thead><tr><th>Mechanism</th><th>Events</th><th>Couples concentrated in</th><th>Where it ends up</th></tr></thead>
  <tbody>
    <tr><td>A — Duration noise</td><td>~502 (already filtered)</td>
        <td>m1fm1, m1fm2, m1fm4, m3fm3</td>
        <td>Removed by &lt;300 ms filter</td></tr>
    <tr><td>B — Residual tonal artifact</td><td>244</td>
        <td>m5fm5, m4fm4, m3fm1, m4fm2</td>
        <td>Survived all current filters; 89.8% in auto_accept tier</td></tr>
  </tbody>
</table>

<p><strong>For Phase 3</strong>: a second couple-aware noise guard targeting
Mechanism B's cohort may be worth adding before computing wild-vs-lab
repertoire statistics.</p>

<h2>Looking ahead — wild-mouse comparison context</h2>

<p>Phase 3 will compare this lab cohort against the wild-mouse dyads
we have on hand. Four wild-mouse dyads are now available, and their
total event counts span <strong>two orders of magnitude</strong> — the
range is much wider than you might guess:</p>

<table>
  <thead><tr><th>Wild dyad</th><th class="num">WAVs</th><th class="num">USV events</th><th>Notes</th></tr></thead>
  <tbody>
    <tr><td>5970 <small>(lmt_034)</small></td>
        <td class="num">6,400</td><td class="num">7,575</td>
        <td>High-USV reference cohort</td></tr>
    <tr><td>9252</td>
        <td class="num">11,580</td><td class="num">597</td>
        <td>Normally quieter cohort</td></tr>
    <tr><td>3452 <small>(lmt_035, reviewed subset)</small></td>
        <td class="num">853</td><td class="num">401</td>
        <td>Normally quieter cohort</td></tr>
    <tr><td><strong>2379 <small>(lmt_031)</small></strong></td>
        <td class="num">1,280</td><td class="num"><strong>31</strong></td>
        <td><strong>Very short cohort — only 31 high-confidence events
        from 1,280 WAVs (1.64% file yield, 48× lower per-file rate than 5970).
        Will appear in Phase 3 only as a descriptive minority cohort, not a
        primary statistical comparator.</strong></td></tr>
  </tbody>
</table>

<div class="callout">
  <strong>Implication for Phase 3 wild-vs-lab statistics.</strong> Raw
  event counts cannot be compared directly — they confound vocal output
  with recording duration and cohort size. Primary statistics will be
  normalized by per-file rate or recording duration. The 2379 cohort is
  small enough (n = 31) that the usual repertoire-clustering tools
  (DeepSqueak k = 26 k-means, UMAP + HDBSCAN with default
  <code>min_cluster_size</code>) are data-starved; only descriptive
  per-call statistics will be reported for it.
</div>

<h2>Open questions for Mickey</h2>
<ol>
  <li>Should we proceed to Phase 3 (wild-vs-lab statistical comparison)?</li>
  <li>Is the continuous (unimodal) repertoire structure biologically expected
      for inbred lab strains? It is the most novel finding and the down-sample
      test rules out a size artifact, but it would be useful to know whether
      this matches prior expectations or is genuinely surprising.</li>
  <li>The four residual-noise couples (<code>m5fm5</code>, <code>m4fm4</code>,
      <code>m3fm1</code>, <code>m4fm2</code>) — do they share an environmental
      factor (same room, same cage rack, same recording session timing) that
      could produce a sustained 49 kHz tonal artifact?</li>
</ol>

<div class="footer-meta">
  <h3 style="margin-top:0">Methodology — short version</h3>
  <ul>
    <li><strong>Source:</strong> 25,770 × 2-second WAV chunks at 300 kHz from 17 couples
        recorded 2013-12-04, ~14 h total.</li>
    <li><strong>Detection:</strong> production CNN with soft-notch preprocessing
        (artifact suppression in the carrier-frequency band).</li>
    <li><strong>Cleaning:</strong> &lt;300 ms duration filter (post-labeling handoff
        finding); 41,061 calls kept of the original detections.</li>
    <li><strong>Features:</strong> DeepSqueak Excel export — 16 standardized
        acoustic measures per call, including call length, peak/low/high
        frequency, bandwidth, slope, sinuosity, tonality, mean power.</li>
    <li><strong>Classifiers run:</strong>
      <ul>
        <li>Scattoni-2008 rule-based 7-type taxonomy</li>
        <li>DeepSqueak k-means (k=26, hard-coded)</li>
        <li>UMAP + HDBSCAN (data-driven; default <code>min_cluster_size=50</code>,
            <code>min_samples=10</code>)</li>
      </ul>
    </li>
    <li><strong>Repertoire comparison:</strong> per-cohort Scattoni-7 proportions
        with chi-square tests and Cramér's V effect sizes.</li>
    <li><strong>Down-sample test:</strong> stratified sample of the lab cohort
        to n=7,920 (matching wild 3452), preserving per-couple proportions
        (random_state=42), then re-ran UMAP+HDBSCAN with default settings.</li>
  </ul>
</div>

</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="Lab repo root containing results/ and (output) reports/")
    ap.add_argument("--output", type=Path, default=None,
                    help="HTML output path (default: <root>/reports/lab_131204_phase2b_mickey.html)")
    args = ap.parse_args()

    root: Path = args.root
    out_path: Path = args.output or (root / "reports" / "lab_131204_phase2b_mickey.html")

    figures = fig_paths(root)
    missing = [name for name, p in figures.items() if not p.exists()]
    if missing:
        details = "\n  ".join(f"{name}: {figures[name]}" for name in missing)
        raise FileNotFoundError(
            "Missing required figures:\n  " + details
            + "\n(Run Phase 2B + Phase 2C down-sample step first.)"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    html_text = build_html(figures)
    out_path.write_text(html_text, encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path}  ({size_kb:,.1f} KB)")


if __name__ == "__main__":
    main()
