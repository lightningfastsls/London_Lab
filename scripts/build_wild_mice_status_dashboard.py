"""Build a self-contained HTML status dashboard for the four wild-mouse cohorts.

Audits the detection + classification pipeline for each cohort
(5970, 3452, 9252, 2379), embeds figures inline as base64, and writes a
single offline-capable HTML file to
``reports/wild_mice_status_dashboard.html``.

Per cohort, the dashboard surfaces:
  - WAV count, JSON count, classified-event count
  - Pipeline stage status (detection, DeepSqueak, 4 classifiers)
  - Headline figures (Scattoni-7 type distribution if available)
  - Per-cohort verdict

Cohort biology (authoritative — confirmed by domain expert 2026-05-15):
  All four are wild mouse dyads, NOT different strains. 5970 is unusually
  vocal; 3452, 9252, and 2379 are normally quieter cohorts. Low USV count
  is biology, not pipeline failure.

Usage::

    .venv/bin/python scripts/build_wild_mice_status_dashboard.py
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path("/home/shachar/projects/mickey_london_lab")


def encode_png(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def img(path: Path, alt: str, caption: str = "") -> str:
    if not path.exists():
        return f'<p class="missing">[figure not yet generated: {html.escape(str(path.name))}]</p>'
    src = encode_png(path)
    cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
    return f'<figure><img src="{src}" alt="{html.escape(alt)}">{cap}</figure>'


def safe_csv_rows(path: Path) -> int | None:
    try:
        return len(pd.read_csv(path))
    except Exception:
        return None


def safe_xlsx_rows(path: Path) -> int | None:
    try:
        return len(pd.read_excel(path))
    except Exception:
        return None


def n_files(path: Path, pattern: str = "*") -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob(pattern))


CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  max-width: 1000px; margin: 2em auto; padding: 0 1.5em;
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
.banner-corr {
  background: #fefcbf; border-left: 4px solid #d69e2e;
  padding: 0.8em 1.2em; margin: 1em 0; border-radius: 4px;
}
.badge {
  display: inline-block; padding: 0.15em 0.7em; border-radius: 12px;
  font-size: 0.78em; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.04em; margin: 0 0.2em;
}
.b-ready    { background: #c6f6d5; color: #22543d; }
.b-ready-lc { background: #bee3f8; color: #2a4365; }
.b-todo     { background: #fed7d7; color: #742a2a; }
.b-na       { background: #edf2f7; color: #4a5568; }
.cohort-card {
  border: 1px solid #cbd5e0; border-radius: 6px;
  margin: 1.5em 0; padding: 1.2em 1.5em; background: #fafbfc;
}
.cohort-header {
  display: flex; justify-content: space-between; align-items: baseline;
  border-bottom: 2px solid #e2e8f0; padding-bottom: .3em; margin-bottom: 1em;
}
.cohort-header h2 { margin: 0; border: none; padding: 0; }
table {
  border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.92em;
}
th, td { padding: 0.5em 0.7em; text-align: left; border: 1px solid #cbd5e0; }
th { background: #edf2f7; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.center { text-align: center; }
.callout {
  background: #fffaf0; border-left: 4px solid #dd6b20;
  padding: 0.8em 1.2em; margin: 1em 0; border-radius: 4px;
}
.callout-good { background: #f0fff4; border-left-color: #38a169; }
figure { margin: 1em 0; text-align: center; }
figure img { max-width: 100%; height: auto; border: 1px solid #e2e8f0;
             border-radius: 4px; }
figcaption { font-size: 0.86em; color: #4a5568; margin-top: 0.4em; font-style: italic; }
.figrow { display: flex; flex-wrap: wrap; gap: 1em; justify-content: center; }
.figrow > figure { flex: 1 1 360px; max-width: 49%; margin: 0; }
code { background: #edf2f7; padding: 0.1em 0.4em; border-radius: 3px; font-size: 0.92em; }
.missing { color: #c53030; font-style: italic; }
.legend { font-size: 0.86em; color: #4a5568; margin-top: -0.6em; }
"""


# -------- cohort facts -------- #
def collect_state(root: Path) -> dict:
    cohorts: list[dict] = []

    # ---- 5970 (lmt_034) ---- #
    c5970 = {
        "name": "5970",
        "lmt": "lmt_034",
        "biology_note": "Unusually vocal cohort — used as the high-USV reference.",
        "wav_dirs": [root / "5970"],
        "batch_dir": root / "results/batch_5970_v2_full",
        "deepsqueak_xlsx": root / "deepsqueak_output_full/classified_Stats.xlsx",
        "raven_dir":       root / "raven_tables_full",
        "classified_csv":  root / "classified_detections_full.csv",
        "taxonomy_csv":    root / "results/traditional_taxonomy/classified_traditional.csv",
        "hdbscan_csv":     root / "results/recluster_umap_hdbscan/reclassified_detections.csv",
        "acoustic_dir":    root / "results/acoustic_feature_analysis",
        "sequential_dir":  root / "results/sequential_structure",
        "tax_fig":         root / "results/traditional_taxonomy/type_distribution.png",
        "status_badge":    "ready",
        "status_label":    "READY",
        "verdict":         "End-to-end pipeline complete (2026-04-03). High-USV reference cohort for Phase 3.",
    }
    cohorts.append(c5970)

    # ---- 3452 (lmt_035) ---- #
    c3452 = {
        "name": "3452",
        "lmt": "lmt_035",
        "biology_note": "Normally quieter cohort. Low USV count reflects the animal, not the pipeline.",
        "wav_dirs": [root / "USV_3452_sample_reviewed", root / "USV_3452_sample"],
        "batch_dir": root / "results/batch_3452_reviewed",
        "deepsqueak_xlsx": root / "deepsqueak_output_3452/classified_Stats.xlsx",
        "raven_dir":       root / "raven_tables_3452",
        "classified_csv":  root / "classified_detections_3452.csv",
        "taxonomy_csv":    root / "results/traditional_taxonomy_3452/classified_traditional.csv",
        "hdbscan_csv":     root / "results/recluster_umap_hdbscan_3452/reclassified_detections.csv",
        "acoustic_dir":    root / "results/acoustic_feature_analysis_3452",
        "sequential_dir":  root / "results/sequential_structure_3452",
        "tax_fig":         root / "results/traditional_taxonomy_3452/type_distribution.png",
        "status_badge":    "ready-lc",
        "status_label":    "READY (low-count cohort)",
        "verdict":         "End-to-end pipeline complete (2026-04-06). Low absolute event count is biology, not a pipeline gap.",
    }
    cohorts.append(c3452)

    # ---- 9252 ---- #
    c9252 = {
        "name": "9252",
        "lmt": "lmt_??? (TBD)",
        "biology_note": "Normally quieter cohort. Across 8 sessions, USV3 dominates while USV4 is nearly silent — biology, not bug.",
        "wav_dirs": [root / "USV_9252"],
        "batch_dir": root / "results/batch_9252",
        "deepsqueak_xlsx": root / "deepsqueak_output_9252/classified_Stats.xlsx",
        "raven_dir":       root / "raven_tables_9252",
        "classified_csv":  root / "classified_detections_9252.csv",
        "taxonomy_csv":    root / "results/traditional_taxonomy_9252/classified_traditional.csv",
        "hdbscan_csv":     root / "results/recluster_umap_hdbscan_9252/reclassified_detections.csv",
        "acoustic_dir":    root / "results/acoustic_feature_analysis_9252",
        "sequential_dir":  root / "results/sequential_structure_9252",
        "tax_fig":         root / "results/traditional_taxonomy_9252/type_distribution.png",
        "status_badge":    "ready-lc",
        "status_label":    "READY (low-count cohort)",
        "verdict":         "End-to-end pipeline complete (2026-04-25). 22.96× fewer events/file than 5970 — supported by per-session vocal-rate evidence as a genuine biological difference, not detection failure.",
    }
    anomaly_json = root / "results/rate_anomaly_9252/rate_anomaly_stats.json"
    if anomaly_json.exists():
        c9252["anomaly_stats"] = json.loads(anomaly_json.read_text())
    cohorts.append(c9252)

    # ---- 2379 (lmt_031) — the missed 4th cohort ---- #
    c2379 = {
        "name": "2379",
        "lmt": "lmt_031",
        "biology_note": "Fourth wild-mouse dyad — quietest cohort observed. 1.64% file yield, 0.024 events/file; 48× lower per-file rate than 5970.",
        "wav_dirs": [root / "USV_2379_sample", root / "USV_2379_sample_reviewed"],
        "batch_dir":       root / "results/batch_2379",
        "deepsqueak_xlsx": None,
        "raven_dir":       None,
        "classified_csv":  None,
        "taxonomy_csv":    None,
        "hdbscan_csv":     None,
        "acoustic_dir":    None,
        "sequential_dir":  None,
        "tax_fig":         None,
        "status_badge":    "todo",
        "status_label":    "DETECTION DONE — 31 EVENTS",
        "verdict":         "Detection run completed 2026-05-15 in 5,850 s on 1,280 WAVs. Produced 31 high-confidence events (17 auto_accept / 4 manual_review tier; rest auto_reject). At this event count, downstream classifiers (DeepSqueak k=26 k-means, HDBSCAN with default min_cluster_size=50, sequential transition matrices) are data-starved and would not produce meaningful cohort statistics. Treat 2379 as a descriptive minority cohort.",
    }
    cohorts.append(c2379)

    for c in cohorts:
        c["n_wavs"]   = sum(n_files(d, "*.wav") for d in (c["wav_dirs"] or []))
        c["n_jsons"]  = n_files(c["batch_dir"], "*.json") if c["batch_dir"] else 0
        c["n_ds"]     = safe_xlsx_rows(c["deepsqueak_xlsx"]) if c["deepsqueak_xlsx"] else None
        c["n_raven"]  = n_files(c["raven_dir"], "*.txt") if c["raven_dir"] else 0
        c["n_classified"] = safe_csv_rows(c["classified_csv"]) if c["classified_csv"] else None
        c["n_taxonomy"]   = safe_csv_rows(c["taxonomy_csv"])   if c["taxonomy_csv"] else None
        c["n_hdbscan"]    = safe_csv_rows(c["hdbscan_csv"])    if c["hdbscan_csv"] else None

    return {"cohorts": cohorts, "root": root}


def render_cohort_card(c: dict, root: Path) -> str:
    badge_class = {
        "ready": "b-ready",
        "ready-lc": "b-ready-lc",
        "todo": "b-todo",
    }[c["status_badge"]]

    stages = [
        ("Raw detection (JSONs in batch dir)",        c["n_jsons"] > 0),
        ("Raven tables exported",                      c["n_raven"] > 0),
        ("DeepSqueak k-means classified Stats",        c["n_ds"] is not None and c["n_ds"] > 0),
        ("Top-level classified_detections CSV",        c["n_classified"] is not None and c["n_classified"] > 0),
        ("Scattoni-7 rule-based taxonomy",             c["n_taxonomy"] is not None and c["n_taxonomy"] > 0),
        ("UMAP + HDBSCAN re-cluster",                  c["n_hdbscan"] is not None and c["n_hdbscan"] > 0),
        ("Acoustic feature analysis",                  bool(c["acoustic_dir"] and c["acoustic_dir"].exists())),
        ("Sequential structure analysis",              bool(c["sequential_dir"] and c["sequential_dir"].exists())),
    ]
    stage_rows = "".join(
        f'<tr><td>{html.escape(label)}</td>'
        f'<td class="center">{"✅" if ok else "❌"}</td></tr>'
        for label, ok in stages
    )

    # Optional rate-evidence callout for 9252 (kept, but reframed as supporting biology, not flagging a bug)
    rate_evidence = ""
    if c["name"] == "9252" and c.get("anomaly_stats"):
        a = c["anomaly_stats"]
        h = a["headline"]
        rate_evidence = f"""
<div class="callout callout-good">
<strong>Cross-cohort detection-rate evidence (2026-04-24)</strong>
<p style="margin-top:.5em">An earlier investigation expected the 9252 vs 5970
event gap to be a detection bug. Hypothesis testing supported a biological
explanation — confirmed by domain expert 2026-05-15.</p>
<table>
<thead><tr><th>Metric</th><th class="num">5970</th><th class="num">9252</th><th class="num">Ratio</th></tr></thead>
<tbody>
  <tr><td>files with ≥1 event</td>
      <td class="num">{h['5970']['n_files_with_events']:,}</td>
      <td class="num">{h['9252']['n_files_with_events']:,}</td><td class="num">—</td></tr>
  <tr><td>file yield %</td>
      <td class="num">{h['5970']['file_yield_pct']:.2f}%</td>
      <td class="num">{h['9252']['file_yield_pct']:.2f}%</td>
      <td class="num">{h['ratios']['file_yield_ratio_5970_over_9252']:.2f}×</td></tr>
  <tr><td>events / file (mean)</td>
      <td class="num">{h['5970']['events_per_file_mean']:.3f}</td>
      <td class="num">{h['9252']['events_per_file_mean']:.3f}</td>
      <td class="num">{h['ratios']['events_per_file_ratio_5970_over_9252']:.2f}×</td></tr>
  <tr><td>total events</td>
      <td class="num">{h['5970']['n_events']:,}</td>
      <td class="num">{h['9252']['n_events']:,}</td><td class="num">—</td></tr>
</tbody>
</table>
<p><strong>H1 recording length:</strong> weak — clips contain events out to ~1+ s.<br>
<strong>H2 animal silence:</strong> supported (per-session range 0.011–0.18 ev/file, 17× dispersion).<br>
<strong>H3 noise floor:</strong> falsified — 9252 is <em>quieter</em>, not louder (KS p ≈ 1e-121).<br>
<strong>H4 date/season:</strong> weak — only 5 days separate the two recording windows.</p>
<div class="figrow">
{img(root / "results/rate_anomaly_9252/fig_h2_per_session_rate.png",
     "Per-session detection rate",
     "Per-session detection rate: USV3 dominates (0.18 ev/file); USV4 nearly silent. Genuine biological variability across sessions.")}
{img(root / "results/rate_anomaly_9252/fig_h3_noise_floor.png",
     "Noise-floor distribution",
     "Noise-floor distributions overlap with 9252 cleaner than 5970 — rules out a noise-floor cause for the event-rate gap.")}
</div>
</div>
"""

    # 2379-specific TODO callout
    todo_block = ""
    if c["status_badge"] == "todo":
        todo_block = """
<div class="callout">
<strong>What's needed to bring 2379 to parity with the other cohorts:</strong>
<ol style="margin:0.5em 0 0 1em">
  <li>Run batch detection (<code>scripts/run_batch_detection.py</code>) on <code>USV_2379_sample_reviewed/</code>.</li>
  <li>Export Raven tables (<code>scripts/export_raven_tables.py --batch-format</code>) → <code>raven_tables_2379/</code>.</li>
  <li>Run the MATLAB DeepSqueak k-means classifier → <code>deepsqueak_output_2379/classified_Stats.xlsx</code>.</li>
  <li>Import (<code>scripts/import_deepsqueak_results.py --tolerance-ms 75</code>) → <code>classified_detections_2379.csv</code>.</li>
  <li>Run the four classifier scripts (Scattoni-7, UMAP+HDBSCAN, acoustic features, sequential structure) with <code>_2379</code>-suffixed output dirs.</li>
</ol>
<p style="margin-top:.5em">At 77 WAVs the cohort is small even by quiet-mouse standards. Expect Phase 3 inclusion only as a descriptive minority cohort, not a primary statistical comparator.</p>
</div>
"""

    fig_block = (
        img(c["tax_fig"], f"Scattoni-7 distribution — {c['name']}",
            f"Per-type call counts for {c['name']}. Total events: {c['n_taxonomy'] or 'n/a'}.")
        if c["tax_fig"] else
        '<p class="missing">[no Scattoni-7 figure yet — needs full classifier pipeline]</p>'
    )

    return f"""
<div class="cohort-card">
  <div class="cohort-header">
    <h2>{html.escape(c['name'])} <small style="color:#718096;font-weight:normal">({html.escape(c['lmt'])})</small></h2>
    <span class="badge {badge_class}">{html.escape(c['status_label'])}</span>
  </div>
  <p><em>{html.escape(c['biology_note'])}</em></p>
  <p>{html.escape(c['verdict'])}</p>

  <table>
    <thead><tr><th>Quantity</th><th class="num">Value</th></tr></thead>
    <tbody>
      <tr><td>WAV files</td><td class="num">{c['n_wavs']:,}</td></tr>
      <tr><td>Raw detection JSONs</td><td class="num">{c['n_jsons']:,}</td></tr>
      <tr><td>Raven tables (.txt)</td><td class="num">{c['n_raven']:,}</td></tr>
      <tr><td>DeepSqueak classified rows</td><td class="num">{(c['n_ds'] or 0):,}</td></tr>
      <tr><td>Top-level classified CSV rows</td><td class="num">{(c['n_classified'] or 0):,}</td></tr>
      <tr><td>Scattoni-7 rows</td><td class="num">{(c['n_taxonomy'] or 0):,}</td></tr>
      <tr><td>HDBSCAN re-cluster rows</td><td class="num">{(c['n_hdbscan'] or 0):,}</td></tr>
    </tbody>
  </table>

  <h3>Pipeline stage completeness</h3>
  <table>
    <thead><tr><th>Stage</th><th class="center">Present?</th></tr></thead>
    <tbody>{stage_rows}</tbody>
  </table>

  {fig_block}
  {rate_evidence}
  {todo_block}
</div>
"""


def build_html(state: dict) -> str:
    root = state["root"]
    cohort_blocks = "\n".join(render_cohort_card(c, root) for c in state["cohorts"])

    overall = "\n".join(
        f'<tr><td>{c["name"]} ({c["lmt"]})</td>'
        f'<td class="center"><span class="badge b-{c["status_badge"]}">{c["status_label"]}</span></td>'
        f'<td class="num">{c["n_wavs"]:,}</td>'
        f'<td class="num">{(c["n_classified"] or 0):,}</td>'
        f'<td>{html.escape(c["biology_note"])}</td></tr>'
        for c in state["cohorts"]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Wild-mouse cohorts — pipeline status dashboard</title>
<style>{CSS}</style>
</head>
<body>

<h1>Wild-mouse cohorts — pipeline status</h1>

<div class="banner">
  <strong>Question:</strong> are all four wild-mouse cohorts ready for the
  Phase 3 wild-vs-lab statistical comparison?<br>
  <strong>Short answer:</strong>
  <span class="badge b-ready">5970 READY</span>
  <span class="badge b-ready-lc">3452 READY (low-count)</span>
  <span class="badge b-ready-lc">9252 READY (low-count)</span>
  <span class="badge b-todo">2379 NOT STARTED</span>
  <p class="legend">Generated 2026-05-15. Wild-mouse cohort biology
  confirmed by domain expert: all four are wild mouse <em>dyads</em>, not
  different strains. 5970 is the unusually vocal one; the others are
  normally quieter.</p>
</div>

<div class="banner-corr">
  <strong>Correction note (2026-05-15).</strong> An earlier draft of this
  dashboard (a) framed 9252's low event count as a "detection anomaly,"
  (b) reported 3452 as "partial because full sample wasn't classified," and
  (c) listed only three cohorts. All three points were wrong:
  9252's low rate is biology (per cohort owner); 3452's DeepSqueak pipeline
  is complete on the reviewed subset (the canonical 3452 product); and the
  fourth dyad <code>2379 (lmt_031)</code> exists but has not yet been pushed
  through detection.
</div>

<h2>Summary</h2>
<table>
  <thead><tr><th>Cohort</th><th class="center">Status</th>
             <th class="num">WAVs</th><th class="num">Classified events</th>
             <th>Biology</th></tr></thead>
  <tbody>{overall}</tbody>
</table>

<p class="legend">
  <span class="badge b-ready">READY</span> end-to-end pipeline complete; high-USV cohort. &nbsp;
  <span class="badge b-ready-lc">READY (low-count)</span> end-to-end pipeline complete; cohort is naturally quieter, so absolute event counts are smaller — biology, not pipeline gap. &nbsp;
  <span class="badge b-todo">NOT STARTED</span> WAVs present, no detection or downstream artifacts yet. &nbsp;
</p>

{cohort_blocks}

<h2>Recommended next actions</h2>
<ol>
  <li><strong>2379 (lmt_031) — process through the pipeline.</strong>
      77 WAVs in <code>USV_2379_sample_reviewed/</code>. Run the five
      pipeline steps listed in that cohort's card. Expect a small absolute
      event count similar to 3452.</li>
  <li><strong>Phase 3 design — weight cohorts by per-file rate, not raw n.</strong>
      Treat 5970 as the high-USV reference and 3452/9252/(2379) as
      low-count cohorts. Cross-cohort statistics must normalize by recording
      duration or per-file rate to avoid n-driven artifacts.</li>
  <li><strong>Optional — record canonical per-cohort vocal-rate baseline.</strong>
      The 2026-04-24 rate-anomaly investigation already produced per-session
      events/file for 9252. Repeating this for 5970, 3452, and (post-pipeline)
      2379 would give Phase 3 a documented prior on expected per-cohort rates.</li>
</ol>

<p class="legend">This dashboard rebuilds with:
<code>.venv/bin/python scripts/build_wild_mice_status_dashboard.py</code></p>

</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    root = args.root
    out_path = args.output or (root / "reports" / "wild_mice_status_dashboard.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    state = collect_state(root)
    out_path.write_text(build_html(state), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path}  ({size_kb:,.1f} KB)")


if __name__ == "__main__":
    main()
