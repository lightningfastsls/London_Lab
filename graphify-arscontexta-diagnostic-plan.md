# Plan: Graphify Diagnostic Run on arscontexta

## Goal

Run [graphify](https://github.com/safishamsi/graphify) on the arscontexta vault as a **diagnostic tool** — not to replace topic maps or retrieval, but to surface structural insights: hidden clusters, god nodes, surprising cross-topic connections, and misalignments between the current 26-topic-map partition and the vault's emergent structure.

## Context for Claude Code

- arscontexta is a ~526-note interconnected markdown vault with 26 topic maps
- The vault already has curated structure — this is about finding what the curation missed
- graphify builds a knowledge graph from files and runs community detection (Leiden clustering)
- We want to **compare** graphify's emergent communities against the existing topic maps

---

## Phase 1: Setup & Discovery

### 1.1 Install graphify
```bash
pip install graphifyy
graphify install
```

### 1.2 Discover vault structure
- Find the arscontexta vault root directory
- List all `.md` files, count them, confirm ~526 notes + 26 topic maps
- Identify which files are topic maps vs. regular notes (topic maps likely have a distinguishing naming convention or frontmatter — discover this, don't assume)
- Report back: total file count, topic map list, any non-md files present

### 1.3 Pre-run sanity check
- Check total token estimate of the vault (rough: `wc -c` on all .md files, divide by 4)
- graphify will call Claude for extraction on each file — estimate API cost/time
- If the vault is very large, consider running on a subset first (e.g., one topic map's linked notes)

---

## Phase 2: Run graphify

### 2.1 Full run
```bash
/graphify /path/to/arscontexta --mode deep --wiki
```

Use `--mode deep` for aggressive inferred edge extraction — we want to find connections Shachar didn't explicitly encode. Use `--wiki` to generate the navigable article output for comparison.

### 2.2 Capture outputs
Confirm these exist in `graphify-out/`:
- `graph.json` — the persistent graph
- `GRAPH_REPORT.md` — god nodes, surprising connections, suggested questions
- `graph.html` — interactive visualization
- `obsidian/` — vault export
- `wiki/` — community articles

---

## Phase 3: Diagnostic Analysis

This is the valuable part. Don't just dump graphify's output — analyze it against arscontexta's existing structure.

### 3.1 God nodes analysis
- Read `GRAPH_REPORT.md`, extract the god nodes list
- For each god node: find which topic map(s) it appears in within the existing vault
- Flag any god node that spans multiple topic maps — these are cross-cutting concerns that might deserve their own topic map or explicit bridge notes
- Flag any god node that doesn't appear in ANY topic map — these are structurally important concepts that the curation missed

### 3.2 Community vs. topic map alignment
- Extract the Leiden communities from `graph.json`
- Map each community to existing topic maps by checking which topic map(s) the community's notes belong to
- Produce an alignment report:
  - **Clean match**: community ≈ one topic map (good — curation matches structure)
  - **Split**: one topic map's notes spread across multiple communities (topic map may be too broad)
  - **Merge**: multiple topic maps collapse into one community (topic maps may be redundant or over-partitioned)
  - **Orphan**: notes in a community that don't belong to any topic map

### 3.3 Surprising connections
- Read the surprising connections from `GRAPH_REPORT.md`
- Filter for cross-topic-map edges (connections between notes in different topic maps)
- Rank by graphify's composite score
- For the top 10: check whether arscontexta already has explicit links between these notes
- Any high-scoring cross-topic connection that ISN'T already linked = a discovery

### 3.4 Confidence distribution
- From `graph.json`, count edges by confidence label (EXTRACTED / INFERRED / AMBIGUOUS)
- High AMBIGUOUS count in a region = that area of the vault might need more explicit linking
- High INFERRED count between two topic maps = there's a latent relationship worth formalizing

---

## Phase 4: Deliverable

### 4.1 Write a diagnostic report
Create `arscontexta-graphify-diagnostic.md` containing:

1. **Summary stats**: nodes, edges, communities, confidence distribution
2. **God nodes table**: node | degree | topic maps it appears in | assessment
3. **Community-topic map alignment matrix**: community ID | primary topic map | alignment type (clean/split/merge/orphan) | notes
4. **Top 15 surprising cross-topic connections**: source | target | score | already linked? | recommendation
5. **Structural recommendations**: specific actions (new bridge notes, topic map splits/merges, missing links to add)
6. **Token benchmark**: graphify's reported reduction ratio for the vault

### 4.2 Format for arscontexta ingestion
The diagnostic report itself should be ingestible as a vault note. Add appropriate frontmatter and internal links so it connects to the topic maps it references.

---

## Hard Stops

- **Do NOT modify any arscontexta notes.** This is read-only diagnostics.
- **Do NOT restructure topic maps based on graphify output alone.** The report is for Shachar to evaluate — graphify's communities are statistical, not semantic.
- If graphify extraction fails on any files, log which ones and continue — don't retry in a loop.
- If the vault is too large for a single run (>500 files × Claude extraction calls), run Phase 2 on a representative subset first and report estimated full-run cost before proceeding.

---

## What to Steal (Regardless of Results)

Even if the diagnostic reveals no surprises, evaluate these graphify patterns for adoption in arscontexta:

1. **Confidence labeling on edges** — could arscontexta topic maps distinguish EXTRACTED (Shachar explicitly linked) vs. INFERRED (found by retrieval/search) connections?
2. **SHA256 caching for incremental updates** — arscontexta currently lacks a formalized ingest workflow; graphify's cache pattern could inform one
3. **The `--watch` pattern** — auto-updating an index when files change, relevant to the missing lint operation identified in the vault health audit
