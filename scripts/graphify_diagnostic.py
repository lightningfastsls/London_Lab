#!/usr/bin/env python3
"""
Graphify Diagnostic: Compare graphify's emergent graph structure
against arscontexta's curated topic maps.

Reads:
  - graphify-out/graph.json         (graphify output)
  - ops/cache/topic-map-index.json  (existing vault structure)
  - notes/*.md                      (for wiki-link extraction)

Writes:
  - graphify-out/arscontexta-graphify-diagnostic.md

Usage:
  python scripts/graphify_diagnostic.py
  python scripts/graphify_diagnostic.py --graph-json path/to/graph.json
  python scripts/graphify_diagnostic.py --help
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_graphify_graph(path: Path) -> dict:
    """Parse graph.json (NetworkX node_link_data format with community annotations).

    Returns dict with:
        nodes: list[dict]        — each has id, label, source_file, file_type, community
        links: list[dict]        — each has source, target, relation, confidence, confidence_score
        hyperedges: list[dict]   — optional group relationships
        node_by_id: dict         — id -> node dict
        source_file_to_nodes: dict — source_file stem -> [node dicts]
        community_to_nodes: dict — community_id -> [node dicts]
    """
    raw = json.loads(path.read_text())

    nodes = raw.get("nodes", [])
    links = raw.get("links", [])
    hyperedges = raw.get("hyperedges", [])

    node_by_id = {n["id"]: n for n in nodes}

    # Reverse index: source file -> nodes
    source_file_to_nodes = defaultdict(list)
    for n in nodes:
        sf = n.get("source_file")
        if sf:
            stem = Path(sf).stem
            source_file_to_nodes[stem].append(n)

    # Reverse index: community -> nodes
    community_to_nodes = defaultdict(list)
    for n in nodes:
        cid = n.get("community")
        if cid is not None:
            community_to_nodes[cid].append(n)

    return {
        "nodes": nodes,
        "links": links,
        "hyperedges": hyperedges,
        "node_by_id": node_by_id,
        "source_file_to_nodes": dict(source_file_to_nodes),
        "community_to_nodes": dict(community_to_nodes),
    }


def load_topic_map_index(path: Path) -> dict:
    """Parse topic-map-index.json.

    Returns dict with:
        note_to_topics: dict    — note_title -> [topic_map_names]
        topic_to_notes: dict    — topic_map_name -> [note_titles]
        maps: dict              — raw maps data (sections, context_phrases, etc.)
        hierarchy: dict         — parent/child relationships
        moc_names: set          — all MOC names
    """
    raw = json.loads(path.read_text())

    # note_title -> topic maps
    note_to_topics = {}
    for title, info in raw.get("notes", {}).items():
        note_to_topics[title] = info.get("topics", [])

    # topic_map -> notes
    topic_to_notes = {}
    for moc_name, moc_data in raw.get("maps", {}).items():
        topic_to_notes[moc_name] = moc_data.get("all_notes", [])

    moc_names = set(raw.get("maps", {}).keys())

    return {
        "note_to_topics": note_to_topics,
        "topic_to_notes": topic_to_notes,
        "maps": raw.get("maps", {}),
        "hierarchy": raw.get("hierarchy", {}),
        "moc_names": moc_names,
    }


# ---------------------------------------------------------------------------
# Mapping: graphify nodes <-> vault topic maps
# ---------------------------------------------------------------------------

def map_nodes_to_topic_maps(graph: dict, tmi: dict) -> dict:
    """For each graphify node, resolve source_file -> note title -> topic map(s).

    Returns dict with:
        node_to_topics: dict       — node_id -> [topic_map_names]
        unmapped_nodes: list       — node_ids that couldn't be resolved
        coverage: dict             — {mapped: int, unmapped: int, total: int}
    """
    node_to_topics = {}
    unmapped = []

    # Build a lookup: filename stem -> note title (since note titles ARE filenames)
    stem_to_title = {}
    for title in tmi["note_to_topics"]:
        stem_to_title[title] = title
    # Also index MOC names
    for moc_name in tmi["moc_names"]:
        stem_to_title[moc_name] = moc_name

    for node in graph["nodes"]:
        nid = node["id"]
        sf = node.get("source_file")
        if not sf:
            unmapped.append(nid)
            node_to_topics[nid] = []
            continue

        stem = Path(sf).stem
        # Try direct match
        if stem in tmi["note_to_topics"]:
            node_to_topics[nid] = tmi["note_to_topics"][stem]
        elif stem in tmi["moc_names"]:
            # Node comes from a MOC file itself
            node_to_topics[nid] = [stem]
        else:
            unmapped.append(nid)
            node_to_topics[nid] = []

    mapped_count = len(graph["nodes"]) - len(unmapped)
    return {
        "node_to_topics": node_to_topics,
        "unmapped_nodes": unmapped,
        "coverage": {
            "mapped": mapped_count,
            "unmapped": len(unmapped),
            "total": len(graph["nodes"]),
        },
    }


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_god_nodes(graph: dict, tmi: dict, mapping: dict, top_n: int = 15) -> list[dict]:
    """Rank nodes by degree, cross-reference against topic maps."""
    # Compute degree from links
    degree = Counter()
    for link in graph["links"]:
        degree[link["source"]] += 1
        degree[link["target"]] += 1

    # Rank by degree, take top N
    ranked = degree.most_common(top_n * 2)  # over-fetch to filter

    results = []
    for nid, deg in ranked:
        if len(results) >= top_n:
            break
        node = graph["node_by_id"].get(nid)
        if not node:
            continue

        topics = mapping["node_to_topics"].get(nid, [])
        source_file = node.get("source_file", "")
        label = node.get("label", nid)

        # Assessment
        if len(topics) > 1:
            assessment = f"Cross-cutting: spans {len(topics)} topic maps"
        elif len(topics) == 0:
            assessment = "Not in any topic map — potential curation gap"
        else:
            assessment = f"Within {topics[0]}"

        results.append({
            "label": label,
            "node_id": nid,
            "degree": deg,
            "source_file": source_file,
            "topic_maps": topics,
            "assessment": assessment,
        })

    return results


def analyze_community_alignment(graph: dict, tmi: dict, mapping: dict) -> list[dict]:
    """For each Leiden community, compute topic map alignment.

    Classifications:
        clean_match: >=70% of files from one topic map
        split: one topic map's notes across 3+ communities
        merge: 2+ topic maps each contribute >=25% of a community
        orphan: files not in any topic map
    """
    results = []

    for cid, nodes in sorted(graph["community_to_nodes"].items()):
        # Collect source files for this community
        source_files = set()
        for node in nodes:
            sf = node.get("source_file")
            if sf:
                source_files.add(Path(sf).stem)

        # Map source files to topic maps
        topic_counts = Counter()
        orphan_files = []
        for stem in source_files:
            topics = tmi["note_to_topics"].get(stem, [])
            if not topics and stem not in tmi["moc_names"]:
                orphan_files.append(stem)
            else:
                for t in topics:
                    topic_counts[t] += 1
                if stem in tmi["moc_names"]:
                    topic_counts[stem] += 1

        total_files = len(source_files)
        if total_files == 0:
            continue

        # Compute distribution
        distribution = {}
        for topic, count in topic_counts.most_common():
            distribution[topic] = round(count / total_files, 2)

        # Classify alignment
        top_topic, top_count = topic_counts.most_common(1)[0] if topic_counts else ("none", 0)
        top_fraction = top_count / total_files if total_files > 0 else 0

        significant_topics = [t for t, c in topic_counts.items() if c / total_files >= 0.25]

        if top_fraction >= 0.70:
            alignment_type = "clean_match"
        elif len(significant_topics) >= 2:
            alignment_type = "merge"
        elif orphan_files:
            alignment_type = "orphan"
        else:
            alignment_type = "mixed"

        results.append({
            "community_id": cid,
            "node_count": len(nodes),
            "file_count": total_files,
            "primary_topic_map": top_topic,
            "primary_fraction": round(top_fraction, 2),
            "alignment_type": alignment_type,
            "topic_distribution": distribution,
            "orphan_files": orphan_files,
        })

    # Also check for splits: topic maps whose notes span 3+ communities
    topic_to_communities = defaultdict(set)
    for r in results:
        for topic in r["topic_distribution"]:
            topic_to_communities[topic].add(r["community_id"])

    splits = {
        t: sorted(cids)
        for t, cids in topic_to_communities.items()
        if len(cids) >= 3
    }

    return {
        "communities": results,
        "splits": splits,
    }


def extract_existing_wikilinks(notes_dir: Path) -> dict[str, set[str]]:
    """Scan all .md files, extract [[wiki-link]] targets.

    Returns {filename_stem: set_of_linked_stems}.
    """
    link_re = re.compile(r"\[\[([^\]]+)\]\]")
    links = {}

    for md_file in sorted(notes_dir.glob("*.md")):
        stem = md_file.stem
        text = md_file.read_text(errors="replace")
        targets = set(link_re.findall(text))
        links[stem] = targets

    return links


def analyze_surprising_connections(
    graph: dict, tmi: dict, mapping: dict, existing_links: dict, top_n: int = 15
) -> list[dict]:
    """Find cross-topic-map edges ranked by surprise score.

    For each: check whether the vault already has an explicit wiki-link.
    """
    cross_topic_edges = []

    for link in graph["links"]:
        src_topics = set(mapping["node_to_topics"].get(link["source"], []))
        tgt_topics = set(mapping["node_to_topics"].get(link["target"], []))

        # Skip same-topic edges
        if src_topics & tgt_topics:
            continue
        # Skip if both unmapped
        if not src_topics and not tgt_topics:
            continue

        # Compute surprise score (replicating graphify's logic)
        conf = link.get("confidence", "EXTRACTED")
        conf_bonus = {"AMBIGUOUS": 3, "INFERRED": 2, "EXTRACTED": 1}.get(conf, 1)
        cross_community_bonus = 1  # already cross-topic by filter
        score = conf_bonus + cross_community_bonus + link.get("confidence_score", 0.5)

        src_node = graph["node_by_id"].get(link["source"], {})
        tgt_node = graph["node_by_id"].get(link["target"], {})

        # Check existing wiki-link
        src_sf = Path(src_node.get("source_file", "")).stem if src_node.get("source_file") else ""
        tgt_sf = Path(tgt_node.get("source_file", "")).stem if tgt_node.get("source_file") else ""

        already_linked = False
        if src_sf and tgt_sf:
            src_links = existing_links.get(src_sf, set())
            tgt_links = existing_links.get(tgt_sf, set())
            already_linked = tgt_sf in src_links or src_sf in tgt_links

        cross_topic_edges.append({
            "source_node": src_node.get("label", link["source"]),
            "target_node": tgt_node.get("label", link["target"]),
            "source_topic_maps": sorted(src_topics),
            "target_topic_maps": sorted(tgt_topics),
            "relation": link.get("relation", "unknown"),
            "confidence": conf,
            "confidence_score": link.get("confidence_score", 0),
            "surprise_score": round(score, 2),
            "already_linked": already_linked,
            "recommendation": "existing link" if already_linked else "NEW — consider adding wiki-link",
        })

    # Sort by surprise score descending
    cross_topic_edges.sort(key=lambda e: e["surprise_score"], reverse=True)
    return cross_topic_edges[:top_n]


def analyze_confidence_distribution(graph: dict, mapping: dict) -> dict:
    """Count edges by confidence label, grouped by topic-map region."""
    total_counts = Counter()
    per_topic = defaultdict(lambda: Counter())
    cross_topic_inferred = Counter()

    for link in graph["links"]:
        conf = link.get("confidence", "EXTRACTED")
        total_counts[conf] += 1

        # Group by source node's topic
        src_topics = mapping["node_to_topics"].get(link["source"], [])
        tgt_topics = mapping["node_to_topics"].get(link["target"], [])

        for t in src_topics:
            per_topic[t][conf] += 1

        # Track cross-topic INFERRED edges
        if conf == "INFERRED":
            src_set = set(src_topics)
            tgt_set = set(tgt_topics)
            if src_set and tgt_set and not (src_set & tgt_set):
                pair = tuple(sorted([src_topics[0], tgt_topics[0]]))
                cross_topic_inferred[pair] += 1

    # Find high-ambiguous regions
    high_ambiguous = {
        t: counts["AMBIGUOUS"]
        for t, counts in per_topic.items()
        if counts.get("AMBIGUOUS", 0) >= 3
    }

    return {
        "total": dict(total_counts),
        "per_topic_map": {t: dict(c) for t, c in per_topic.items()},
        "high_ambiguous_regions": high_ambiguous,
        "cross_topic_inferred_pairs": dict(cross_topic_inferred.most_common(10)),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    graph: dict,
    tmi: dict,
    mapping: dict,
    god_nodes: list[dict],
    alignment: dict,
    surprises: list[dict],
    confidence: dict,
) -> str:
    """Generate vault-ingestible diagnostic report."""
    today = date.today().isoformat()
    n_nodes = len(graph["nodes"])
    n_edges = len(graph["links"])
    n_communities = len(graph["community_to_nodes"])
    n_hyperedges = len(graph["hyperedges"])
    cov = mapping["coverage"]

    lines = []

    # --- Frontmatter ---
    lines.append("---")
    lines.append(f'description: "Graphify diagnostic comparing {n_communities} emergent Leiden communities against 26 curated topic maps — god nodes, alignment gaps, and surprising cross-topic connections"')
    lines.append("type: finding")
    lines.append("confidence: experimental")
    lines.append(f"created: {today}")
    lines.append("topics:")
    lines.append('  - "[[graph-structure]]"')
    lines.append("---")
    lines.append("")

    # --- Title ---
    lines.append("# Graphify Diagnostic: arscontexta Vault")
    lines.append("")

    # --- Section 1: Summary stats ---
    lines.append("## 1. Summary Statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Nodes (concepts) | {n_nodes} |")
    lines.append(f"| Edges (relationships) | {n_edges} |")
    lines.append(f"| Hyperedges | {n_hyperedges} |")
    lines.append(f"| Leiden communities | {n_communities} |")
    lines.append(f"| Node-to-file mapping | {cov['mapped']}/{cov['total']} mapped ({cov['unmapped']} unmapped) |")

    conf_total = confidence["total"]
    for label in ["EXTRACTED", "INFERRED", "AMBIGUOUS"]:
        count = conf_total.get(label, 0)
        pct = round(100 * count / n_edges, 1) if n_edges else 0
        lines.append(f"| Edges: {label} | {count} ({pct}%) |")

    lines.append("")

    # --- Section 2: God nodes ---
    lines.append("## 2. God Nodes")
    lines.append("")
    lines.append("High-degree hub concepts that connect many other nodes.")
    lines.append("")
    lines.append("| Rank | Label | Degree | Topic Maps | Assessment |")
    lines.append("|------|-------|--------|------------|------------|")
    for i, gn in enumerate(god_nodes, 1):
        topics_str = ", ".join(gn["topic_maps"]) if gn["topic_maps"] else "none"
        lines.append(f"| {i} | {gn['label']} | {gn['degree']} | {topics_str} | {gn['assessment']} |")
    lines.append("")

    # --- Section 3: Community-topic map alignment ---
    lines.append("## 3. Community-Topic Map Alignment")
    lines.append("")
    lines.append("Each Leiden community mapped to existing topic maps.")
    lines.append("")
    lines.append("| Community | Nodes | Files | Primary Topic Map | Fraction | Type |")
    lines.append("|-----------|-------|-------|-------------------|----------|------|")
    for c in alignment["communities"]:
        lines.append(
            f"| {c['community_id']} | {c['node_count']} | {c['file_count']} | "
            f"{c['primary_topic_map']} | {c['primary_fraction']:.0%} | {c['alignment_type']} |"
        )
    lines.append("")

    if alignment["splits"]:
        lines.append("### Topic Maps Split Across Communities")
        lines.append("")
        lines.append("These topic maps have notes scattered across 3+ communities (may be too broad):")
        lines.append("")
        for topic, cids in sorted(alignment["splits"].items()):
            lines.append(f"- **{topic}**: communities {', '.join(str(c) for c in cids)}")
        lines.append("")

    # --- Section 4: Surprising connections ---
    lines.append("## 4. Top Surprising Cross-Topic Connections")
    lines.append("")
    lines.append("Edges between notes in different topic maps, ranked by surprise score.")
    lines.append("")
    lines.append("| Source | Target | Score | Confidence | Already Linked? | Recommendation |")
    lines.append("|--------|--------|-------|------------|-----------------|----------------|")
    for s in surprises:
        src_topics = ", ".join(s["source_topic_maps"]) if s["source_topic_maps"] else "?"
        tgt_topics = ", ".join(s["target_topic_maps"]) if s["target_topic_maps"] else "?"
        linked = "yes" if s["already_linked"] else "**no**"
        lines.append(
            f"| {s['source_node']} ({src_topics}) | {s['target_node']} ({tgt_topics}) | "
            f"{s['surprise_score']} | {s['confidence']} | {linked} | {s['recommendation']} |"
        )
    lines.append("")

    # --- Section 5: Confidence distribution ---
    lines.append("## 5. Confidence Distribution")
    lines.append("")
    if confidence["high_ambiguous_regions"]:
        lines.append("### High-Ambiguous Regions (need more explicit linking)")
        lines.append("")
        for topic, count in sorted(confidence["high_ambiguous_regions"].items(), key=lambda x: -x[1]):
            lines.append(f"- **{topic}**: {count} AMBIGUOUS edges")
        lines.append("")

    if confidence["cross_topic_inferred_pairs"]:
        lines.append("### Cross-Topic INFERRED Pairs (latent relationships worth formalizing)")
        lines.append("")
        for pair, count in confidence["cross_topic_inferred_pairs"].items():
            if isinstance(pair, tuple):
                lines.append(f"- **{pair[0]}** <-> **{pair[1]}**: {count} inferred edges")
            else:
                lines.append(f"- {pair}: {count} inferred edges")
        lines.append("")

    # --- Section 6: Structural recommendations ---
    lines.append("## 6. Structural Recommendations")
    lines.append("")

    # Generate recommendations from the data
    recs = []

    # From god nodes
    cross_cutting = [gn for gn in god_nodes if len(gn["topic_maps"]) > 1]
    if cross_cutting:
        recs.append(
            f"**Bridge notes needed**: {len(cross_cutting)} god nodes span multiple topic maps. "
            f"Consider creating explicit bridge notes for: "
            + ", ".join(gn["label"] for gn in cross_cutting[:5])
        )

    uncurated = [gn for gn in god_nodes if not gn["topic_maps"]]
    if uncurated:
        recs.append(
            f"**Curation gaps**: {len(uncurated)} high-degree nodes are not in any topic map: "
            + ", ".join(gn["label"] for gn in uncurated[:5])
        )

    # From alignment
    if alignment["splits"]:
        recs.append(
            f"**Topic map splits to consider**: {len(alignment['splits'])} topic maps have notes "
            f"scattered across 3+ communities: {', '.join(alignment['splits'].keys())}"
        )

    merge_communities = [c for c in alignment["communities"] if c["alignment_type"] == "merge"]
    if merge_communities:
        recs.append(
            f"**Merge candidates**: {len(merge_communities)} communities span multiple topic maps equally — "
            f"those topic maps may be over-partitioned"
        )

    # From surprising connections
    new_links = [s for s in surprises if not s["already_linked"]]
    if new_links:
        recs.append(
            f"**Missing wiki-links**: {len(new_links)} high-scoring cross-topic connections "
            f"have no explicit link in the vault"
        )

    for rec in recs:
        lines.append(f"- {rec}")

    if not recs:
        lines.append("No structural issues detected — curation aligns well with emergent structure.")

    lines.append("")

    # --- Section 7: What to steal ---
    lines.append("## 7. Patterns Worth Adopting from Graphify")
    lines.append("")
    lines.append("1. **Confidence labeling on edges**: arscontexta could distinguish EXTRACTED (Shachar explicitly linked) vs INFERRED (found by retrieval/search) connections in topic maps")
    lines.append("2. **SHA256 caching for incremental updates**: graphify's cache pattern (`graphify-out/cache/`) could inform a vault ingest cache to avoid re-processing unchanged notes")
    lines.append("3. **The `--watch` pattern**: auto-updating an index when files change — relevant to the missing lint operation identified in vault health audits")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Graphify diagnostic: compare emergent graph structure against arscontexta topic maps"
    )
    parser.add_argument(
        "--graph-json",
        type=Path,
        default=Path("graphify-out/graph.json"),
        help="Path to graphify graph.json (default: graphify-out/graph.json)",
    )
    parser.add_argument(
        "--topic-map-index",
        type=Path,
        default=Path("ops/cache/topic-map-index.json"),
        help="Path to topic-map-index.json (default: ops/cache/topic-map-index.json)",
    )
    parser.add_argument(
        "--notes-dir",
        type=Path,
        default=Path("notes"),
        help="Path to vault notes directory (default: notes/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("graphify-out/arscontexta-graphify-diagnostic.md"),
        help="Output path for diagnostic report",
    )
    parser.add_argument(
        "--top-god-nodes",
        type=int,
        default=15,
        help="Number of god nodes to report (default: 15)",
    )
    parser.add_argument(
        "--top-surprises",
        type=int,
        default=15,
        help="Number of surprising connections to report (default: 15)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.graph_json.exists():
        print(f"ERROR: graph.json not found at {args.graph_json}", file=sys.stderr)
        print("Run /graphify first to generate the graph.", file=sys.stderr)
        sys.exit(1)

    if not args.topic_map_index.exists():
        print(f"ERROR: topic-map-index.json not found at {args.topic_map_index}", file=sys.stderr)
        print("Run: node ops/scripts/topic-map-index.mjs", file=sys.stderr)
        sys.exit(1)

    if not args.notes_dir.is_dir():
        print(f"ERROR: notes directory not found at {args.notes_dir}", file=sys.stderr)
        sys.exit(1)

    # Load data
    print(f"Loading graph from {args.graph_json}...")
    graph = load_graphify_graph(args.graph_json)
    print(f"  {len(graph['nodes'])} nodes, {len(graph['links'])} edges, "
          f"{len(graph['community_to_nodes'])} communities")

    print(f"Loading topic map index from {args.topic_map_index}...")
    tmi = load_topic_map_index(args.topic_map_index)
    print(f"  {len(tmi['moc_names'])} MOCs, {len(tmi['note_to_topics'])} notes")

    # Map nodes to topic maps
    print("Mapping graphify nodes to topic maps...")
    mapping = map_nodes_to_topic_maps(graph, tmi)
    cov = mapping["coverage"]
    print(f"  {cov['mapped']}/{cov['total']} nodes mapped ({cov['unmapped']} unmapped)")

    if mapping["unmapped_nodes"]:
        print(f"  Unmapped nodes (first 5): {mapping['unmapped_nodes'][:5]}")

    # Extract existing wiki-links
    print(f"Extracting wiki-links from {args.notes_dir}...")
    existing_links = extract_existing_wikilinks(args.notes_dir)
    total_links = sum(len(v) for v in existing_links.values())
    print(f"  {len(existing_links)} files, {total_links} wiki-links")

    # Run analyses
    print("Analyzing god nodes...")
    gods = analyze_god_nodes(graph, tmi, mapping, top_n=args.top_god_nodes)

    print("Analyzing community-topic map alignment...")
    alignment = analyze_community_alignment(graph, tmi, mapping)

    print("Analyzing surprising connections...")
    surprises = analyze_surprising_connections(
        graph, tmi, mapping, existing_links, top_n=args.top_surprises
    )

    print("Analyzing confidence distribution...")
    confidence = analyze_confidence_distribution(graph, mapping)

    # Generate report
    print("Generating diagnostic report...")
    report = generate_report(graph, tmi, mapping, gods, alignment, surprises, confidence)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"\nDiagnostic report written to {args.output}")

    # Print key findings summary
    print("\n--- Key Findings ---")
    cross_cutting = [gn for gn in gods if len(gn["topic_maps"]) > 1]
    new_links = [s for s in surprises if not s["already_linked"]]
    print(f"  God nodes: {len(gods)} (top {args.top_god_nodes}), {len(cross_cutting)} cross-cutting")
    print(f"  Communities: {len(alignment['communities'])}, {len(alignment['splits'])} topic maps split across 3+ communities")
    print(f"  Surprising connections: {len(surprises)}, {len(new_links)} not yet linked")
    print(f"  Confidence: {confidence['total']}")


if __name__ == "__main__":
    main()
