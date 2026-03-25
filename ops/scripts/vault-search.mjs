#!/usr/bin/env node
/**
 * Vault Search Engine
 * 3-layer search: topic map routing → ripgrep content → merge & rank
 *
 * Usage:
 *   node vault-search.mjs --query "search terms" [--context "additional context"] [--limit N]
 *   node vault-search.mjs --mode find-related --note "note title"
 *   node vault-search.mjs --mode dedup-check --title "proposed title"
 */

import { readFile } from 'node:fs/promises';
import { execSync } from 'node:child_process';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const VAULT_ROOT = join(__dirname, '..', '..');
const INDEX_FILE = join(__dirname, '..', 'cache', 'topic-map-index.json');
const NOTES_DIR = join(VAULT_ROOT, 'notes');

// --- Scoring weights (tune here) ---
const WEIGHTS = {
  // Layer 1: Topic map structural matching
  topicMapNameMatch: 3,
  topicMapDescMatch: 2,
  topicMapSectionMatch: 1.5,
  noteInMatchedSection: 3,
  noteContextPhraseMatch: 1,
  crossRefHop: 0.5,
  multiMapAppearance: 1,

  // Layer 2: Ripgrep content matching
  ripgrepBaseMatch: 2,
  ripgrepPerExtraTerm: 1,

  // Layer 3: Metadata bonuses
  noteTitleMatch: 0.5,   // per query token found in note title (only when 2+ match)
  descriptionMatch: 0.5,
};

// Compound terms to preserve during tokenization
const COMPOUND_TERMS = [
  'vq-vae', 'deepsqueak', 'stft', 'usv', 'lmt', 'rlhf', 'ppo', 'grpo',
  'dpo', 'icl', 'lora', 'peft', 'cnn', 'mcp', '300-khz', 'k-means',
  'raven', 'deep-squeak', 'bootssnap', 'topic-map', 'wiki-link',
  'entropy-rate', 'zipf', 'codebook', 'spectrogram', 'bout',
];

// Status words to strip from queries
const STATUS_WORDS = new Set([
  'done', 'complete', 'completed', 'in', 'progress', 'phase', 'blocked',
  'todo', 'pending', 'started', 'finished', 'n', 'the', 'a', 'an',
  'is', 'are', 'was', 'were', 'for', 'of', 'to', 'and', 'or', 'with',
  'from', 'by', 'at', 'on', 'that', 'this', 'it', 'its', 'has', 'have',
]);

// --- Tokenizer ---
function tokenize(text) {
  if (!text) return [];
  let lower = text.toLowerCase();

  // Preserve compound terms
  const preserved = [];
  for (const term of COMPOUND_TERMS) {
    const variants = [term, term.replace(/-/g, ' '), term.replace(/-/g, '')];
    for (const v of variants) {
      if (lower.includes(v)) {
        preserved.push(term);
        // Remove from text to avoid double-counting
        lower = lower.replace(new RegExp(escapeRegex(v), 'g'), ' ');
        break;
      }
    }
  }

  // Tokenize remaining
  const words = lower
    .replace(/[^a-z0-9\s-]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 1 && !STATUS_WORDS.has(w));

  return [...new Set([...preserved, ...words])];
}

function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// --- Load index ---
async function loadIndex() {
  try {
    const raw = await readFile(INDEX_FILE, 'utf-8');
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

// --- Layer 1: Topic Map Routing ---
function topicMapSearch(index, tokens, limit) {
  if (!index) return [];

  const results = new Map(); // note -> { score, sources }

  // Score each topic map by keyword overlap
  const mapScores = [];
  for (const [mapName, mapData] of Object.entries(index.maps)) {
    let score = 0;
    const matchedTokens = new Set();

    // Map name matches
    const nameTokens = tokenize(mapName);
    for (const t of tokens) {
      if (nameTokens.some(nt => nt.includes(t) || t.includes(nt))) {
        score += WEIGHTS.topicMapNameMatch;
        matchedTokens.add(t);
      }
    }

    // Description matches
    const descTokens = tokenize(mapData.description);
    for (const t of tokens) {
      if (descTokens.some(dt => dt.includes(t) || t.includes(dt))) {
        score += WEIGHTS.topicMapDescMatch;
        matchedTokens.add(t);
      }
    }

    // Section header matches
    for (const section of Object.keys(mapData.sections)) {
      const secTokens = tokenize(section);
      for (const t of tokens) {
        if (secTokens.some(st => st.includes(t) || t.includes(st))) {
          score += WEIGHTS.topicMapSectionMatch;
          matchedTokens.add(t);
        }
      }
    }

    if (score > 0) {
      mapScores.push({ name: mapName, score, matchedTokens, data: mapData });
    }
  }

  // Select top 3 maps
  mapScores.sort((a, b) => b.score - a.score);
  const topMaps = mapScores.slice(0, 3);

  // Collect notes from top maps, scoring by section relevance
  for (const map of topMaps) {
    // Score sections within this map
    const sectionScores = [];
    for (const [secName, secNotes] of Object.entries(map.data.sections)) {
      const secTokens = tokenize(secName);
      let secScore = 0;
      for (const t of tokens) {
        if (secTokens.some(st => st.includes(t) || t.includes(st))) {
          secScore += 2;
        }
      }
      sectionScores.push({ name: secName, notes: secNotes, score: secScore });
    }
    sectionScores.sort((a, b) => b.score - a.score);

    // Add notes from matched sections — score scales with section keyword relevance
    for (const sec of sectionScores) {
      // Section score 0 = no keywords matched in header → minimal base score
      // Section score 4+ = strong match → full score + bonus
      const sectionBonus = sec.score > 0
        ? WEIGHTS.noteInMatchedSection + sec.score * 0.5
        : WEIGHTS.noteInMatchedSection * 0.3;
      for (const noteTitle of sec.notes) {
        const existing = results.get(noteTitle) || { score: 0, topic_maps: [], sections: [], source: new Set() };

        existing.score += sectionBonus;
        existing.topic_maps.push(map.name);
        existing.sections.push(sec.name);
        existing.source.add('topic-map');

        // Context phrase match bonus
        const ctxPhrase = map.data.context_phrases[noteTitle];
        if (ctxPhrase) {
          const ctxTokens = tokenize(ctxPhrase);
          for (const t of tokens) {
            if (ctxTokens.some(ct => ct.includes(t) || t.includes(ct))) {
              existing.score += WEIGHTS.noteContextPhraseMatch;
            }
          }
        }

        // Description match bonus (from notes index)
        const noteData = index.notes[noteTitle];
        if (noteData?.description) {
          const descTokens = tokenize(noteData.description);
          for (const t of tokens) {
            if (descTokens.some(dt => dt.includes(t) || t.includes(dt))) {
              existing.score += WEIGHTS.descriptionMatch;
            }
          }
        }

        results.set(noteTitle, existing);
      }
    }

    // Follow cross_refs 1 hop
    for (const xref of map.data.cross_refs) {
      const xrefMap = index.maps[xref];
      if (!xrefMap) continue;
      // Check section headers of cross-ref map — require 2+ token matches to avoid noise
      for (const [secName, secNotes] of Object.entries(xrefMap.sections)) {
        const secTokens = tokenize(secName);
        let matchCount = 0;
        for (const t of tokens) {
          if (secTokens.some(st => st.includes(t) || t.includes(st))) {
            matchCount++;
          }
        }
        if (matchCount >= 2) {
          for (const noteTitle of secNotes) {
            const existing = results.get(noteTitle) || { score: 0, topic_maps: [], sections: [], source: new Set() };
            existing.score += WEIGHTS.crossRefHop;
            existing.source.add('cross-ref');
            if (!existing.topic_maps.includes(xref)) {
              existing.topic_maps.push(xref);
              existing.sections.push(secName);
            }
            results.set(noteTitle, existing);
          }
        }
      }
    }
  }

  // Multi-map appearance bonus
  for (const [noteTitle, data] of results) {
    const uniqueMaps = new Set(data.topic_maps);
    if (uniqueMaps.size > 1) {
      data.score += WEIGHTS.multiMapAppearance * (uniqueMaps.size - 1);
    }
  }

  return results;
}

// --- Layer 2: Ripgrep content search ---
function ripgrepSearch(tokens, limit) {
  const results = new Map(); // filename -> Set of matched tokens
  if (tokens.length === 0) return new Map();

  // Run one rg per token (N tokens = N calls, much better than N*M)
  for (const token of tokens) {
    try {
      const cmd = `rg --files-with-matches --ignore-case "${escapeRegex(token)}" "${NOTES_DIR}" 2>/dev/null || true`;
      const output = execSync(cmd, { encoding: 'utf-8', timeout: 5000 });
      const files = output.trim().split('\n').filter(Boolean);
      for (const filepath of files) {
        const filename = filepath.split('/').pop().replace('.md', '');
        if (!results.has(filename)) results.set(filename, new Set());
        results.get(filename).add(token);
      }
    } catch { /* rg failed for this token */ }
  }

  // Convert to scored results
  const scored = new Map();
  for (const [filename, matchedTokens] of results) {
    const termMatches = matchedTokens.size;
    const score = WEIGHTS.ripgrepBaseMatch + WEIGHTS.ripgrepPerExtraTerm * Math.max(0, termMatches - 1);
    scored.set(filename, { score, termMatches, source: new Set(['ripgrep']) });
  }
  return scored;
}

// --- Layer 3: Merge & Rank ---
function mergeAndRank(topicResults, rgResults, index, tokens, limit) {
  const merged = new Map();

  // Add topic map results
  for (const [title, data] of topicResults) {
    merged.set(title, { ...data });
  }

  // Merge ripgrep results
  for (const [title, rgData] of rgResults) {
    const existing = merged.get(title);
    if (existing) {
      existing.score += rgData.score;
      existing.source = new Set([...existing.source, ...rgData.source]);
    } else {
      merged.set(title, {
        score: rgData.score,
        topic_maps: [],
        sections: [],
        source: rgData.source
      });
    }
  }

  // Note title matching bonus — only when 2+ query tokens appear in the title
  for (const [title, data] of merged) {
    const titleTokens = tokenize(title);
    let titleMatches = 0;
    for (const t of tokens) {
      if (titleTokens.some(tt => tt.includes(t) || t.includes(tt))) {
        titleMatches++;
      }
    }
    if (titleMatches >= 2) {
      data.score += WEIGHTS.noteTitleMatch * titleMatches;
    }
  }

  // Build sorted results
  const sorted = [...merged.entries()]
    .sort(([, a], [, b]) => b.score - a.score)
    .slice(0, limit);

  // Enrich with metadata
  return sorted.map(([title, data]) => {
    const noteData = index?.notes[title] || {};
    return {
      note: title,
      description: noteData.description || '',
      type: noteData.type || '',
      confidence: noteData.confidence || '',
      source: [...data.source].join('+'),
      topic_map: data.topic_maps[0] || '',
      section: data.sections[0] || '',
      context_phrase: index?.maps[data.topic_maps[0]]?.context_phrases[title] || '',
      score: Math.round(data.score * 10) / 10
    };
  });
}

// --- Mode: find-related ---
function findRelated(index, noteTitle, limit) {
  const noteData = index.notes[noteTitle];
  if (!noteData) {
    // Try fuzzy match
    const lower = noteTitle.toLowerCase();
    const match = Object.keys(index.notes).find(k => k.toLowerCase().includes(lower));
    if (match) return findRelated(index, match, limit);
    return [];
  }

  const results = new Map();
  const parentMaps = noteData.topics || [];

  for (const mapName of parentMaps) {
    const mapData = index.maps[mapName];
    if (!mapData) continue;

    // Find which section this note is in
    for (const [secName, secNotes] of Object.entries(mapData.sections)) {
      const isInSection = secNotes.includes(noteTitle);

      for (const sibling of secNotes) {
        if (sibling === noteTitle) continue;
        const existing = results.get(sibling) || { score: 0, via: [] };
        existing.score += isInSection ? 2 : 1; // same section vs same map
        existing.via.push(`${mapName} > ${secName}`);
        results.set(sibling, existing);
      }
    }

    // Cross-refs 1 hop
    for (const xref of mapData.cross_refs) {
      const xrefMap = index.maps[xref];
      if (!xrefMap) continue;
      for (const [secName, secNotes] of Object.entries(xrefMap.sections)) {
        for (const note of secNotes) {
          if (note === noteTitle) continue;
          const existing = results.get(note) || { score: 0, via: [] };
          existing.score += WEIGHTS.crossRefHop;
          existing.via.push(`${xref} > ${secName} (cross-ref)`);
          results.set(note, existing);
        }
      }
    }
  }

  return [...results.entries()]
    .sort(([, a], [, b]) => b.score - a.score)
    .slice(0, limit)
    .map(([title, data]) => ({
      note: title,
      description: index.notes[title]?.description || '',
      score: data.score,
      via: data.via[0] || ''
    }));
}

// --- Mode: dedup-check ---
function dedupCheck(index, proposedTitle, tokens) {
  if (!tokens) tokens = tokenize(proposedTitle);
  const results = [];

  for (const [title, noteData] of Object.entries(index.notes)) {
    const titleTokens = tokenize(title);
    // Count shared tokens
    const shared = tokens.filter(t => titleTokens.some(tt => tt.includes(t) || t.includes(tt)));
    const overlap = shared.length / Math.max(tokens.length, 1);

    if (overlap > 0.6) {
      results.push({
        note: title,
        description: noteData.description || '',
        overlap: Math.round(overlap * 100) + '%',
        shared_tokens: shared
      });
    }
  }

  // Also try ripgrep for the most distinctive token (longest)
  if (tokens.length > 0) {
    const distinctive = tokens.reduce((a, b) => a.length > b.length ? a : b);
    try {
      const cmd = `rg --files-with-matches --ignore-case "${escapeRegex(distinctive)}" "${NOTES_DIR}" 2>/dev/null | head -10 || true`;
      const output = execSync(cmd, { encoding: 'utf-8', timeout: 3000 });
      const files = output.trim().split('\n').filter(Boolean);
      for (const f of files) {
        const title = f.split('/').pop().replace('.md', '');
        if (!results.some(r => r.note === title)) {
          results.push({ note: title, description: '', overlap: 'rg-match', shared_tokens: [distinctive] });
        }
      }
    } catch { /* */ }
  }

  return results.sort((a, b) => {
    const aOverlap = parseInt(a.overlap) || 0;
    const bOverlap = parseInt(b.overlap) || 0;
    return bOverlap - aOverlap;
  }).slice(0, 10);
}

// --- CLI ---
async function main() {
  const args = process.argv.slice(2);
  const getArg = (name) => {
    const i = args.indexOf(name);
    return i >= 0 && i + 1 < args.length ? args[i + 1] : null;
  };

  const mode = getArg('--mode') || 'search';
  const limit = parseInt(getArg('--limit') || '5');

  const index = await loadIndex();
  if (!index) {
    console.error('Index not found. Run topic-map-index.mjs first.');
    process.exit(1);
  }

  if (mode === 'find-related') {
    const noteTitle = getArg('--note');
    if (!noteTitle) { console.error('--note required for find-related mode'); process.exit(1); }
    const results = findRelated(index, noteTitle, limit);
    console.log(JSON.stringify(results, null, 2));
    return;
  }

  if (mode === 'dedup-check') {
    const title = getArg('--title');
    if (!title) { console.error('--title required for dedup-check mode'); process.exit(1); }
    const results = dedupCheck(index, title);
    console.log(JSON.stringify(results, null, 2));
    return;
  }

  // Default: search mode
  const query = getArg('--query');
  const context = getArg('--context') || '';
  if (!query) { console.error('--query required'); process.exit(1); }

  const fullText = `${query} ${context}`;
  const tokens = tokenize(fullText);

  // Layer 1: Topic map routing
  const topicResults = topicMapSearch(index, tokens, limit * 3);

  // Layer 2: Ripgrep content search
  const rgResults = ripgrepSearch(tokens, limit * 3);

  // Filter out MOC files from ripgrep results
  const mocNames = new Set(Object.keys(index.maps));
  for (const key of rgResults.keys()) {
    if (mocNames.has(key)) rgResults.delete(key);
  }

  // Layer 3: Merge & rank
  const results = mergeAndRank(topicResults, rgResults, index, tokens, limit);

  console.log(JSON.stringify(results, null, 2));
}

main().catch(err => {
  console.error('Search failed:', err.message);
  process.exit(1);
});
