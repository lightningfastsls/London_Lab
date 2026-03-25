#!/usr/bin/env node
/**
 * Topic Map Index Generator
 * Parses the vault's topic maps and note frontmatter into a static JSON index.
 *
 * Usage: node topic-map-index.mjs [notes-dir] [output-file]
 * Defaults: notes-dir = ../../notes, output = ../cache/topic-map-index.json
 */

import { readdir, readFile, writeFile, mkdir } from 'node:fs/promises';
import { join, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DEFAULT_NOTES_DIR = join(__dirname, '..', '..', 'notes');
const DEFAULT_OUTPUT = join(__dirname, '..', 'cache', 'topic-map-index.json');

const notesDir = process.argv[2] || DEFAULT_NOTES_DIR;
const outputFile = process.argv[3] || DEFAULT_OUTPUT;

// --- Frontmatter parser ---
function parseFrontmatter(content) {
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match) return {};
  const fm = {};
  const lines = match[1].split(/\r?\n/);
  let currentKey = null;
  let inArray = false;

  for (const line of lines) {
    // Array item: "  - value"
    if (inArray && /^\s+-\s+/.test(line)) {
      const val = line.replace(/^\s+-\s+/, '').replace(/^"|"$/g, '').trim();
      if (!Array.isArray(fm[currentKey])) fm[currentKey] = [];
      fm[currentKey].push(val);
      continue;
    }
    // Key: value
    const kvMatch = line.match(/^(\w[\w_]*)\s*:\s*(.*)/);
    if (kvMatch) {
      currentKey = kvMatch[1];
      const val = kvMatch[2].replace(/^"|"$/g, '').trim();
      if (val === '' || val === '[]') {
        fm[currentKey] = val === '[]' ? [] : '';
        inArray = val === '';
      } else {
        fm[currentKey] = val;
        inArray = false;
      }
    } else if (line.trim() === '' && currentKey) {
      inArray = false;
    }
  }
  return fm;
}

// --- Extract [[wiki-links]] from a line ---
function extractLinks(text) {
  const matches = [...text.matchAll(/\[\[([^\]]+)\]\]/g)];
  return matches.map(m => m[1]);
}

// --- Extract topics from frontmatter (handles string and array) ---
function extractTopics(fm) {
  const topics = [];
  if (typeof fm.topics === 'string') {
    topics.push(...extractLinks(fm.topics));
  } else if (Array.isArray(fm.topics)) {
    for (const t of fm.topics) {
      topics.push(...extractLinks(t));
    }
  }
  return topics;
}

// --- Parse a MOC file ---
function parseMOC(content, filename) {
  const fm = parseFrontmatter(content);
  const lines = content.split(/\r?\n/);
  const sections = {};
  const allNotes = new Set();
  const contextPhrases = {};
  const crossRefs = new Set();
  let currentSection = null;

  // Skip frontmatter
  let i = 0;
  if (lines[0] === '---') {
    i = 1;
    while (i < lines.length && lines[i] !== '---') i++;
    i++; // skip closing ---
  }

  const mocName = basename(filename, '.md');

  for (; i < lines.length; i++) {
    const line = lines[i];

    // Section header
    const sectionMatch = line.match(/^## (.+)/);
    if (sectionMatch) {
      currentSection = sectionMatch[1].trim();
      if (!sections[currentSection]) sections[currentSection] = [];
      continue;
    }

    // Wiki-link line: "- [[title]] -- context" or "- [[title]] — context"
    const links = extractLinks(line);
    for (const link of links) {
      // Is this link a MOC reference or a note?
      // We'll classify later; for now collect everything
      const contextMatch = line.match(
        new RegExp(`\\[\\[${escapeRegex(link)}\\]\\]\\s*(?:--|—|–)\\s*(.+)`)
      );
      if (contextMatch) {
        contextPhrases[link] = contextMatch[1].trim();
      }

      if (currentSection) {
        sections[currentSection].push(link);
      }
      allNotes.add(link);
    }
  }

  return {
    description: fm.description || '',
    parent_map: fm.parent_map || null,
    sections,
    all_notes: [...allNotes],
    context_phrases: contextPhrases,
    cross_refs: [] // filled in post-processing
  };
}

function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// --- Parse index.md hierarchy ---
function parseIndexHierarchy(content) {
  const hierarchy = {};
  const lines = content.split(/\r?\n/);
  let inTopicMaps = false;
  let lastTopLevel = null;

  for (const line of lines) {
    if (line.startsWith('## Topic Maps')) {
      inTopicMaps = true;
      continue;
    }
    if (inTopicMaps && line.startsWith('## ') && !line.startsWith('## Topic Maps')) {
      break;
    }
    if (!inTopicMaps) continue;

    const links = extractLinks(line);
    if (links.length === 0) continue;

    const indent = line.search(/\S/);
    const mapName = links[0];

    if (indent < 2) {
      // Top-level entry (starts at column 0: "- [[name]]")
      hierarchy[mapName] = { parent: 'index', children: [] };
      lastTopLevel = mapName;
    } else if (lastTopLevel) {
      // Child entry (indented)
      hierarchy[mapName] = { parent: lastTopLevel, children: [] };
      if (!hierarchy[lastTopLevel]) {
        hierarchy[lastTopLevel] = { parent: 'index', children: [] };
      }
      hierarchy[lastTopLevel].children.push(mapName);
    }
  }

  // Add index itself
  hierarchy['index'] = { parent: null, children: Object.keys(hierarchy).filter(k => hierarchy[k].parent === 'index') };

  return hierarchy;
}

// --- Main ---
async function main() {
  const files = (await readdir(notesDir)).filter(f => f.endsWith('.md'));
  const t0 = Date.now();

  // Read all files
  const fileContents = {};
  await Promise.all(files.map(async f => {
    fileContents[f] = await readFile(join(notesDir, f), 'utf-8');
  }));

  // Identify MOCs
  const mocFiles = new Set();
  const noteFiles = new Set();
  for (const [f, content] of Object.entries(fileContents)) {
    const fm = parseFrontmatter(content);
    if (fm.type === 'moc') {
      mocFiles.add(f);
    } else {
      noteFiles.add(f);
    }
  }

  // Parse hierarchy from index.md
  const hierarchy = parseIndexHierarchy(fileContents['index.md'] || '');

  // Parse all MOCs
  const maps = {};
  const mocNames = new Set([...mocFiles].map(f => basename(f, '.md')));

  for (const f of mocFiles) {
    const name = basename(f, '.md');
    const parsed = parseMOC(fileContents[f], f);

    // Separate note links from MOC cross-refs
    const noteLinks = new Set();
    const crossRefs = new Set();
    for (const link of parsed.all_notes) {
      if (mocNames.has(link) && link !== name) {
        crossRefs.add(link);
      } else if (!mocNames.has(link)) {
        noteLinks.add(link);
      }
    }

    // Clean sections: remove MOC-only links, keep note links
    const cleanSections = {};
    for (const [sec, links] of Object.entries(parsed.sections)) {
      const noteOnly = links.filter(l => !mocNames.has(l));
      if (noteOnly.length > 0) {
        cleanSections[sec] = noteOnly;
      }
    }

    maps[name] = {
      description: parsed.description,
      sections: cleanSections,
      all_notes: [...noteLinks],
      context_phrases: parsed.context_phrases,
      cross_refs: [...crossRefs]
    };
  }

  // Parse all notes
  const notes = {};
  for (const f of noteFiles) {
    const name = basename(f, '.md');
    const fm = parseFrontmatter(fileContents[f]);
    notes[name] = {
      description: fm.description || '',
      type: fm.type || '',
      confidence: fm.confidence || '',
      topics: extractTopics(fm)
    };
  }

  // Build index
  const index = {
    generated: new Date().toISOString(),
    stats: {
      moc_count: mocFiles.size,
      note_count: noteFiles.size,
      generation_ms: Date.now() - t0
    },
    hierarchy,
    maps,
    notes
  };

  // Write output
  await mkdir(dirname(outputFile), { recursive: true });
  await writeFile(outputFile, JSON.stringify(index, null, 2));

  const elapsed = Date.now() - t0;
  console.log(`Index generated: ${mocFiles.size} MOCs, ${noteFiles.size} notes in ${elapsed}ms`);
  console.log(`Output: ${outputFile}`);
}

main().catch(err => {
  console.error('Index generation failed:', err.message);
  process.exit(1);
});
