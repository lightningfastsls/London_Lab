#!/usr/bin/env python3
"""Corpus canary hook — enforces the canonical parameter registry.

Fires on:
  * SessionStart — always emits a primer pointing at corpus.py + corpus_facts/*.json
  * PreToolUse (Edit|Write|MultiEdit) — emits a decision-tree warning when the
    edit payload contains a canonical name or value from the corpus

Contract: reads one JSON object from stdin, writes one JSON object to stdout
with hookSpecificOutput.additionalContext. Always exits 0 (fail-open) so a
broken hook cannot block Claude. See docs/modules/corpus-constants.md.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("CLAUDE_PROJECT_DIR", str(Path(__file__).resolve().parents[2]))
)
CORPUS_PY = PROJECT_ROOT / "src" / "usv_spectrogram" / "corpus.py"
CORPUS_FACTS_DIR = PROJECT_ROOT / "data" / "corpus_facts"

LAYER2_PARAMETER_SECTIONS = ("timing", "bout_detection_a2", "counts", "references")

LAYER1_ALIASES = {
    "sample_rate", "sr", "fs",
    "freq_min", "freq_min_hz",
    "freq_max", "freq_max_hz",
    "n_fft", "hop", "hop_length",
}

FLOAT_TOL = 0.001
MAX_MATCHES_SHOWN = 10

SESSION_PRIMER = """[CORPUS-INVARIANT] This repo uses a canonical parameter system.
  Layer 1 physical constants : src/usv_spectrogram/corpus.py
  Layer 2 empirical facts    : data/corpus_facts/{dataset}.json
  Module doc                 : docs/modules/corpus-constants.md

Before reasoning about, writing, or citing any shared parameter — sample
rate, USV band, STFT, ICI, bout threshold, call duration, transition MI —
consult these files. Never invent values from memory. Never redeclare
canonical constants; import them."""

EDIT_WARNING_TEMPLATE = """[CORPUS-INVARIANT] This edit declares or consumes parameters that
match canonical names or values in the corpus. Pick one of A/B/C/D in your
next response before proceeding:

(A) Using EXISTING physical constants (sample rate, USV freq, STFT)
    -> Import from src/usv_spectrogram/corpus.py — do NOT redeclare.
       SAMPLE_RATE_HZ | USV_FREQ_MIN_HZ | USV_FREQ_MAX_HZ
       STFT_N_FFT | STFT_HOP

(B) Using EXISTING empirical facts (ICI, bout, call counts)
    -> Read from data/corpus_facts/<dataset>.json — do NOT re-derive.

(C) Introducing a NEW parameter
    -> Classify: physical fact (Layer 1) or empirical measurement (Layer 2)?
       Layer 1: add constant to corpus.py + update corpus-constants.md
                + add drift assertion if CNN-relevant
       Layer 2: add key to scripts/audit_corpus.py + regenerate JSON

(D) Unrelated edit (docstring, typo, logging, etc.)
    -> Proceed.

Matched tokens in your edit: {matched}
Doc: docs/modules/corpus-constants.md
CNN FREEZE: ExtractionConfig.freq_{{min,max}}_hz require CNN retrain."""


def load_canonical_registry() -> tuple[set[str], set[float]]:
    """Scan corpus.py + corpus_facts/*.json. Return (names, values)."""
    names: set[str] = set(LAYER1_ALIASES)
    values: set[float] = set()

    if CORPUS_PY.exists():
        try:
            tree = ast.parse(CORPUS_PY.read_text())
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in tree.body:
                _harvest_module_constant(node, names, values)

    if CORPUS_FACTS_DIR.exists():
        for jf in sorted(CORPUS_FACTS_DIR.glob("*.json")):
            try:
                data = json.loads(jf.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, dict):
                continue
            for section in LAYER2_PARAMETER_SECTIONS:
                sec = data.get(section)
                if isinstance(sec, dict):
                    _walk_dict(sec, names, values)

    return names, values


def _harvest_module_constant(
    node: ast.stmt, names: set[str], values: set[float]
) -> None:
    """Pull UPPER_CASE numeric constants out of an AST module-level statement."""
    if isinstance(node, ast.AnnAssign):
        target, value = node.target, node.value
        if value is None or not isinstance(target, ast.Name):
            return
        ident = target.id
        if not ident.isupper():
            return
        _record_literal(ident, value, names, values)
    elif isinstance(node, ast.Assign):
        if not isinstance(node.value, (ast.Constant, ast.UnaryOp)):
            return
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id.isupper():
                _record_literal(tgt.id, node.value, names, values)


def _record_literal(
    ident: str, value_node: ast.expr, names: set[str], values: set[float]
) -> None:
    try:
        v = ast.literal_eval(value_node)
    except (ValueError, TypeError, SyntaxError):
        return
    if isinstance(v, bool):
        return
    if isinstance(v, (int, float)):
        names.add(ident)
        names.add(ident.lower())
        values.add(float(v))


SMALL_INT_THRESHOLD = 100


def _add_numeric(val, values: set[float]) -> None:
    """Add a numeric value to the registry, skipping small integers.

    Small ints (|v| < SMALL_INT_THRESHOLD) drive false positives: counts like
    `n_sessions=2` or `n_files=19` are canonical but the bare literal `2`
    appears in unrelated code constantly. Floats at any magnitude stay in —
    `0.22` as a Hertz reference is still a surgical match.
    """
    if isinstance(val, bool):
        return
    if isinstance(val, int):
        if abs(val) >= SMALL_INT_THRESHOLD:
            values.add(float(val))
    elif isinstance(val, float):
        values.add(val)


def _walk_dict(d: dict, names: set[str], values: set[float]) -> None:
    for key, val in d.items():
        if isinstance(key, str):
            names.add(key)
            names.add(key.lower())
        if isinstance(val, dict):
            _walk_dict(val, names, values)
        elif isinstance(val, list):
            for item in val:
                _add_numeric(item, values)
        else:
            _add_numeric(val, values)


IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
NUM_RE = re.compile(r"-?\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?")


def match_canonical(
    text: str, names: set[str], values: set[float]
) -> list[str]:
    """Return canonical tokens found in the payload (deduped, order-preserving)."""
    matches: list[str] = []

    for ident in IDENT_RE.findall(text):
        if ident in names:
            matches.append(ident)

    for raw in NUM_RE.findall(text):
        try:
            num = float(raw.replace("_", ""))
        except ValueError:
            continue
        for v in values:
            if v == 0.0:
                if num == 0.0:
                    matches.append(raw)
                    break
            elif abs(num - v) / abs(v) < FLOAT_TOL:
                if num == v:
                    matches.append(raw)
                else:
                    matches.append(f"{raw} (~= {v:g})")
                break

    seen: set[str] = set()
    out: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def extract_payload_text(tool_name: str, tool_input: dict) -> str:
    """Combine user-authored text from a tool_input (both old and new strings)."""
    parts: list[str] = []
    if tool_name == "Edit":
        parts.append(str(tool_input.get("old_string", "")))
        parts.append(str(tool_input.get("new_string", "")))
    elif tool_name == "Write":
        parts.append(str(tool_input.get("content", "")))
    elif tool_name == "MultiEdit":
        for edit in tool_input.get("edits") or []:
            if not isinstance(edit, dict):
                continue
            parts.append(str(edit.get("old_string", "")))
            parts.append(str(edit.get("new_string", "")))
    return "\n".join(parts)


def emit_context(additional_context: str, event_name: str) -> None:
    payload = {
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "additionalContext": additional_context,
        }
    }
    sys.stdout.write(json.dumps(payload))


def handle_session_start(_event: dict) -> None:
    emit_context(SESSION_PRIMER, "SessionStart")


def handle_pre_tool_use(event: dict) -> None:
    tool_name = event.get("tool_name", "")
    if tool_name not in {"Edit", "Write", "MultiEdit"}:
        return
    tool_input = event.get("tool_input")
    if not isinstance(tool_input, dict):
        return
    text = extract_payload_text(tool_name, tool_input)
    if not text.strip():
        return
    names, values = load_canonical_registry()
    matched = match_canonical(text, names, values)
    if not matched:
        return
    shown = ", ".join(matched[:MAX_MATCHES_SHOWN])
    if len(matched) > MAX_MATCHES_SHOWN:
        shown += f", ... ({len(matched) - MAX_MATCHES_SHOWN} more)"
    emit_context(
        EDIT_WARNING_TEMPLATE.format(matched=shown),
        "PreToolUse",
    )


def main() -> int:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return 0
        event = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"corpus_canary: bad stdin: {e}", file=sys.stderr)
        return 0

    if not isinstance(event, dict):
        return 0

    try:
        hook_event = event.get("hook_event_name", "")
        if hook_event == "SessionStart":
            handle_session_start(event)
        elif hook_event == "PreToolUse":
            handle_pre_tool_use(event)
    except Exception as e:
        print(f"corpus_canary: {type(e).__name__}: {e}", file=sys.stderr)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
