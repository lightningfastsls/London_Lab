"""Build a balanced noise dataset by sampling noise-labeled WAVs from each recording session.

Overview
--------
The USV Detection App labels files as "noise" and saves JSON metadata to
``USV_Detections/noise_labeled_files/``.  Each JSON contains the original WAV
path, which encodes the recording session (USV_1, USV_2, ...).

This script reads those labels, groups noise files by session, and copies a
configurable number of WAV files per session into an output directory.  This
ensures the noise dataset is representative of *all* recording conditions, not
just one session.

Path resolution
---------------
WAV files may be moved after labeling (e.g. into a ``_reviewed`` folder).
The script handles this with a three-tier lookup:

  1. Try the exact absolute path stored in the JSON.
  2. Try the session-relative path (``USV_N/subdir/file.wav``) under each
     search directory.
  3. Search ALL known WAV directories for a file with the same name.

This means labels are never "lost" as long as the WAV file still exists
somewhere under the repo.

Usage
-----
    # Dry run -- see what would be copied without touching disk:
    python scripts/build_noise_dataset.py --dry-run

    # Copy up to 20 noise files per session (default):
    python scripts/build_noise_dataset.py

    # Copy up to 50 per session to a custom output directory:
    python scripts/build_noise_dataset.py --per-session 50 --output datasets/noise

    # Copy ALL available noise files (no cap):
    python scripts/build_noise_dataset.py --per-session 0

    # Fix broken paths in existing noise label JSONs (updates in place):
    python scripts/build_noise_dataset.py --fix-paths
    python scripts/build_noise_dataset.py --fix-paths --dry-run   # preview only

Output structure
----------------
    <output_dir>/
        USV_1/
            2024-09-30_18-57-01_0002402.wav
            ...
        USV_2/
            ...
        manifest.json          # Tracks every file: source, session, copy time

The manifest lets you trace every noise sample back to its source session and
original path, even if files are later moved around.
"""

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def _build_filename_index(search_dirs: list[Path]) -> dict[str, list[Path]]:
    """Build a mapping of WAV filename -> list of full paths.

    Scans all search_dirs recursively.  This is the last-resort fallback when
    the stored absolute path and the session-relative path both fail (e.g.
    because the file was moved to a ``_reviewed`` folder after labeling).
    """
    index: dict[str, list[Path]] = {}
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for wav in search_dir.rglob("*.wav"):
            index.setdefault(wav.name, []).append(wav)
    return index


def _resolve_wav_path(
    wav_path: Path,
    search_dirs: list[Path],
    filename_index: dict[str, list[Path]],
) -> Path:
    """Find the actual WAV file using a three-tier lookup.

    1. Exact stored path.
    2. Session-relative path under each search dir (handles moves between
       sibling directories like USV_sample/).
    3. Filename-based search across all indexed WAVs (handles moves to
       _reviewed folders or any other reorganization).
    """
    # Tier 1: exact path
    if wav_path.exists():
        return wav_path

    # Tier 2: session-relative path under each search dir
    match = re.search(r"(USV_\d+[\\/].+)", str(wav_path))
    if match:
        relative_tail = Path(match.group(1))
        for search_dir in search_dirs:
            candidate = search_dir / relative_tail
            if candidate.exists():
                return candidate

    # Tier 3: filename-based search
    matches = filename_index.get(wav_path.name, [])
    if matches:
        return matches[0]

    return wav_path  # Return original (caller handles missing)


def find_noise_labels(
    noise_label_dir: Path,
    search_dirs: list[Path],
    filename_index: dict[str, list[Path]],
) -> list[dict]:
    """Read all noise label JSONs and return metadata records.

    Each record contains:
      - json_path:       path to the label JSON
      - wav_path:        resolved WAV file path
      - original_path:   the path stored in the JSON (for diagnostics)
      - session:         recording session name (e.g. "USV_1"), or "unknown"
      - filename:        WAV filename without directory
      - path_was_broken: True if the stored path didn't exist and was resolved
    """
    records = []
    for json_path in sorted(noise_label_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARNING: Could not read {json_path.name}: {e}")
            continue

        metadata = data.get("metadata", {})
        if metadata.get("file_label") != "noise":
            continue

        wav_path_str = metadata.get("wav_file", "")
        original_path = Path(wav_path_str)
        resolved_path = _resolve_wav_path(original_path, search_dirs, filename_index)
        path_was_broken = not original_path.exists() and resolved_path != original_path

        # Extract session name (USV_1, USV_2, ...) from the ORIGINAL path
        # (this is the true recording session, not where the file ended up)
        match = re.search(r"(USV_\d+)", wav_path_str)
        session = match.group(1) if match else "unknown"

        records.append({
            "json_path": json_path,
            "wav_path": resolved_path,
            "original_path": original_path,
            "session": session,
            "filename": resolved_path.name,
            "path_was_broken": path_was_broken,
        })

    return records


def group_by_session(records: list[dict]) -> dict[str, list[dict]]:
    """Group noise records by recording session."""
    groups = defaultdict(list)
    for rec in records:
        groups[rec["session"]].append(rec)
    return dict(sorted(groups.items()))


def sample_per_session(
    groups: dict[str, list[dict]], per_session: int
) -> dict[str, list[dict]]:
    """Limit each session to at most ``per_session`` files.

    If per_session is 0, all files are kept (no cap).
    Files are sorted by filename for deterministic selection.
    """
    sampled = {}
    for session, recs in groups.items():
        sorted_recs = sorted(recs, key=lambda r: r["filename"])
        if per_session > 0:
            sampled[session] = sorted_recs[:per_session]
        else:
            sampled[session] = sorted_recs
    return sampled


def copy_files(
    sampled: dict[str, list[dict]], output_dir: Path, dry_run: bool
) -> list[dict]:
    """Copy WAV files into output_dir/<session>/ and return manifest entries."""
    manifest_entries = []

    for session, recs in sorted(sampled.items()):
        session_dir = output_dir / session
        if not dry_run:
            session_dir.mkdir(parents=True, exist_ok=True)

        for rec in recs:
            src = rec["wav_path"]
            dst = session_dir / rec["filename"]
            resolved_note = " (resolved)" if rec["path_was_broken"] else ""

            if dry_run:
                exists = src.exists()
                status = "OK" if exists else "MISSING"
                print(f"  [{status}] {src}{resolved_note} -> {dst}")
            else:
                if not src.exists():
                    print(f"  WARNING: Source missing, skipped: {src}")
                    continue
                shutil.copy2(src, dst)
                print(f"  Copied{resolved_note}: {src.name} -> {session}/")

            manifest_entries.append({
                "filename": rec["filename"],
                "session": session,
                "source_wav": str(rec["wav_path"]),
                "original_stored_path": str(rec["original_path"]),
                "path_was_resolved": rec["path_was_broken"],
                "source_json": str(rec["json_path"]),
                "copied_to": str(dst),
            })

    return manifest_entries


def fix_broken_paths(records: list[dict], dry_run: bool) -> int:
    """Update noise label JSONs to point to the correct WAV paths.

    When files are moved after labeling (e.g. to a _reviewed folder), the
    stored path becomes stale.  This updates the JSON metadata.wav_file
    to the resolved path so future tools don't need the fallback logic.
    """
    fixed = 0
    for rec in records:
        if not rec["path_was_broken"]:
            continue
        if not rec["wav_path"].exists():
            print(f"  SKIP (still missing): {rec['filename']}")
            continue

        json_path = rec["json_path"]
        new_path_str = str(rec["wav_path"].resolve())

        if dry_run:
            print(f"  WOULD FIX: {json_path.name}")
            print(f"    Old: {rec['original_path']}")
            print(f"    New: {new_path_str}")
            fixed += 1
        else:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            data["metadata"]["wav_file"] = new_path_str
            data["metadata"]["path_fixed_at"] = datetime.now().isoformat()
            data["metadata"]["original_wav_file"] = str(rec["original_path"])
            json_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"  FIXED: {json_path.name} -> {new_path_str}")
            fixed += 1

    return fixed


def write_manifest(manifest_entries: list[dict], output_dir: Path) -> None:
    """Write a manifest.json tracking all copied files."""
    manifest = {
        "created_at": datetime.now().isoformat(),
        "total_files": len(manifest_entries),
        "sessions": sorted({e["session"] for e in manifest_entries}),
        "files": manifest_entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nManifest written to: {manifest_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Build a balanced noise dataset from noise-labeled WAV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--noise-labels",
        type=Path,
        default=repo_root / "USV_Detections" / "noise_labeled_files",
        help="Directory containing noise label JSONs "
             "(default: USV_Detections/noise_labeled_files)",
    )
    parser.add_argument(
        "--wav-dirs",
        type=Path,
        nargs="*",
        default=None,
        help="Directories to search for WAV files. "
             "Default: auto-discovers all USV_* folders under the repo root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "datasets" / "noise",
        help="Output directory for the noise dataset (default: datasets/noise)",
    )
    parser.add_argument(
        "--per-session",
        type=int,
        default=20,
        help="Max noise files per session. 0 = no limit (default: 20)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without actually writing anything",
    )
    parser.add_argument(
        "--fix-paths",
        action="store_true",
        help="Update broken paths in noise label JSONs to point to the "
             "actual WAV file locations. Combine with --dry-run to preview.",
    )
    args = parser.parse_args()

    # --- Auto-discover WAV directories ---
    if args.wav_dirs is None:
        wav_dirs = sorted(
            d for d in repo_root.iterdir()
            if d.is_dir() and d.name.startswith("USV_")
        )
        # Also include the repo root itself and common subdirs
        wav_dirs = [repo_root] + wav_dirs
    else:
        wav_dirs = args.wav_dirs

    # --- Step 1: Build filename index ---
    print("Building WAV filename index...")
    filename_index = _build_filename_index(wav_dirs)
    print(f"Indexed {sum(len(v) for v in filename_index.values())} WAV files "
          f"across {len(wav_dirs)} directories\n")

    # --- Step 2: Find noise labels ---
    print(f"Scanning noise labels in: {args.noise_labels}")
    if not args.noise_labels.is_dir():
        print(f"ERROR: Noise label directory not found: {args.noise_labels}")
        sys.exit(1)

    records = find_noise_labels(args.noise_labels, wav_dirs, filename_index)
    print(f"Found {len(records)} noise-labeled files")

    # Report path health
    broken = [r for r in records if r["path_was_broken"]]
    resolved = [r for r in broken if r["wav_path"].exists()]
    still_missing = [r for r in records if not r["wav_path"].exists()]
    if broken:
        print(f"  Broken paths: {len(broken)} "
              f"({len(resolved)} resolved, {len(still_missing)} still missing)")
        if still_missing:
            for r in still_missing[:5]:
                print(f"    MISSING: {r['filename']}")
            if len(still_missing) > 5:
                print(f"    ... and {len(still_missing) - 5} more")
    else:
        print("  All paths valid")
    print()

    if not records:
        print("Nothing to do.")
        sys.exit(0)

    # --- Fix paths mode ---
    if args.fix_paths:
        if not broken:
            print("No broken paths to fix.")
            sys.exit(0)

        if args.dry_run:
            print("=== DRY RUN -- no JSONs will be modified ===\n")
        else:
            print("Fixing broken paths in noise label JSONs...\n")

        fixed = fix_broken_paths(records, args.dry_run)
        verb = "would be fixed" if args.dry_run else "fixed"
        print(f"\n{fixed} label(s) {verb}.")
        if args.dry_run:
            print("Re-run without --dry-run to apply fixes.")
        sys.exit(0)

    # --- Step 3: Group by session ---
    groups = group_by_session(records)
    print("Noise files per recording session:")
    for session, recs in groups.items():
        ok = sum(1 for r in recs if r["wav_path"].exists())
        print(f"  {session}: {len(recs)} labeled ({ok} available)")
    print()

    # --- Step 4: Sample ---
    sampled = sample_per_session(groups, args.per_session)
    cap_label = (f"(capped at {args.per_session})"
                 if args.per_session > 0 else "(no cap)")
    total_selected = sum(len(recs) for recs in sampled.values())
    print(f"Selected {total_selected} files {cap_label}:")
    for session, recs in sampled.items():
        print(f"  {session}: {len(recs)} files")
    print()

    # --- Step 5: Copy ---
    if args.dry_run:
        print("=== DRY RUN -- no files will be copied ===\n")
    else:
        print(f"Copying to: {args.output}\n")

    manifest_entries = copy_files(sampled, args.output, args.dry_run)

    # --- Step 6: Manifest ---
    if not args.dry_run and manifest_entries:
        write_manifest(manifest_entries, args.output)

    # --- Summary ---
    verb = "would be" if args.dry_run else ""
    print(f"\nDone. {len(manifest_entries)} files {verb} copied.")
    if args.dry_run:
        print("\nRe-run without --dry-run to actually copy files.")


if __name__ == "__main__":
    main()
