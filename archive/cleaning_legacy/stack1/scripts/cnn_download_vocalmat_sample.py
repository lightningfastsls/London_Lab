#!/usr/bin/env python
"""VocalMat Sample Download — Module 18.2a.

Downloads a ~200-spectrogram-per-class sample of the VocalMat dataset
(OSF project ``bk2uj``) into ``data/vocalmat_sample/`` for Module 18.1's
real-data cleaning-validation gate. Optionally bridges to 18.2b's full
pull via ``--full``.

Class folder names on OSF use the snake_case Grimsley naming
(``mult_steps`` not ``multi_steps``, etc.) — the 12 names are listed in
``GRIMSLEY_OSF_CLASSES``.

Examples
--------
Dry-run (no downloads, prints plan)::

    python scripts/cnn_download_vocalmat_sample.py --dry-run

Default 200/class small sample::

    python scripts/cnn_download_vocalmat_sample.py \\
        --output-dir data/vocalmat_sample/ \\
        --n-per-class 200

Full pull (18.2b, only after gate verdict = GO)::

    python scripts/cnn_download_vocalmat_sample.py --full
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, NamedTuple, Protocol

# Bootstrap (patterns.md §4) — kept even though this script does not import
# from src/, so future helpers added here can use the same path.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OSF_PROJECT_ID = "bk2uj"
DATASET_FOLDER_NAME = "Dataset"

OSF_API_BASE = "https://api.osf.io/v2"
OSF_DOWNLOAD_BASE = "https://osf.io/download"

# Socket timeout for OSF HTTP calls. OSF can be slow but not minutes-slow.
# Without a timeout, urllib will block forever on stalled connections —
# which is exactly the failure mode that 3-hour-hung the first attempt
# (osfclient ships with no socket timeout configured).
HTTP_TIMEOUT_S = 30

# OSF JSON-API page-size cap. Tested at 100; OSF's documented max is 100
# for resource lists. Cutting 12,200 file entries from 1,222 calls (at
# page_size=10) to 122 calls (at page_size=100).
OSF_PAGE_SIZE = 100

# Class folder names on OSF. Order is the source of truth for the manifest's
# class-iteration order. VocalMat uses snake_case throughout.
GRIMSLEY_OSF_CLASSES: tuple[str, ...] = (
    "noise",
    "step_up",
    "down_fm",
    "short",
    "chevron",
    "up_fm",
    "flat",
    "two_steps",
    "step_down",
    "complex",
    "rev_chevron",
    "mult_steps",
)

# Stable seed ensures the same sample of 200 files is selected across runs,
# which is what makes the downloader idempotent at the FILE-CHOICE layer
# (the FILE-PRESENCE layer is separately idempotent via existence-check).
SAMPLING_SEED = 1729

DEFAULT_N_PER_CLASS = 200

MANIFEST_FILENAME = "manifest.csv"
MANIFEST_COLUMNS = ("path", "class", "source_recording", "osf_path", "size_bytes")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class FileEntry(NamedTuple):
    """Metadata for one file on OSF, plus a stable identifier for download.

    ``osf_path`` is the OSF storage UUID path (e.g. ``/5ec...``). We carry it
    so the manifest can be traced back to OSF revisions, and so the test
    fake can use synthetic UUIDs without coupling to filesystem layout.
    """

    osf_path: str
    name: str
    size_bytes: int
    source_recording: str


# ---------------------------------------------------------------------------
# Retry helper — OSF returns 429 on burst usage
# ---------------------------------------------------------------------------


def _retry_on_429(func, *, max_retries: int = 5, base_delay_s: float = 10.0):
    """Run ``func()`` and retry with exponential backoff on HTTP 429.

    OSF's anti-burst behaviour can persist for tens of seconds after a
    parallel-request spike. Backoff schedule (base=10s): 10s, 20s, 40s,
    80s, 160s — total worst case ~5 min, which covers any reasonable
    OSF cool-down window without hammering them further.
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except RuntimeError as e:
            msg = str(e)
            if "429" in msg and attempt < max_retries:
                delay = base_delay_s * (2 ** attempt)
                print(
                    f"  [429] OSF rate-limited; sleeping {delay:.0f}s "
                    f"(attempt {attempt + 1}/{max_retries})",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
                continue
            raise


# ---------------------------------------------------------------------------
# Source abstraction (dependency injection for testability)
# ---------------------------------------------------------------------------


class VocalMatSource(Protocol):
    """Abstraction over the VocalMat data source.

    Production implementation (:class:`OSFVocalMatSource`) hits OSF via
    stdlib urllib. Tests substitute :class:`FakeVocalMatSource` to verify
    enumeration / sampling / manifest logic without touching the network.
    """

    def list_files(self, class_name: str) -> list[FileEntry]:
        """Return every file in ``Dataset/<class_name>/`` as ``FileEntry``."""
        ...

    def download(self, entry: FileEntry, dest: Path) -> None:
        """Fetch ``entry`` into ``dest`` (must create parent if needed)."""
        ...


def _http_get_json(url: str, timeout_s: float = HTTP_TIMEOUT_S) -> dict:
    """GET an OSF JSON-API endpoint with a socket timeout and 429-as-RuntimeError.

    OSF returns 4xx/5xx via ``urllib.error.HTTPError``. We re-raise 429 as
    a ``RuntimeError`` whose message starts with ``"429"`` so that
    :func:`_retry_on_429` catches it; other errors propagate unchanged.
    """
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.api+json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RuntimeError(f"429: OSF rate-limited at {url}") from None
        raise


def _http_download(url: str, dest: Path, timeout_s: float = HTTP_TIMEOUT_S) -> None:
    """Download ``url`` to ``dest``, streaming, with 429-as-RuntimeError."""
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            with open(dest, "wb") as fp:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    fp.write(chunk)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RuntimeError(f"429: OSF rate-limited at {url}") from None
        raise


def _with_page_size(url: str, page_size: int) -> str:
    """Append ``?page[size]=N`` (URL-encoded) to ``url``."""
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}page%5Bsize%5D={page_size}"


class OSFVocalMatSource:
    """Direct-REST implementation of :class:`VocalMatSource`.

    Hits the OSF v2 JSON-API (``api.osf.io/v2``) with stdlib urllib —
    explicit socket timeouts and ``page[size]=100`` per request. This
    replaces an earlier osfclient-backed implementation that hung
    indefinitely on slow OSF responses (no socket timeout) and was
    capped to per_page=10 (123 calls inflated to 1,222).

    Two-stage walk:
    1. ``_ensure_class_folders`` does a single GET of
       ``/nodes/<project>/files/osfstorage/<dataset_folder_id>/`` and
       caches per-class ``files`` URLs.
    2. ``list_files(class_name)`` paginates the cached class URL,
       producing :class:`FileEntry` rows AND caching OSF file IDs for
       O(1) download lookup.
    """

    def __init__(self, project_id: str = OSF_PROJECT_ID) -> None:
        self._project_id = project_id
        # class_name -> URL pointing at "files" of that class folder
        self._class_files_urls: dict[str, str] = {}
        # (class_name) -> {file_name: osf_file_id} for O(1) download
        self._file_id_cache: dict[str, dict[str, str]] = {}

    def _ensure_class_folders(self) -> None:
        if self._class_files_urls:
            return

        def _do_walk() -> dict[str, str]:
            # Step 1: list top-level folders under osfstorage to find Dataset
            root_url = _with_page_size(
                f"{OSF_API_BASE}/nodes/{self._project_id}/files/osfstorage/",
                OSF_PAGE_SIZE,
            )
            root = _http_get_json(root_url)
            dataset_link: str | None = None
            for item in root.get("data", []):
                a = item.get("attributes", {})
                if a.get("kind") == "folder" and a.get("name") == DATASET_FOLDER_NAME:
                    dataset_link = (
                        item["relationships"]["files"]["links"]["related"]["href"]
                    )
                    break
            if dataset_link is None:
                raise RuntimeError(
                    f"OSF project {self._project_id!r} has no top-level "
                    f"'{DATASET_FOLDER_NAME}' folder."
                )

            # Step 2: list class folders under Dataset/ (12 entries today)
            urls: dict[str, str] = {}
            next_url = _with_page_size(dataset_link, OSF_PAGE_SIZE)
            while next_url:
                page = _http_get_json(next_url)
                for item in page.get("data", []):
                    a = item.get("attributes", {})
                    if a.get("kind") != "folder":
                        continue
                    name = a.get("name")
                    files_url = (
                        item["relationships"]["files"]["links"]["related"]["href"]
                    )
                    urls[name] = files_url
                next_url = page.get("links", {}).get("next")
            return urls

        self._class_files_urls = _retry_on_429(_do_walk)

    def list_files(self, class_name: str) -> list[FileEntry]:
        self._ensure_class_folders()
        files_url = self._class_files_urls.get(class_name)
        if files_url is None:
            raise RuntimeError(
                f"OSF Dataset has no class folder {class_name!r}. "
                f"Available: {sorted(self._class_files_urls)}"
            )

        def _do_enumerate() -> tuple[list[FileEntry], dict[str, str]]:
            out: list[FileEntry] = []
            ids: dict[str, str] = {}
            next_url = _with_page_size(files_url, OSF_PAGE_SIZE)
            while next_url:
                page = _http_get_json(next_url)
                for item in page.get("data", []):
                    a = item.get("attributes", {})
                    if a.get("kind") != "file":
                        continue
                    name = a.get("name")
                    file_id = item.get("id") or ""
                    size = int(a.get("size") or 0)
                    osf_path = a.get("path") or f"/{file_id}"
                    ids[name] = file_id
                    out.append(
                        FileEntry(
                            osf_path=osf_path,
                            name=name,
                            size_bytes=size,
                            source_recording=_parse_source_recording(name),
                        )
                    )
                next_url = page.get("links", {}).get("next")
            return out, ids

        out, ids = _retry_on_429(_do_enumerate)
        self._file_id_cache[class_name] = ids
        return out

    def download(self, entry: FileEntry, dest: Path) -> None:
        # Class is identified by the immediate parent dir of dest, which
        # matches the layout ``output_dir/<class_name>/<file_name>``.
        class_name = dest.parent.name
        ids = self._file_id_cache.get(class_name)
        if ids is None:
            raise RuntimeError(
                f"download() called for class {class_name!r} but list_files() "
                "was never invoked for it — id cache is empty."
            )
        file_id = ids.get(entry.name)
        if not file_id:
            raise RuntimeError(
                f"File {entry.name!r} not in id cache for class "
                f"{class_name!r} (cache has {len(ids)} entries)."
            )
        url = f"{OSF_DOWNLOAD_BASE}/{file_id}/"
        dest.parent.mkdir(parents=True, exist_ok=True)
        _retry_on_429(lambda: _http_download(url, dest))


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------


def _parse_source_recording(filename: str) -> str:
    """Extract a recording identifier from a VocalMat filename.

    VocalMat filenames embed the source recording in the stem. Two known
    patterns:

    1. ``<idx>_<idx>_isolation<N>_<animalID>.png`` — animal ID is e.g.
       ``5662F`` (sex-suffixed numeric tag).
    2. ``<idx>_<animalID>_<descriptor>.png`` — e.g.
       ``53_4829_Control_Baseline.png``.

    Returns the everything-after-prefix token, or ``"unknown"`` if no
    pattern matches. Used as the ``source_recording`` field in the
    manifest CSV so downstream code can group by recording.
    """
    stem = Path(filename).stem
    m = re.match(r".+_isolation\d+_(.+)$", stem)
    if m:
        return m.group(1)
    parts = stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[1:])
    return "unknown"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_per_class(
    entries_by_class: dict[str, list[FileEntry]],
    n_per_class: int | None,
    seed: int = SAMPLING_SEED,
) -> dict[str, list[FileEntry]]:
    """Random sample without replacement, deterministic across runs.

    If ``n_per_class`` is ``None`` (the ``--full`` case), returns the full
    file lists per class unchanged (still sorted for determinism).

    If a class has fewer than ``n_per_class`` files, returns all of them
    (silently — but ``main`` reports the actual count). VocalMat's
    ``mult_steps`` class only has 74 files, so a 200-cap silently
    degrades to 74 for that class.
    """
    out: dict[str, list[FileEntry]] = {}
    rng = random.Random(seed)
    for class_name, entries in entries_by_class.items():
        # Sort first so the seed picks the same files regardless of OSF
        # enumeration order on a given run.
        sorted_entries = sorted(entries, key=lambda e: e.osf_path)
        if n_per_class is None:
            out[class_name] = list(sorted_entries)
        elif len(sorted_entries) <= n_per_class:
            out[class_name] = list(sorted_entries)
        else:
            out[class_name] = rng.sample(sorted_entries, n_per_class)
    return out


# ---------------------------------------------------------------------------
# Download orchestration
# ---------------------------------------------------------------------------


def plan_paths(
    samples: dict[str, list[FileEntry]],
    output_dir: Path,
) -> list[tuple[str, FileEntry, Path]]:
    """Compute (class, entry, dest_path) tuples for the full download plan."""
    out: list[tuple[str, FileEntry, Path]] = []
    for class_name, entries in samples.items():
        cls_dir = output_dir / class_name
        for e in entries:
            out.append((class_name, e, cls_dir / e.name))
    return out


def download_plan(
    plan: list[tuple[str, FileEntry, Path]],
    source: VocalMatSource,
    on_progress: callable | None = None,
    workers: int = 1,
) -> dict[str, int]:
    """Execute the plan, skipping files that already exist on disk.

    Returns counts ``{"downloaded": N, "skipped": M, "failed": K}``.
    Failed entries are logged via ``on_progress`` if supplied; the
    function does NOT raise on individual failures — partial downloads
    are common and should not abort the whole 2,000-file run.

    ``workers`` > 1 runs downloads in a ThreadPoolExecutor. Downloads go
    through OSF's separate ``osf.io/download/<file_id>/`` endpoint, which
    is bandwidth-bound rather than API-rate-limited, so modest
    parallelism (e.g., 4) is safe.
    """
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}

    def _process_one(class_name: str, entry: FileEntry, dest: Path) -> str:
        """Return one of ('downloaded', 'skipped', 'failed')."""
        if dest.exists() and dest.stat().st_size > 0:
            if on_progress:
                on_progress("skip", class_name, entry.name, None)
            return "skipped"
        try:
            source.download(entry, dest)
            if on_progress:
                on_progress("ok", class_name, entry.name, None)
            return "downloaded"
        except Exception as exc:  # noqa: BLE001
            if on_progress:
                on_progress("fail", class_name, entry.name, str(exc))
            return "failed"

    if workers <= 1:
        for class_name, entry, dest in plan:
            counts[_process_one(class_name, entry, dest)] += 1
    else:
        # Local import to avoid pulling concurrent at module load.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_process_one, c, e, d) for c, e, d in plan
            ]
            for fut in as_completed(futures):
                counts[fut.result()] += 1
    return counts


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest(
    plan: list[tuple[str, FileEntry, Path]],
    output_dir: Path,
) -> Path:
    """Emit the manifest CSV; returns its path. Overwrites if existing."""
    manifest_path = output_dir / MANIFEST_FILENAME
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(MANIFEST_COLUMNS)
        for class_name, entry, dest in plan:
            w.writerow([
                str(dest.relative_to(output_dir.parent) if dest.is_absolute() and output_dir.is_absolute() else dest),
                class_name,
                entry.source_recording,
                entry.osf_path,
                entry.size_bytes,
            ])
    return manifest_path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_counts_table(
    samples: dict[str, list[FileEntry]],
    classes: Iterable[str] = GRIMSLEY_OSF_CLASSES,
) -> str:
    """Render a per-class sample-count table for stdout."""
    rows = ["class            count"]
    rows.append("---------------- -----")
    total = 0
    for c in classes:
        n = len(samples.get(c, []))
        total += n
        rows.append(f"{c:<16s} {n:>5d}")
    rows.append("---------------- -----")
    rows.append(f"{'TOTAL':<16s} {total:>5d}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a small VocalMat sample from OSF for the cleaning "
            "validation gate (Module 18.2a)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/cnn_download_vocalmat_sample.py --dry-run\n"
            "  python scripts/cnn_download_vocalmat_sample.py "
            "--output-dir data/vocalmat_sample/ --n-per-class 200\n"
            "  python scripts/cnn_download_vocalmat_sample.py --full"
        ),
    )
    parser.add_argument(
        "--osf-project", default=OSF_PROJECT_ID,
        help=f"OSF project ID (default: {OSF_PROJECT_ID})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "data" / "vocalmat_sample",
        help="Destination root (default: data/vocalmat_sample/)",
    )
    parser.add_argument(
        "--n-per-class", type=int, default=DEFAULT_N_PER_CLASS,
        help=(
            f"Per-class sample cap (default: {DEFAULT_N_PER_CLASS}). "
            "Silently degrades to class total if smaller — note "
            "'mult_steps' has only 74 files on OSF."
        ),
    )
    parser.add_argument(
        "--full", action="store_true",
        help=(
            "Bridge to 18.2b: download every file from every class. "
            "Ignores --n-per-class. Only use after gate verdict = GO."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Plan and report counts without fetching anything.",
    )
    parser.add_argument(
        "--seed", type=int, default=SAMPLING_SEED,
        help=f"Sampling seed for determinism (default: {SAMPLING_SEED}).",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help=(
            "Number of parallel download threads (default: 4). Downloads "
            "use OSF's /download/ endpoint which is bandwidth-bound rather "
            "than API-rate-limited; modest parallelism (~4-8) is safe. "
            "Set to 1 if you start seeing 429s."
        ),
    )
    return parser.parse_args(argv)


def _print_progress(state: str, class_name: str, file_name: str, err: str | None) -> None:
    """Default progress callback — single line per file."""
    if state == "ok":
        print(f"  [download] {class_name}/{file_name}", flush=True)
    elif state == "skip":
        print(f"  [skip   ] {class_name}/{file_name} (already on disk)", flush=True)
    elif state == "fail":
        print(f"  [FAIL   ] {class_name}/{file_name}: {err}", flush=True, file=sys.stderr)


def main(
    argv: list[str] | None = None,
    source_factory: callable | None = None,
) -> int:
    """Entry point.

    ``source_factory`` is a hook for tests: a callable that returns a
    :class:`VocalMatSource`. Defaults to ``OSFVocalMatSource``. Tests pass
    in a factory that returns a :class:`FakeVocalMatSource` so the script
    can be exercised end-to-end without OSF access.
    """
    args = parse_args(argv)

    if args.n_per_class is not None and args.n_per_class <= 0 and not args.full:
        print(
            f"ERROR: --n-per-class must be > 0 (got {args.n_per_class}). "
            "Use --full to pull every class without a cap.",
            file=sys.stderr,
        )
        return 2

    n_per_class = None if args.full else args.n_per_class

    if source_factory is None:
        source_factory = lambda: OSFVocalMatSource(args.osf_project)  # noqa: E731
    source: VocalMatSource = source_factory()

    print(f"[plan] Enumerating {len(GRIMSLEY_OSF_CLASSES)} classes on OSF "
          f"project {args.osf_project!r} (sequential, page_size=100) ...",
          flush=True)
    t_enum = time.monotonic()
    entries_by_class: dict[str, list[FileEntry]] = {}
    # Sequential enumeration to avoid HTTP 429 from OSF (parallel hits at
    # >10 rps trip their rate limit). Combined with the page-size=100
    # boost in OSFVocalMatSource, total enumeration time is ~30-60s for
    # ~12,200 files — fine for a one-time pull.
    for class_name in GRIMSLEY_OSF_CLASSES:
        entries_by_class[class_name] = source.list_files(class_name)
        print(f"  {class_name:<16s} total on OSF: "
              f"{len(entries_by_class[class_name]):>5d}", flush=True)
    print(f"[plan] Enumeration done in {time.monotonic() - t_enum:.1f}s.",
          flush=True)

    samples = sample_per_class(entries_by_class, n_per_class, seed=args.seed)
    print()
    print("[plan] Per-class samples to fetch:")
    print(render_counts_table(samples))
    print()

    plan = plan_paths(samples, args.output_dir)
    if args.dry_run:
        print(f"[dry-run] Would fetch {len(plan)} files into "
              f"{args.output_dir!r}. No downloads performed.")
        return 0

    print(f"[fetch] Starting {len(plan)} file downloads into {args.output_dir} ...")
    t_dl = time.monotonic()
    counts = download_plan(plan, source, on_progress=_print_progress,
                           workers=args.workers)
    dt = time.monotonic() - t_dl
    print(f"[fetch] Done in {dt:.1f}s: "
          f"{counts['downloaded']} downloaded, "
          f"{counts['skipped']} skipped, "
          f"{counts['failed']} failed.")

    manifest_path = write_manifest(plan, args.output_dir)
    print(f"[manifest] Wrote {manifest_path}")

    if counts["failed"] > 0:
        print(f"[warning] {counts['failed']} files failed to download. "
              "Re-running the script will retry (idempotent).",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
