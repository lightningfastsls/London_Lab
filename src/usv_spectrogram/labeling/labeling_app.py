"""Streamlit app for labeling USV candidate spectrograms."""

from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import soundfile as sf

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from usv_spectrogram.detection.candidate import Candidate
from usv_spectrogram.detection.extraction_config import ExtractionConfig
from usv_spectrogram.detection.spectrogram_extractor import SpectrogramExtractor
from usv_spectrogram.io_wav import get_default_wav_dir

LABEL_OPTIONS = ["USV", "Not USV", "Uncertain"]
LABELING_GUIDE_URL = "https://github.com/your-org/usv-labeling-guide"  # TODO: Update with actual URL


def load_candidates(csv_path: Path) -> pd.DataFrame:
    """Load candidates from CSV and sort by candidate_id."""
    df = pd.read_csv(csv_path)
    df = df.sort_values("candidate_id").reset_index(drop=True)
    return df


def load_existing_labels(labels_path: Path) -> dict[str, dict[str, Any]]:
    """Load existing labels from CSV into a dict keyed by candidate_id."""
    labels_dict = {}
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                expand_raw = row.get("expand_ms", "")
                try:
                    expand_ms = float(expand_raw) if expand_raw not in (None, "") else 0.0
                except ValueError:
                    expand_ms = 0.0
                labels_dict[row["candidate_id"]] = {
                    "label": row["label"],
                    "labeled_at": row["labeled_at"],
                    "expand_ms": expand_ms,
                }
    return labels_dict


def get_wav_duration_ms(wav_path: Path) -> float | None:
    """Return WAV duration in milliseconds, or None if unavailable."""
    try:
        with sf.SoundFile(str(wav_path)) as wav:
            return (len(wav) / float(wav.samplerate)) * 1000.0
    except Exception:
        return None


def expand_and_reextract(
    candidate_row: pd.Series,
    expand_ms: float,
    wav_dir: Path,
    output_dir: Path,
) -> Path | None:
    """Expand the segment and re-extract spectrogram.

    expand_ms > 0: Expand start earlier
    expand_ms < 0: Expand end later
    """
    if expand_ms == 0:
        return None

    original_start = float(candidate_row["start_ms"])
    original_end = float(candidate_row["end_ms"])
    context_before = original_start - float(candidate_row["context_start_ms"])
    context_after = float(candidate_row["context_end_ms"]) - original_end

    if expand_ms > 0:
        new_start = max(0.0, original_start - expand_ms)
        new_end = original_end
    else:
        new_start = original_start
        new_end = original_end - expand_ms  # expand_ms is negative

    wav_path = wav_dir / candidate_row["source_file"]
    duration_ms = get_wav_duration_ms(wav_path)
    if duration_ms is not None:
        new_end = min(new_end, duration_ms)

    if new_end <= new_start:
        return None

    expanded_id = f"{candidate_row['candidate_id']}_expanded"
    new_duration = new_end - new_start

    candidate = Candidate(
        source_file=wav_path,
        candidate_id=expanded_id,
        start_ms=new_start,
        end_ms=new_end,
        duration_ms=new_duration,
        context_start_ms=max(0.0, new_start - context_before),
        context_end_ms=(
            min(new_end + context_after, duration_ms)
            if duration_ms is not None
            else new_end + context_after
        ),
        peak_freq_hz=0.0,
        peak_energy_db=0.0,
        interference_flag=False,
    )

    config = ExtractionConfig(sample_rate=300000, default_render_mode="review")
    extractor = SpectrogramExtractor(config)

    return extractor.extract_single(candidate, wav_dir, output_dir, "review")


def save_label(
    labels_path: Path,
    candidate_id: str,
    label: str,
    expand_ms: float | None,
    existing_labels: dict[str, dict[str, Any]],
) -> None:
    """Save a label to the labels CSV file (append or update)."""
    labeled_at = datetime.now().isoformat()
    if expand_ms is None:
        expand_ms = float(existing_labels.get(candidate_id, {}).get("expand_ms", 0.0))
    existing_labels[candidate_id] = {
        "label": label,
        "labeled_at": labeled_at,
        "expand_ms": float(expand_ms),
    }

    # Write entire labels dict to CSV
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "label", "labeled_at", "expand_ms"],
        )
        writer.writeheader()
        for cid, lbl_data in existing_labels.items():
            writer.writerow({
                "candidate_id": cid,
                "label": lbl_data["label"],
                "labeled_at": lbl_data["labeled_at"],
                "expand_ms": lbl_data.get("expand_ms", 0.0),
            })


def archive_labels_and_spectrograms(
    labels_path: Path,
    spectrograms_dir: Path,
    archive_root: Path,
    labels: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Archive current labels and move labeled spectrograms to a new folder."""
    if not labels:
        return {"archive_dir": None, "labels_count": 0, "moved": 0, "missing": 0}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = archive_root / f"labeling_archive_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    backup_path = archive_dir / "labels.csv"
    if labels_path.exists():
        shutil.copy2(labels_path, backup_path)
    else:
        with open(backup_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["candidate_id", "label", "labeled_at", "expand_ms"],
            )
            writer.writeheader()
            for cid, lbl_data in labels.items():
                writer.writerow({
                    "candidate_id": cid,
                    "label": lbl_data["label"],
                    "labeled_at": lbl_data["labeled_at"],
                    "expand_ms": lbl_data.get("expand_ms", 0.0),
                })

    spectrogram_archive = archive_dir / "spectrograms"
    spectrogram_archive.mkdir(parents=True, exist_ok=True)

    moved = 0
    missing = 0
    for candidate_id in labels.keys():
        src = spectrograms_dir / f"{candidate_id}.png"
        if not src.exists():
            missing += 1
            continue
        dst = spectrogram_archive / src.name
        shutil.move(str(src), str(dst))
        moved += 1

    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "label", "labeled_at", "expand_ms"],
        )
        writer.writeheader()

    return {
        "archive_dir": archive_dir,
        "labels_count": len(labels),
        "moved": moved,
        "missing": missing,
    }


def get_spectrogram_path(spectrograms_dir: Path, candidate_id: str) -> Path:
    """Get path to spectrogram PNG for a candidate."""
    return spectrograms_dir / f"{candidate_id}.png"


def load_spectrogram_bytes(spectrogram_path: Path) -> bytes | None:
    """Load spectrogram image bytes from disk."""
    try:
        return spectrogram_path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError:
        return None


def resolve_input_path(raw_path: str, repo_root: Path) -> Path:
    """Resolve a user-provided path (absolute or repo-relative)."""
    if not raw_path:
        return Path()
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return path


def initialize_session_state(
    candidates: pd.DataFrame,
    labels: dict[str, dict[str, Any]],
) -> None:
    """Initialize Streamlit session state variables."""
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "labels" not in st.session_state:
        st.session_state.labels = labels
    if "total_candidates" not in st.session_state:
        st.session_state.total_candidates = len(candidates)
    if "expand_value" not in st.session_state:
        st.session_state.expand_value = 0
    if "pending_expands" not in st.session_state:
        st.session_state.pending_expands = {}
    if "label_filter" not in st.session_state:
        st.session_state.label_filter = "All"


def reset_session_state() -> None:
    """Clear session state keys to reload data from disk."""
    for key in (
        "current_index",
        "labels",
        "total_candidates",
        "expand_value",
        "expand_preview_path",
        "last_candidate_id",
        "pending_expands",
    ):
        st.session_state.pop(key, None)


def get_candidate_label(candidate_id: str, labels: dict[str, dict[str, Any]]) -> str | None:
    """Get the label for a candidate, or None if unlabeled."""
    if candidate_id in labels:
        return labels[candidate_id]["label"]
    return None


def count_labeled(candidates: pd.DataFrame, labels: dict[str, dict[str, Any]]) -> int:
    """Count how many candidates have been labeled."""
    return sum(1 for cid in candidates["candidate_id"] if cid in labels)


def filter_candidates(
    candidates: pd.DataFrame,
    labels: dict[str, dict[str, Any]],
    filter_type: str,
) -> pd.DataFrame:
    """Filter candidates based on label status.

    Args:
        candidates: All candidates DataFrame
        labels: Dictionary of labels
        filter_type: One of "All", "Only USV", "Only Not USV", "Only Uncertain", "Unlabeled"

    Returns:
        Filtered DataFrame with original indices preserved
    """
    if filter_type == "All":
        return candidates

    if filter_type == "Unlabeled":
        mask = [cid not in labels for cid in candidates["candidate_id"]]
        return candidates[mask].reset_index(drop=True)

    # Filter by specific label
    target_label = None
    if filter_type == "Only USV":
        target_label = "USV"
    elif filter_type == "Only Not USV":
        target_label = "Not USV"
    elif filter_type == "Only Uncertain":
        target_label = "Uncertain"

    if target_label:
        mask = [
            cid in labels and labels[cid]["label"] == target_label
            for cid in candidates["candidate_id"]
        ]
        return candidates[mask].reset_index(drop=True)

    return candidates


def find_next_unlabeled(
    candidates: pd.DataFrame,
    labels: dict[str, dict[str, Any]],
    start_index: int,
) -> int | None:
    """Find the next unlabeled candidate starting from start_index."""
    for i in range(start_index, len(candidates)):
        candidate_id = candidates.iloc[i]["candidate_id"]
        if candidate_id not in labels:
            return i
    return None


def render_navigation(candidates: pd.DataFrame, labels: dict[str, dict[str, Any]]) -> None:
    """Render navigation controls."""
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        if st.button("Previous", disabled=st.session_state.current_index == 0):
            st.session_state.current_index -= 1
            st.rerun()

    with col2:
        if st.button(
            "Next",
            disabled=st.session_state.current_index >= len(candidates) - 1,
        ):
            st.session_state.current_index += 1
            st.rerun()

    with col3:
        if st.button("Jump to Unlabeled"):
            next_unlabeled = find_next_unlabeled(
                candidates,
                labels,
                st.session_state.current_index + 1,
            )
            if next_unlabeled is not None:
                st.session_state.current_index = next_unlabeled
                st.rerun()
            else:
                st.info("No more unlabeled candidates found")

    with col4:
        # Progress indicator
        labeled_count = count_labeled(candidates, labels)
        st.write(f"Progress: {labeled_count} / {len(candidates)} labeled")


def render_candidate_info(candidate: pd.Series) -> None:
    """Render candidate metadata."""
    st.subheader("Candidate Information")

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Candidate ID:** {candidate['candidate_id']}")
        st.write(f"**Source File:** {candidate['source_file']}")
        st.write(f"**Start Time:** {candidate['start_ms']:.2f} ms")
        st.write(f"**End Time:** {candidate['end_ms']:.2f} ms")

    with col2:
        st.write(f"**Duration:** {candidate['duration_ms']:.2f} ms")
        st.write(f"**Peak Frequency:** {candidate['peak_freq_hz']:.1f} Hz")
        st.write(f"**Peak Energy:** {candidate['peak_energy_db']:.2f} dB")
        st.write(f"**Interference Flag:** {candidate['interference_flag']}")


def render_labeling_controls(
    candidates: pd.DataFrame,
    candidate_id: str,
    current_label: str | None,
    expand_ms: float,
    labels_path: Path,
    current_index: int,
    labels: dict[str, dict[str, Any]],
) -> None:
    """Render labeling buttons."""
    st.subheader("Label this Candidate")

    def apply_label(next_label: str) -> None:
        save_label(
            labels_path,
            candidate_id,
            next_label,
            expand_ms,
            labels,
        )
        if "pending_expands" in st.session_state:
            st.session_state.pending_expands.pop(candidate_id, None)
        next_index = find_next_unlabeled(
            candidates,
            labels,
            current_index + 1,
        )
        if next_index is None:
            next_index = (
                current_index + 1
                if current_index < len(candidates) - 1
                else current_index
            )
        st.session_state.current_index = next_index
        st.success(f"Labeled as: {next_label}")
        st.rerun()

    components.html(
        """
        <script>
        (function () {
            const parentDoc = window.parent && window.parent.document ? window.parent.document : document;
            if (parentDoc._usvShortcutBound) return;
            parentDoc._usvShortcutBound = true;
            parentDoc.addEventListener("keydown", (event) => {
                if (event.repeat) return;
                const tag = event.target.tagName.toLowerCase();
                const isInput = ["input", "textarea", "select"].includes(tag) || event.target.isContentEditable;
                if (isInput) return;
                if (!["1", "2", "3"].includes(event.key)) return;
                event.preventDefault();
                const labelMap = { "1": "USV", "2": "Not USV", "3": "Uncertain" };
                const targetLabel = labelMap[event.key];
                const buttons = Array.from(parentDoc.querySelectorAll("button"));
                const target = buttons.find((btn) => {
                    const text = (btn.innerText || "").trim();
                    return text.startsWith(targetLabel);
                });
                if (target) {
                    target.click();
                }
            });
        })();
        </script>
        """,
        height=0,
        width=0,
    )

    if current_label:
        st.info(f"Current label: **{current_label}** (expand: {expand_ms:.0f} ms)")

    col1, col2, col3 = st.columns(3)

    for i, (col, label) in enumerate(zip([col1, col2, col3], LABEL_OPTIONS)):
        with col:
            # Keyboard shortcuts: 1=USV, 2=Not USV, 3=Uncertain
            keyboard_hint = f" (Press {i+1})"
            button_label = f"{label}{keyboard_hint}"

            if st.button(button_label, key=f"label_{label}", use_container_width=True):
                apply_label(label)


def render_expand_controls(
    candidates: pd.DataFrame,
    candidate: pd.Series,
    candidate_id: str,
    current_label: str | None,
    labels_path: Path,
    current_index: int,
    labels: dict[str, dict[str, Any]],
    wav_dir: Path,
    spectrograms_dir: Path,
) -> None:
    """Render expand detection controls."""
    st.subheader("Expand Detection")
    st.markdown("""
    - **Positive value**: Expand **start** earlier (e.g., `20` shows 20ms more before)
    - **Negative value**: Expand **end** later (e.g., `-20` shows 20ms more after)
    """)

    if not wav_dir.exists():
        st.error(f"WAV directory not found: {wav_dir}")
        return

    expanded_output_dir = spectrograms_dir / "expanded"
    preview_dir = expanded_output_dir / "preview"

    col_expand, col_preview, col_save = st.columns([2, 1, 1])

    with col_expand:
        expand_value = st.number_input(
            "Expand amount (ms)",
            min_value=-200,
            max_value=200,
            value=int(st.session_state.expand_value),
            step=5,
            help="Positive = expand start earlier, Negative = expand end later",
        )
        st.session_state.expand_value = expand_value

    with col_preview:
        if st.button("Preview Expand"):
            if expand_value == 0:
                st.warning("Enter a non-zero expand value first")
            else:
                wav_path = wav_dir / candidate["source_file"]
                if not wav_path.exists():
                    st.error(f"WAV not found: {wav_path}")
                else:
                    with st.spinner("Generating preview..."):
                        preview_path = expand_and_reextract(
                            candidate,
                            expand_value,
                            wav_dir,
                            preview_dir,
                        )
                        if preview_path and preview_path.exists():
                            st.session_state.expand_preview_path = str(preview_path)
                            st.rerun()
                        else:
                            st.error("Failed to generate expanded preview")

    with col_save:
        if st.button("Save Expanded", type="primary"):
            if expand_value == 0:
                st.warning("Enter a non-zero expand value first")
            else:
                wav_path = wav_dir / candidate["source_file"]
                if not wav_path.exists():
                    st.error(f"WAV not found: {wav_path}")
                else:
                    final_path = expand_and_reextract(
                        candidate,
                        expand_value,
                        wav_dir,
                        expanded_output_dir,
                    )
                    if final_path:
                        if current_label:
                            save_label(
                                labels_path,
                                candidate_id,
                                current_label,
                                expand_value,
                                labels,
                            )
                            if "pending_expands" in st.session_state:
                                st.session_state.pending_expands.pop(candidate_id, None)
                            next_index = find_next_unlabeled(
                                candidates,
                                labels,
                                current_index + 1,
                            )
                            if next_index is None:
                                next_index = (
                                    current_index + 1
                                    if current_index < len(candidates) - 1
                                    else current_index
                                )
                            st.session_state.current_index = next_index
                            st.session_state.expand_preview_path = None
                            st.success("Saved expanded spectrogram and expansion value.")
                            st.rerun()
                        else:
                            st.session_state.pending_expands[candidate_id] = float(expand_value)
                            st.warning(
                                "Expanded spectrogram saved. Label this candidate to persist expand_ms in labels.csv."
                            )
                        st.session_state.expand_preview_path = str(final_path)
                    else:
                        st.error("Expansion failed - check bounds and WAV availability")

    if getattr(st.session_state, "expand_preview_path", None):
        preview_path = Path(st.session_state.expand_preview_path)
        if preview_path.exists():
            st.subheader("Expanded Preview")
            st.image(str(preview_path), use_container_width=True)


def render_labeling_guide() -> None:
    """Render labeling guide information."""
    with st.expander("Labeling Guide"):
        st.markdown(f"""
        ### How to Label USV Candidates

        **USV (Ultrasonic Vocalization):**
        - Clear, distinct frequency sweep or harmonic pattern
        - Typical frequency range: 30-110 kHz
        - Duration: 5-100 ms
        - Smooth, continuous energy pattern

        **Not USV:**
        - Random noise or interference
        - Background hum or electrical noise
        - Artifacts from equipment
        - Non-biological signals

        **Uncertain:**
        - Ambiguous patterns that could be USV or noise
        - Very faint signals
        - Overlapping signals
        - When you need expert review

        **Keyboard Shortcuts:**
        - Press **1** for USV
        - Press **2** for Not USV
        - Press **3** for Uncertain

        ### Expand Detection
        - **Positive expand (e.g., +20):** Expand the **start** earlier
        - **Negative expand (e.g., -20):** Expand the **end** later
        - Expanded PNGs are saved under the spectrograms folder in `expanded/`

        For more details, see the [full labeling guide]({LABELING_GUIDE_URL}).
        """)


def run() -> None:
    """Run the USV labeling tool (DEPRECATED — use the PyQt6 desktop app)."""
    import sys

    sys.stderr.write(
        "\n[DEPRECATED] The Streamlit USV labeling tool has been retired.\n"
        "Use the PyQt6 desktop app instead:\n"
        "    .venv/bin/python scripts/run_app.py\n\n"
    )
    sys.exit(1)
    # --- unreachable: legacy Streamlit implementation preserved for reference ---
    st.set_page_config(page_title="USV Labeling Tool", layout="wide")
    st.title("USV Candidate Labeling Tool")

    archive_notice = st.session_state.pop("archive_notice", None)
    if archive_notice:
        st.success(archive_notice)

    # Paths
    repo_root = Path(__file__).resolve().parents[3]
    default_candidates_csv = repo_root / "candidates_optimized.csv"
    default_spectrograms_dir = repo_root / "spectrograms_review"
    labels_csv = repo_root / "labels.csv"
    archive_root = repo_root / "labeling_archives"
    wav_dir = get_default_wav_dir()

    candidates_csv_text = st.session_state.get("candidates_csv_path", str(default_candidates_csv))
    spectrograms_dir_text = st.session_state.get("spectrograms_dir_path", str(default_spectrograms_dir))

    with st.sidebar:
        st.header("Controls")
        with st.form("path_selector"):
            candidates_input = st.text_input("Candidates CSV", value=candidates_csv_text)
            spectrograms_input = st.text_input("Spectrograms folder", value=spectrograms_dir_text)
            apply_paths = st.form_submit_button("Use paths")
        if apply_paths:
            st.session_state["candidates_csv_path"] = candidates_input.strip()
            st.session_state["spectrograms_dir_path"] = spectrograms_input.strip()
            reset_session_state()
            st.rerun()

        if st.button("Reload data from disk"):
            reset_session_state()
            st.rerun()
        st.caption("Use after regenerating spectrograms in spectrograms_review.")
        st.divider()

        # Label filter
        st.subheader("Filter")
        filter_options = ["All", "Only USV", "Only Not USV", "Only Uncertain", "Unlabeled"]
        selected_filter = st.selectbox(
            "Show candidates:",
            options=filter_options,
            index=filter_options.index(st.session_state.get("label_filter", "All")),
            key="filter_selector"
        )
        if selected_filter != st.session_state.get("label_filter"):
            st.session_state.label_filter = selected_filter
            st.session_state.current_index = 0  # Reset to first item when filter changes
            st.rerun()
        st.divider()

    candidates_csv = resolve_input_path(candidates_csv_text, repo_root)
    spectrograms_dir = resolve_input_path(spectrograms_dir_text, repo_root)

    # Load data
    if not candidates_csv.exists():
        st.error(f"Candidates file not found: {candidates_csv}")
        return

    if not spectrograms_dir.exists():
        st.error(f"Spectrograms directory not found: {spectrograms_dir}")
        return

    candidates = load_candidates(candidates_csv)
    existing_labels = load_existing_labels(labels_csv)

    # Initialize session state
    initialize_session_state(candidates, existing_labels)

    # Apply filter
    filtered_candidates = filter_candidates(
        candidates,
        st.session_state.labels,
        st.session_state.label_filter
    )

    # Check if filter resulted in empty set
    if len(filtered_candidates) == 0:
        st.warning(f"No candidates match filter: {st.session_state.label_filter}")
        st.info("Try selecting a different filter from the sidebar.")
        return

    # Ensure current_index is valid for filtered view
    if st.session_state.current_index >= len(filtered_candidates):
        st.session_state.current_index = 0

    # Render labeling guide
    render_labeling_guide()

    # Navigation
    render_navigation(filtered_candidates, st.session_state.labels)

    st.divider()

    # Get current candidate
    current_index = st.session_state.current_index
    candidate = filtered_candidates.iloc[current_index]
    candidate_id = candidate["candidate_id"]
    pending_expand = st.session_state.pending_expands.get(candidate_id, 0.0)
    current_expand = float(
        st.session_state.labels.get(candidate_id, {}).get("expand_ms", pending_expand)
    )
    if st.session_state.get("last_candidate_id") != candidate_id:
        st.session_state.expand_value = int(current_expand)
        st.session_state.expand_preview_path = None
        st.session_state.last_candidate_id = candidate_id

    # Display candidate info with filter context
    if st.session_state.label_filter == "All":
        st.write(f"### Candidate {current_index + 1} of {len(filtered_candidates)}")
    else:
        st.write(
            f"### Candidate {current_index + 1} of {len(filtered_candidates)} "
            f"({len(candidates)} total, filter: {st.session_state.label_filter})"
        )
    render_candidate_info(candidate)

    st.divider()

    # Display spectrogram
    spectrogram_path = get_spectrogram_path(spectrograms_dir, candidate_id)
    spectrogram_bytes = load_spectrogram_bytes(spectrogram_path)
    if spectrogram_bytes is not None:
        st.image(spectrogram_bytes, use_container_width=True)
    else:
        st.warning(f"Spectrogram not found: {spectrogram_path}")

    st.divider()

    # Labeling controls
    current_label = get_candidate_label(candidate_id, st.session_state.labels)
    render_labeling_controls(
        filtered_candidates,
        candidate_id,
        current_label,
        float(st.session_state.expand_value),
        labels_csv,
        current_index,
        st.session_state.labels,
    )

    st.divider()

    render_expand_controls(
        filtered_candidates,
        candidate,
        candidate_id,
        current_label,
        labels_csv,
        current_index,
        st.session_state.labels,
        wav_dir,
        spectrograms_dir,
    )

    # Show labeling statistics
    with st.sidebar:
        st.header("Statistics")

        # Show filtered view stats
        if st.session_state.label_filter != "All":
            st.metric("Filtered View", f"{len(filtered_candidates)} candidates")

        labeled_count = count_labeled(candidates, st.session_state.labels)
        st.metric("Total Labeled", f"{labeled_count} / {len(candidates)}")

        if labeled_count > 0:
            percentage = (labeled_count / len(candidates)) * 100
            st.metric("Progress", f"{percentage:.1f}%")

            # Count by label type
            st.subheader("Labels by Type")
            label_counts = {"USV": 0, "Not USV": 0, "Uncertain": 0}
            for lbl_data in st.session_state.labels.values():
                label = lbl_data["label"]
                if label in label_counts:
                    label_counts[label] += 1

            for label, count in label_counts.items():
                st.write(f"**{label}:** {count}")

        st.divider()
        st.subheader("Archive / Reset")
        st.caption("Move current labels and labeled spectrograms out of spectrograms_review, then start fresh.")
        confirm_archive = st.checkbox(
            "I understand this will move labeled PNGs and clear labels.csv.",
            key="confirm_archive",
        )
        if st.button("Archive labels + labeled spectrograms", disabled=not confirm_archive):
            if not st.session_state.labels:
                st.info("No labels found to archive.")
            else:
                result = archive_labels_and_spectrograms(
                    labels_csv,
                    spectrograms_dir,
                    archive_root,
                    st.session_state.labels,
                )
                st.session_state.labels = {}
                st.session_state.current_index = 0
                st.session_state.archive_notice = (
                    "Archived {labels_count} labels to {archive_dir} and moved {moved} PNGs "
                    "(missing {missing})."
                ).format(
                    labels_count=result["labels_count"],
                    archive_dir=result["archive_dir"],
                    moved=result["moved"],
                    missing=result["missing"],
                )
                st.rerun()


if __name__ == "__main__":
    run()
