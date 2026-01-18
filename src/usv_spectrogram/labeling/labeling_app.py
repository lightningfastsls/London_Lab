"""Streamlit app for labeling USV candidate spectrograms."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


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
                labels_dict[row["candidate_id"]] = {
                    "label": row["label"],
                    "labeled_at": row["labeled_at"],
                }
    return labels_dict


def save_label(
    labels_path: Path,
    candidate_id: str,
    label: str,
    existing_labels: dict[str, dict[str, Any]],
) -> None:
    """Save a label to the labels CSV file (append or update)."""
    labeled_at = datetime.now().isoformat()
    existing_labels[candidate_id] = {"label": label, "labeled_at": labeled_at}

    # Write entire labels dict to CSV
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "label", "labeled_at"])
        writer.writeheader()
        for cid, lbl_data in existing_labels.items():
            writer.writerow({
                "candidate_id": cid,
                "label": lbl_data["label"],
                "labeled_at": lbl_data["labeled_at"],
            })


def get_spectrogram_path(spectrograms_dir: Path, candidate_id: str) -> Path:
    """Get path to spectrogram PNG for a candidate."""
    return spectrograms_dir / f"{candidate_id}.png"


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


def get_candidate_label(candidate_id: str, labels: dict[str, dict[str, Any]]) -> str | None:
    """Get the label for a candidate, or None if unlabeled."""
    if candidate_id in labels:
        return labels[candidate_id]["label"]
    return None


def count_labeled(candidates: pd.DataFrame, labels: dict[str, dict[str, Any]]) -> int:
    """Count how many candidates have been labeled."""
    return sum(1 for cid in candidates["candidate_id"] if cid in labels)


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
            labels,
        )
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
        st.info(f"Current label: **{current_label}**")

    col1, col2, col3 = st.columns(3)

    for i, (col, label) in enumerate(zip([col1, col2, col3], LABEL_OPTIONS)):
        with col:
            # Keyboard shortcuts: 1=USV, 2=Not USV, 3=Uncertain
            keyboard_hint = f" (Press {i+1})"
            button_label = f"{label}{keyboard_hint}"

            if st.button(button_label, key=f"label_{label}", use_container_width=True):
                apply_label(label)


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

        For more details, see the [full labeling guide]({LABELING_GUIDE_URL}).
        """)


def run() -> None:
    """Run the USV labeling tool."""
    st.set_page_config(page_title="USV Labeling Tool", layout="wide")
    st.title("USV Candidate Labeling Tool")

    # Paths
    repo_root = Path(__file__).resolve().parents[3]
    candidates_csv = repo_root / "candidates_optimized.csv"
    spectrograms_dir = repo_root / "spectrograms_review"
    labels_csv = repo_root / "labels.csv"

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

    # Render labeling guide
    render_labeling_guide()

    # Navigation
    render_navigation(candidates, st.session_state.labels)

    st.divider()

    # Get current candidate
    current_index = st.session_state.current_index
    candidate = candidates.iloc[current_index]
    candidate_id = candidate["candidate_id"]

    # Display candidate info
    st.write(f"### Candidate {current_index + 1} of {len(candidates)}")
    render_candidate_info(candidate)

    st.divider()

    # Display spectrogram
    spectrogram_path = get_spectrogram_path(spectrograms_dir, candidate_id)
    if spectrogram_path.exists():
        st.image(str(spectrogram_path), use_container_width=True)
    else:
        st.warning(f"Spectrogram not found: {spectrogram_path}")

    st.divider()

    # Labeling controls
    current_label = get_candidate_label(candidate_id, st.session_state.labels)
    render_labeling_controls(
        candidates,
        candidate_id,
        current_label,
        labels_csv,
        current_index,
        st.session_state.labels,
    )

    # Show labeling statistics
    with st.sidebar:
        st.header("Statistics")
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


if __name__ == "__main__":
    run()
