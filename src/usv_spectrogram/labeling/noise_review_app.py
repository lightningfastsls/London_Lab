"""Streamlit app for reviewing noise samples and trimming partial USVs."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.detection.candidate import Candidate
from usv_spectrogram.detection.extraction_config import ExtractionConfig
from usv_spectrogram.detection.spectrogram_extractor import SpectrogramExtractor


LABEL_OPTIONS = ["Clean", "Contains USV", "Skip"]


def load_noise_samples(csv_path: Path) -> pd.DataFrame:
    """Load noise samples from CSV and sort by candidate_id."""
    df = pd.read_csv(csv_path)
    df = df.sort_values("candidate_id").reset_index(drop=True)
    return df


def load_existing_reviews(reviews_path: Path) -> dict[str, dict[str, Any]]:
    """Load existing reviews from CSV into a dict keyed by candidate_id."""
    reviews_dict = {}
    if reviews_path.exists():
        with open(reviews_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                reviews_dict[row["candidate_id"]] = {
                    "status": row["status"],
                    "trim_ms": float(row.get("trim_ms", 0)),
                    "reviewed_at": row["reviewed_at"],
                }
    return reviews_dict


def save_review(
    reviews_path: Path,
    candidate_id: str,
    status: str,
    trim_ms: float,
    existing_reviews: dict[str, dict[str, Any]],
) -> None:
    """Save a review to the reviews CSV file."""
    reviewed_at = datetime.now().isoformat()
    existing_reviews[candidate_id] = {
        "status": status,
        "trim_ms": trim_ms,
        "reviewed_at": reviewed_at,
    }

    with open(reviews_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "status", "trim_ms", "reviewed_at"])
        writer.writeheader()
        for cid, review_data in existing_reviews.items():
            writer.writerow({
                "candidate_id": cid,
                "status": review_data["status"],
                "trim_ms": review_data["trim_ms"],
                "reviewed_at": review_data["reviewed_at"],
            })


def get_spectrogram_path(spectrograms_dir: Path, candidate_id: str) -> Path:
    """Get path to spectrogram PNG for a noise sample."""
    return spectrograms_dir / f"{candidate_id}.png"


def initialize_session_state(
    samples: pd.DataFrame,
    reviews: dict[str, dict[str, Any]],
) -> None:
    """Initialize Streamlit session state variables."""
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "reviews" not in st.session_state:
        st.session_state.reviews = reviews
    if "total_samples" not in st.session_state:
        st.session_state.total_samples = len(samples)
    if "trim_value" not in st.session_state:
        st.session_state.trim_value = 0


def get_sample_review(candidate_id: str, reviews: dict[str, dict[str, Any]]) -> Optional[dict]:
    """Get the review for a sample, or None if not reviewed."""
    return reviews.get(candidate_id)


def count_reviewed(samples: pd.DataFrame, reviews: dict[str, dict[str, Any]]) -> int:
    """Count how many samples have been reviewed."""
    return sum(1 for cid in samples["candidate_id"] if cid in reviews)


def find_next_unreviewed(
    samples: pd.DataFrame,
    reviews: dict[str, dict[str, Any]],
    start_index: int,
) -> Optional[int]:
    """Find the next unreviewed sample starting from start_index."""
    for i in range(start_index, len(samples)):
        candidate_id = samples.iloc[i]["candidate_id"]
        if candidate_id not in reviews:
            return i
    return None


def trim_and_reextract(
    sample: pd.Series,
    trim_ms: float,
    wav_dir: Path,
    output_dir: Path,
) -> Optional[Path]:
    """Trim the segment and re-extract spectrogram.

    trim_ms > 0: Remove from start
    trim_ms < 0: Remove from end
    """
    if trim_ms == 0:
        return None

    # Calculate new boundaries
    original_start = float(sample["start_ms"])
    original_end = float(sample["end_ms"])
    context_before = original_start - float(sample["context_start_ms"])
    context_after = float(sample["context_end_ms"]) - original_end

    if trim_ms > 0:
        # Remove from start
        new_start = original_start + trim_ms
        new_end = original_end
    else:
        # Remove from end
        new_start = original_start
        new_end = original_end + trim_ms  # trim_ms is negative

    # Validate new duration
    new_duration = new_end - new_start
    if new_duration < 10:  # Minimum 10ms segment
        return None

    # Create new candidate with trimmed boundaries
    candidate_id = sample["candidate_id"]
    trimmed_id = f"{candidate_id}_trimmed"

    candidate = Candidate(
        source_file=wav_dir / sample["source_file"],
        candidate_id=trimmed_id,
        start_ms=new_start,
        end_ms=new_end,
        duration_ms=new_duration,
        context_start_ms=max(0.0, new_start - context_before),
        context_end_ms=new_end + context_after,
        peak_freq_hz=0.0,
        peak_energy_db=0.0,
        interference_flag=False,
    )

    # Extract new spectrogram
    config = ExtractionConfig(sample_rate=300000, default_render_mode="review")
    extractor = SpectrogramExtractor(config)

    return extractor.extract_single(candidate, wav_dir, output_dir, "review")


def render_navigation(samples: pd.DataFrame, reviews: dict[str, dict[str, Any]]) -> None:
    """Render navigation controls."""
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        if st.button("← Previous", disabled=st.session_state.current_index == 0):
            st.session_state.current_index -= 1
            st.session_state.trim_value = 0
            st.rerun()

    with col2:
        if st.button(
            "Next →",
            disabled=st.session_state.current_index >= len(samples) - 1,
        ):
            st.session_state.current_index += 1
            st.session_state.trim_value = 0
            st.rerun()

    with col3:
        if st.button("Jump to Unreviewed"):
            next_unreviewed = find_next_unreviewed(
                samples,
                reviews,
                st.session_state.current_index + 1,
            )
            if next_unreviewed is not None:
                st.session_state.current_index = next_unreviewed
                st.session_state.trim_value = 0
                st.rerun()
            else:
                st.info("No more unreviewed samples")

    with col4:
        reviewed_count = count_reviewed(samples, reviews)
        st.write(f"Progress: {reviewed_count} / {len(samples)} reviewed")


def render_sample_info(sample: pd.Series) -> None:
    """Render sample metadata."""
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Sample ID:** {sample['candidate_id']}")
        st.write(f"**Source File:** {sample['source_file']}")
    with col2:
        st.write(f"**Duration:** {sample['duration_ms']:.1f} ms")
        st.write(f"**Time Range:** {sample['start_ms']:.1f} - {sample['end_ms']:.1f} ms")


def render_review_controls(
    samples: pd.DataFrame,
    sample: pd.Series,
    candidate_id: str,
    current_review: Optional[dict],
    reviews_path: Path,
    current_index: int,
    reviews: dict[str, dict[str, Any]],
    wav_dir: Path,
    output_dir: Path,
) -> None:
    """Render review and trimming controls."""

    # Keyboard shortcuts
    components.html(
        """
        <script>
        (function () {
            const parentDoc = window.parent && window.parent.document ? window.parent.document : document;
            if (parentDoc._noiseShortcutBound) return;
            parentDoc._noiseShortcutBound = true;
            parentDoc.addEventListener("keydown", (event) => {
                if (event.repeat) return;
                const tag = event.target.tagName.toLowerCase();
                const isInput = ["input", "textarea", "select"].includes(tag) || event.target.isContentEditable;
                if (isInput) return;
                if (!["1", "2", "3"].includes(event.key)) return;
                event.preventDefault();
                const labelMap = { "1": "Clean", "2": "Contains USV", "3": "Skip" };
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

    if current_review:
        st.info(f"Current status: **{current_review['status']}** (trim: {current_review['trim_ms']} ms)")

    st.subheader("Review Options")

    # Quick label buttons
    col1, col2, col3 = st.columns(3)

    def apply_review(status: str, trim: float = 0) -> None:
        save_review(reviews_path, candidate_id, status, trim, reviews)
        next_idx = find_next_unreviewed(samples, reviews, current_index + 1)
        if next_idx is None:
            next_idx = current_index + 1 if current_index < len(samples) - 1 else current_index
        st.session_state.current_index = next_idx
        st.session_state.trim_value = 0
        st.rerun()

    with col1:
        if st.button("Clean (1)", use_container_width=True, type="primary"):
            apply_review("Clean", 0)

    with col2:
        if st.button("Contains USV (2)", use_container_width=True):
            # Don't auto-advance, let user specify trim
            st.session_state.show_trim = True
            st.rerun()

    with col3:
        if st.button("Skip (3)", use_container_width=True):
            apply_review("Skip", 0)

    # Trim controls (shown when "Contains USV" is selected or already has that status)
    show_trim = getattr(st.session_state, "show_trim", False) or (
        current_review and current_review["status"] == "Contains USV"
    )

    if show_trim:
        st.divider()
        st.subheader("Trim to Remove USV")
        st.markdown("""
        - **Positive value**: Remove from **start** (e.g., `20` removes first 20ms)
        - **Negative value**: Remove from **end** (e.g., `-20` removes last 20ms)
        """)

        col_trim, col_preview, col_save = st.columns([2, 1, 1])

        with col_trim:
            trim_value = st.number_input(
                "Trim amount (ms)",
                min_value=-200,
                max_value=200,
                value=int(st.session_state.trim_value),
                step=5,
                help="Positive = trim start, Negative = trim end",
            )
            st.session_state.trim_value = trim_value

        with col_preview:
            if st.button("Preview Trim"):
                with st.spinner("Generating preview..."):
                    preview_path = trim_and_reextract(
                        sample, trim_value, wav_dir, output_dir / "preview"
                    )
                    if preview_path and preview_path.exists():
                        st.session_state.preview_path = str(preview_path)
                        st.rerun()
                    else:
                        st.error("Failed to generate preview")

        with col_save:
            if st.button("Save Trimmed", type="primary"):
                if trim_value != 0:
                    # Generate final trimmed version
                    final_path = trim_and_reextract(
                        sample, trim_value, wav_dir, output_dir
                    )
                    if final_path:
                        apply_review("Trimmed", trim_value)
                    else:
                        st.error("Trim failed - segment too short")
                else:
                    st.warning("Enter a trim value first")

        # Show preview if available
        if hasattr(st.session_state, "preview_path") and st.session_state.preview_path:
            preview_path = Path(st.session_state.preview_path)
            if preview_path.exists():
                st.subheader("Trimmed Preview")
                st.image(str(preview_path), use_container_width=True)


def render_guide() -> None:
    """Render review guide."""
    with st.expander("Review Guide"):
        st.markdown("""
        ### Noise Sample Review Guide

        **Clean:** No USV patterns visible - pure background noise
        - Diffuse, random energy distribution
        - No clear frequency sweeps or arcs
        - May have vertical clicks or broadband noise

        **Contains USV:** Partial or full USV visible
        - Curved frequency sweep pattern
        - Bright arc in 50-100 kHz range
        - Use trim to remove the USV portion

        **Skip:** Unclear or corrupted sample
        - When you're not sure
        - Corrupted or unreadable spectrogram

        ### Trimming Instructions
        - **Positive trim (e.g., +20):** Removes 20ms from the START
        - **Negative trim (e.g., -20):** Removes 20ms from the END

        **Keyboard Shortcuts:**
        - Press **1** for Clean
        - Press **2** for Contains USV
        - Press **3** for Skip
        """)


def run() -> None:
    """Run the noise sample review tool."""
    st.set_page_config(page_title="Noise Sample Review", layout="wide")
    st.title("Noise Sample Review Tool")

    # Paths
    repo_root = Path(__file__).resolve().parents[3]
    noise_dir = repo_root / "noise_samples"
    samples_csv = noise_dir / "noise_samples.csv"
    reviews_csv = noise_dir / "noise_reviews.csv"
    wav_dir = repo_root / "5970 USV"

    # Load data
    if not samples_csv.exists():
        st.error(f"Noise samples CSV not found: {samples_csv}")
        st.info("Run extract_noise_samples.py first to generate noise samples.")
        return

    if not noise_dir.exists():
        st.error(f"Noise samples directory not found: {noise_dir}")
        return

    samples = load_noise_samples(samples_csv)
    existing_reviews = load_existing_reviews(reviews_csv)

    # Initialize session state
    initialize_session_state(samples, existing_reviews)

    # Render guide
    render_guide()

    # Navigation
    render_navigation(samples, st.session_state.reviews)

    st.divider()

    # Get current sample
    current_index = st.session_state.current_index
    sample = samples.iloc[current_index]
    candidate_id = sample["candidate_id"]

    # Display sample info
    st.write(f"### Sample {current_index + 1} of {len(samples)}")
    render_sample_info(sample)

    st.divider()

    # Display spectrogram
    spectrogram_path = get_spectrogram_path(noise_dir, candidate_id)
    if spectrogram_path.exists():
        st.image(str(spectrogram_path), use_container_width=True)
    else:
        st.warning(f"Spectrogram not found: {spectrogram_path}")

    st.divider()

    # Review controls
    current_review = get_sample_review(candidate_id, st.session_state.reviews)
    render_review_controls(
        samples,
        sample,
        candidate_id,
        current_review,
        reviews_csv,
        current_index,
        st.session_state.reviews,
        wav_dir,
        noise_dir,
    )

    # Sidebar statistics
    with st.sidebar:
        st.header("Statistics")
        reviewed_count = count_reviewed(samples, st.session_state.reviews)
        st.metric("Total Reviewed", f"{reviewed_count} / {len(samples)}")

        if reviewed_count > 0:
            percentage = (reviewed_count / len(samples)) * 100
            st.metric("Progress", f"{percentage:.1f}%")

            st.subheader("Review Status")
            status_counts = {"Clean": 0, "Contains USV": 0, "Trimmed": 0, "Skip": 0}
            for review_data in st.session_state.reviews.values():
                status = review_data["status"]
                if status in status_counts:
                    status_counts[status] += 1

            for status, count in status_counts.items():
                st.write(f"**{status}:** {count}")


if __name__ == "__main__":
    run()
