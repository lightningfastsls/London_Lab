"""Training data assembly pipeline.

Unifies the multi-step process of collecting labels, generating positives
(with jitter augmentation), creating negatives from 3 sources, extracting
spectrograms, splitting by recording, and writing train/val/test CSVs.

ADR compliance:
  - ADR-004: Split by recording (no leakage)
  - ADR-008: 3-source negatives (random, inter-USV gap, low-energy)
  - ADR-010: LabelStorage JSON format (parsed directly, no app dependency)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from PIL import Image
from scipy import signal as scipy_signal

from .._stft_core import compute_stft_frames_db, extract_frames
from ..detection.candidate import Candidate
from ..detection.extraction_config import ExtractionConfig
from ..detection.spectrogram_extractor import SpectrogramExtractor
from ..io_wav import load_wav_mono
from .quality_checks import (
    check_class_balance,
    check_no_duplicate_ids,
    check_no_recording_leakage,
    check_spectrogram_files_exist,
    check_split_sizes,
)
from .splits import (
    Sample,
    SplitConfig,
    _write_split_csv,
    group_samples_by_recording,
    split_recordings,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssemblyConfig:
    """Configuration for the training data assembly pipeline."""

    wav_dir: Optional[Path] = None  # WAV fallback dir (legacy or fallback for unresolved paths)
    labels_dir: Optional[Path] = None  # Dir with LabelStorage JSON files (legacy)

    # Global-MAD extraction (new pipeline)
    use_global_mad: bool = True  # True = new matched pipeline, False = legacy
    window_columns: int = 100  # STFT columns per window (matches SlidingInference)
    image_height_px: int = 256  # CNN input height
    colormap: str = "magma"
    mad_vmin_scale: float = 2.0
    mad_vmax_scale: float = 4.0

    # Minimum USV duration filter (removes spurious detections)
    min_usv_duration_ms: float = 5.0

    # Rolling window for long USVs
    rolling_stride_ms: float = 10.0  # ~23 STFT columns at hop=128/sr=300k

    # Noise recording negatives
    noise_wav_dir: Optional[Path] = None
    noise_stride_ms: float = 42.7  # ~100 columns, systematic slicing
    noise_max_ratio: float = 2.0  # Cap noise negatives at this multiple of positives

    # Unified labels input (new pipeline)
    unified_labels_path: Optional[Path] = None

    # Jitter augmentation
    jitter_n_samples: int = 2  # Jittered versions per positive
    jitter_window_ms: float = 40.0  # Window size for jittered extraction
    jitter_context_padding_ms: float = 20.0  # Padding around jitter window
    jitter_min_overlap: float = 0.5  # Min fraction of USV in jittered window

    # Negative generation fractions
    neg_random_frac: float = 0.5  # Fraction from random positions
    neg_inter_usv_frac: float = 0.3  # Fraction from gaps between USVs
    neg_low_energy_frac: float = 0.2  # Fraction from low-energy regions
    neg_ratio: float = 1.0  # Negatives per positive

    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    output_dir: Path = Path("data/training/assembled")

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        # Validate that at least one label source is provided
        if self.labels_dir is None and self.unified_labels_path is None:
            raise ValueError(
                "Must provide either labels_dir (legacy) or unified_labels_path (new)"
            )

        # Validate split ratios
        split_total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(split_total - 1.0) > 0.001:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {split_total:.3f} "
                f"({self.train_ratio} + {self.val_ratio} + {self.test_ratio})"
            )

        # Validate negative fractions
        frac_total = self.neg_random_frac + self.neg_inter_usv_frac + self.neg_low_energy_frac
        if abs(frac_total - 1.0) > 0.001:
            raise ValueError(
                f"Negative fractions must sum to 1.0, got {frac_total:.3f} "
                f"({self.neg_random_frac} + {self.neg_inter_usv_frac} + {self.neg_low_energy_frac})"
            )

        # Validate all positive
        for name, val in [
            ("jitter_n_samples", self.jitter_n_samples),
            ("jitter_window_ms", self.jitter_window_ms),
            ("jitter_context_padding_ms", self.jitter_context_padding_ms),
            ("jitter_min_overlap", self.jitter_min_overlap),
            ("neg_ratio", self.neg_ratio),
            ("train_ratio", self.train_ratio),
            ("val_ratio", self.val_ratio),
            ("test_ratio", self.test_ratio),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")

        # Convert Path fields from strings if needed
        if self.wav_dir is not None:
            object.__setattr__(self, "wav_dir", Path(self.wav_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.labels_dir is not None:
            object.__setattr__(self, "labels_dir", Path(self.labels_dir))
        if self.unified_labels_path is not None:
            object.__setattr__(self, "unified_labels_path", Path(self.unified_labels_path))
        if self.noise_wav_dir is not None:
            object.__setattr__(self, "noise_wav_dir", Path(self.noise_wav_dir))


@dataclass
class AssemblyReport:
    """Report from the assembly pipeline."""

    total_positives: int = 0
    total_negatives: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    n_recordings: int = 0
    warnings: list[str] = field(default_factory=list)
    output_dir: Path = Path(".")


class DatasetAssembler:
    """Assembles training data from label JSONs into ready-to-train CSV splits.

    Pipeline: collect labels -> create positives (with jitter) -> create negatives
    (3 sources) -> extract spectrograms -> split by recording -> validate -> write.
    """

    # DSP constants (ADR-002)
    SAMPLE_RATE = 300_000
    N_FFT = 512
    HOP_LENGTH = 128

    # Negative generation
    DETECTION_BUFFER_MS = 50.0  # Buffer around detections for negative rejection
    MIN_GAP_MS = 100.0  # Minimum gap between detections for gap negatives

    def __init__(self, config: AssemblyConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def assemble(self, dry_run: bool = False) -> AssemblyReport:
        """Run the full assembly pipeline.

        Routes to global-MAD pipeline (use_global_mad=True) or legacy
        per-candidate pipeline (use_global_mad=False).

        Parameters
        ----------
        dry_run : bool
            If True, compute statistics without extracting spectrograms or
            writing output files.

        Returns
        -------
        AssemblyReport with pipeline statistics.
        """
        cfg = self.config
        if cfg.use_global_mad:
            return self._assemble_global_mad(dry_run)
        else:
            return self._assemble_legacy(dry_run)

    def _assemble_global_mad(self, dry_run: bool = False) -> AssemblyReport:
        """New pipeline: global-MAD normalized windows matching inference."""
        cfg = self.config
        report = AssemblyReport(output_dir=cfg.output_dir)

        # Step 1: Load unified labels
        if cfg.unified_labels_path is None:
            raise ValueError("Global-MAD pipeline requires unified_labels_path")

        logger.info("Loading unified labels from %s", cfg.unified_labels_path)
        with open(cfg.unified_labels_path, "r", encoding="utf-8") as f:
            unified = json.load(f)

        positives = unified["positives"]
        noise_recordings = unified.get("noise_recordings", [])

        # Filter out spurious detections below minimum duration
        if cfg.min_usv_duration_ms > 0:
            min_dur_s = cfg.min_usv_duration_ms / 1000.0
            before = len(positives)
            positives = [
                p for p in positives
                if (p["end_s"] - p["start_s"]) >= min_dur_s
            ]
            filtered = before - len(positives)
            if filtered > 0:
                logger.info("Filtered %d detections below %.1fms", filtered, cfg.min_usv_duration_ms)

        # Build WAV path index: stem -> Path
        # Uses wav_path from unified labels if available, falls back to wav_dir
        wav_path_index: dict[str, Path] = {}
        for p in positives:
            stem = p["recording_stem"]
            if stem in wav_path_index:
                continue
            wp = p.get("wav_path")
            if wp:
                wav_path_index[stem] = Path(wp)
            elif cfg.wav_dir:
                # Fallback to wav_dir
                candidate = cfg.wav_dir / f"{stem}.wav"
                if candidate.exists():
                    wav_path_index[stem] = candidate

        # Also index noise recordings
        for n in noise_recordings:
            stem = n["recording_stem"]
            if stem in wav_path_index:
                continue
            wp = n.get("wav_path")
            if wp:
                wav_path_index[stem] = Path(wp)
            elif cfg.noise_wav_dir:
                candidate = cfg.noise_wav_dir / f"{stem}.wav"
                if candidate.exists():
                    wav_path_index[stem] = candidate

        # Group positives by recording
        labels_by_recording: dict[str, list[dict]] = {}
        for p in positives:
            stem = p["recording_stem"]
            labels_by_recording.setdefault(stem, []).append(p)

        report.n_recordings = len(labels_by_recording)
        n_with_wav = sum(1 for s in labels_by_recording if s in wav_path_index)
        logger.info(
            "Loaded %d positives across %d recordings (%d with WAV)",
            len(positives), report.n_recordings, n_with_wav,
        )

        # Step 2: Create positive windows
        logger.info("Creating positive windows (global-MAD, matched to inference)")
        positive_samples = self._create_positive_windows_global(
            labels_by_recording, wav_path_index, cfg.output_dir, dry_run,
        )
        report.total_positives = len(positive_samples)
        logger.info("Created %d positive windows", report.total_positives)

        # Step 3: Create negative windows (3 sources + noise recordings)
        logger.info("Creating negative windows (3-source, global-MAD)")
        negative_samples = self._create_negative_windows_global(
            labels_by_recording, wav_path_index, cfg.output_dir,
            report.total_positives, spec_cache={}, dry_run=dry_run,
        )
        logger.info("Created %d negative windows (3-source)", len(negative_samples))

        # Step 4: Noise recording negatives
        noise_negatives: list[Sample] = []
        if noise_recordings:
            max_noise = int(report.total_positives * cfg.noise_max_ratio)
            noise_negatives = self._create_noise_recording_negatives(
                noise_recordings, wav_path_index, cfg.output_dir,
                max_noise, dry_run,
            )

        all_negatives = negative_samples + noise_negatives
        report.total_negatives = len(all_negatives)
        logger.info("Total negatives: %d", report.total_negatives)

        # Step 5: Split by recording (ADR-004)
        all_samples = positive_samples + all_negatives
        logger.info("Splitting %d samples by recording", len(all_samples))
        splits = self._create_splits(all_samples)
        report.train_count = len(splits.get("train", []))
        report.val_count = len(splits.get("val", []))
        report.test_count = len(splits.get("test", []))

        # Step 6: Validate
        logger.info("Running quality checks")
        report.warnings = self._validate(splits, dry_run=dry_run)
        for warning in report.warnings:
            logger.warning("Quality check: %s", warning)

        # Step 7: Write output
        if not dry_run:
            self._write_csvs(splits)
            self._write_report(report, cfg)
            logger.info("Assembly complete. Output: %s", cfg.output_dir)

        return report

    def _assemble_legacy(self, dry_run: bool = False) -> AssemblyReport:
        """Legacy pipeline: per-candidate extraction with variable-width images."""
        cfg = self.config
        report = AssemblyReport(output_dir=cfg.output_dir)

        if cfg.labels_dir is None:
            raise ValueError("Legacy pipeline requires labels_dir")
        if cfg.wav_dir is None:
            raise ValueError("Legacy pipeline requires wav_dir")

        # Step 1: Collect labels from JSON files
        logger.info("Collecting labels from %s", cfg.labels_dir)
        labels_df = self._collect_labels()
        report.n_recordings = labels_df["source_file"].nunique()
        logger.info(
            "Found %d detections across %d recordings",
            len(labels_df),
            report.n_recordings,
        )

        # Step 2: Create positive candidates (with jitter augmentation)
        logger.info("Creating positive candidates with jitter augmentation")
        positive_candidates = self._create_positive_candidates(labels_df)
        report.total_positives = len(positive_candidates)
        logger.info("Created %d positive candidates (including jitter)", report.total_positives)

        # Step 3: Create negative candidates from 3 sources
        logger.info("Creating negative candidates (3-source)")
        negative_candidates = self._create_negative_candidates(labels_df, report.total_positives)
        report.total_negatives = len(negative_candidates)
        logger.info("Created %d negative candidates", report.total_negatives)

        # Step 4: Extract spectrograms (unless dry-run)
        all_candidates = positive_candidates + negative_candidates
        if not dry_run:
            logger.info("Extracting spectrograms for %d candidates", len(all_candidates))
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            self._extract_spectrograms(all_candidates)

        # Step 5: Convert to samples and split
        positive_samples = self._candidates_to_samples(positive_candidates, "USV")
        negative_samples = self._candidates_to_samples(negative_candidates, "Not USV")
        all_samples = positive_samples + negative_samples

        logger.info("Splitting %d samples by recording", len(all_samples))
        splits = self._create_splits(all_samples)
        report.train_count = len(splits.get("train", []))
        report.val_count = len(splits.get("val", []))
        report.test_count = len(splits.get("test", []))

        # Step 6: Validate
        logger.info("Running quality checks")
        report.warnings = self._validate(splits, dry_run=dry_run)
        for warning in report.warnings:
            logger.warning("Quality check: %s", warning)

        # Step 7: Write output (unless dry-run)
        if not dry_run:
            self._write_csvs(splits)
            self._write_report(report, cfg)
            logger.info("Assembly complete. Output: %s", cfg.output_dir)

        return report

    # ── Label Collection ──────────────────────────────────────────────

    def _collect_labels(self) -> pd.DataFrame:
        """Parse LabelStorage JSON files and return a DataFrame of detections.

        Returns
        -------
        DataFrame with columns: source_file, start_time_s, end_time_s,
            duration_s, max_probability
        """
        cfg = self.config
        json_files = sorted(cfg.labels_dir.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {cfg.labels_dir}")

        rows: list[dict] = []
        for json_path in json_files:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            metadata = data.get("metadata", {})
            wav_file = metadata.get("wav_file", "")
            # Use full filename with .wav extension (consistent with existing pipeline)
            source_stem = Path(wav_file).name if wav_file else f"{json_path.stem}.wav"

            detections = data.get("detections", [])
            for det in detections:
                # Skip user-deleted detections
                if det.get("user_action") == "deleted_by_user":
                    continue

                rows.append({
                    "source_file": source_stem,
                    "start_time_s": det["start_time_s"],
                    "end_time_s": det["end_time_s"],
                    "duration_s": det.get("duration_s", det["end_time_s"] - det["start_time_s"]),
                    "max_probability": det.get("max_probability", 0.0),
                })

        if not rows:
            raise ValueError(
                f"No valid detections found in {len(json_files)} JSON files in {cfg.labels_dir}"
            )

        return pd.DataFrame(rows)

    # ── Positive Candidate Creation ───────────────────────────────────

    def _create_positive_candidates(self, labels_df: pd.DataFrame) -> list[Candidate]:
        """Create positive candidates from detections, with jitter augmentation."""
        cfg = self.config
        candidates: list[Candidate] = []

        for _, row in labels_df.iterrows():
            source_file = Path(row["source_file"])
            start_ms = row["start_time_s"] * 1000.0
            end_ms = row["end_time_s"] * 1000.0

            # Original candidate
            original = Candidate.create(
                source_file=source_file,
                start_ms=start_ms,
                end_ms=end_ms,
                peak_freq_hz=0.0,
                peak_energy_db=0.0,
                context_before_ms=cfg.jitter_context_padding_ms,
                context_after_ms=cfg.jitter_context_padding_ms,
            )
            candidates.append(original)

            # Get recording duration for jitter bounds
            wav_path = cfg.wav_dir / row["source_file"]
            if wav_path.exists():
                info = sf.info(str(wav_path))
                recording_dur_ms = info.duration * 1000.0
            else:
                # Fallback: estimate from last detection in this recording
                rec_dets = labels_df[labels_df["source_file"] == row["source_file"]]
                recording_dur_ms = rec_dets["end_time_s"].max() * 1000.0 + 5000.0

            # Jittered versions
            jittered = self._jitter_candidate(
                original, recording_dur_ms, original.candidate_id
            )
            candidates.extend(jittered)

        return candidates

    def _jitter_candidate(
        self,
        candidate: Candidate,
        recording_duration_ms: float,
        base_id: str,
    ) -> list[Candidate]:
        """Generate jittered versions of a candidate.

        Each jitter shifts the extraction window while maintaining minimum
        overlap with the original USV.
        """
        cfg = self.config
        window_ms = cfg.jitter_window_ms
        usv_dur = candidate.duration_ms
        min_overlap = cfg.jitter_min_overlap

        # Compute valid range for window start.
        # Window must contain at least min_overlap fraction of the USV.
        #
        # Critical duration threshold: jitter is impossible when
        #   usv_dur >= window_ms / (2 * min_overlap)
        # With defaults (40ms window, 0.5 overlap), USVs >= 40ms get no jitter.
        min_start = max(0.0, candidate.end_ms - window_ms + min_overlap * usv_dur)
        max_start = min(
            candidate.start_ms + usv_dur * (1.0 - min_overlap),
            recording_duration_ms - window_ms,
        )

        if max_start <= min_start:
            # Can't jitter — window too small for this USV duration
            return []

        jittered: list[Candidate] = []
        for i in range(cfg.jitter_n_samples):
            window_start = self._rng.uniform(min_start, max_start)
            context_start = max(0.0, window_start - cfg.jitter_context_padding_ms)
            context_end = min(
                recording_duration_ms,
                window_start + window_ms + cfg.jitter_context_padding_ms,
            )

            jit_id = f"{base_id}_jit{i:02d}"
            jit_candidate = Candidate(
                source_file=candidate.source_file,
                candidate_id=jit_id,
                start_ms=candidate.start_ms,
                end_ms=candidate.end_ms,
                duration_ms=candidate.duration_ms,
                context_start_ms=context_start,
                context_end_ms=context_end,
                peak_freq_hz=0.0,
                peak_energy_db=0.0,
            )
            jittered.append(jit_candidate)

        return jittered

    # ── Negative Candidate Creation ───────────────────────────────────

    def _create_negative_candidates(
        self, labels_df: pd.DataFrame, n_positives: int
    ) -> list[Candidate]:
        """Create negative candidates from 3 sources, distributed across recordings."""
        cfg = self.config
        n_total = int(n_positives * cfg.neg_ratio)
        if n_total == 0:
            return []

        candidates: list[Candidate] = []
        recordings = labels_df.groupby("source_file")

        # Distribute negatives proportional to detection count per recording
        total_detections = len(labels_df)
        for source_file, rec_df in recordings:
            proportion = len(rec_df) / total_detections
            n_rec = max(1, round(n_total * proportion))

            # Largest-remainder allocation (avoids rounding starving a category)
            fracs = [cfg.neg_random_frac, cfg.neg_inter_usv_frac, cfg.neg_low_energy_frac]
            floors = [int(n_rec * f) for f in fracs]
            remainder = n_rec - sum(floors)
            # Give remainder to categories with largest fractional parts
            fractional = sorted(
                enumerate(n_rec * f - int(n_rec * f) for f in fracs),
                key=lambda x: x[1], reverse=True,
            )
            for idx, _ in fractional[:remainder]:
                floors[idx] += 1
            n_random, n_gap, n_low_energy = floors

            # Detection times in ms for this recording
            detections_ms = list(zip(
                rec_df["start_time_s"].values * 1000.0,
                rec_df["end_time_s"].values * 1000.0,
            ))

            # Get recording duration
            wav_path = cfg.wav_dir / source_file
            if wav_path.exists():
                info = sf.info(str(wav_path))
                recording_dur_ms = info.duration * 1000.0
            else:
                recording_dur_ms = rec_df["end_time_s"].max() * 1000.0 + 5000.0

            source_name = str(source_file)  # Full name with .wav extension
            source_stem = Path(source_file).stem  # Stem for candidate IDs

            # Generate from each source
            candidates.extend(
                self._random_negatives(source_name, source_stem, detections_ms, recording_dur_ms, n_random)
            )
            candidates.extend(
                self._gap_negatives(source_name, source_stem, detections_ms, recording_dur_ms, n_gap)
            )
            if n_low_energy > 0:
                candidates.extend(
                    self._low_energy_negatives(
                        source_name, source_stem, wav_path, detections_ms, n_low_energy
                    )
                )

        # Log if actual count overshoots the requested total (due to max(1,...) floor)
        if len(candidates) > n_total * 1.2:
            logger.warning(
                "Negative count overshoot: requested %d, generating %d "
                "(due to min-1 floor across %d recordings)",
                n_total, len(candidates), len(recordings),
            )

        return candidates

    def _random_negatives(
        self,
        source_name: str,
        source_stem: str,
        detections_ms: list[tuple[float, float]],
        recording_dur_ms: float,
        n: int,
    ) -> list[Candidate]:
        """Generate random negative candidates that don't overlap detections."""
        if n <= 0:
            return []

        cfg = self.config
        window_ms = cfg.jitter_window_ms
        buffer = self.DETECTION_BUFFER_MS
        max_attempts = n * 20
        candidates: list[Candidate] = []

        for attempt in range(max_attempts):
            if len(candidates) >= n:
                break

            start = self._rng.uniform(0, max(0, recording_dur_ms - window_ms))
            end = start + window_ms

            # Reject if within buffer of any detection
            overlaps = any(
                start < det_end + buffer and end > det_start - buffer
                for det_start, det_end in detections_ms
            )
            if overlaps:
                continue

            cid = f"{source_stem}_neg_rand_{len(candidates):04d}"
            candidates.append(Candidate(
                source_file=Path(source_name),
                candidate_id=cid,
                start_ms=start,
                end_ms=end,
                duration_ms=window_ms,
                context_start_ms=max(0.0, start - cfg.jitter_context_padding_ms),
                context_end_ms=min(recording_dur_ms, end + cfg.jitter_context_padding_ms),
                peak_freq_hz=0.0,
                peak_energy_db=0.0,
            ))

        if len(candidates) < n:
            logger.warning(
                "Could only generate %d/%d random negatives for %s",
                len(candidates), n, source_stem,
            )
        return candidates

    def _gap_negatives(
        self,
        source_name: str,
        source_stem: str,
        detections_ms: list[tuple[float, float]],
        recording_dur_ms: float,
        n: int,
    ) -> list[Candidate]:
        """Generate negatives from gaps between consecutive detections."""
        if n <= 0:
            return []

        cfg = self.config
        window_ms = cfg.jitter_window_ms
        min_gap = self.MIN_GAP_MS

        # Sort detections by start time
        sorted_dets = sorted(detections_ms, key=lambda d: d[0])

        # Find eligible gaps
        gap_midpoints: list[float] = []
        for i in range(len(sorted_dets) - 1):
            gap_start = sorted_dets[i][1]  # end of current
            gap_end = sorted_dets[i + 1][0]  # start of next
            gap_size = gap_end - gap_start
            if gap_size >= min_gap:
                gap_midpoints.append((gap_start + gap_end) / 2.0)

        if not gap_midpoints:
            logger.debug("No gaps >= %.0f ms in %s", min_gap, source_stem)
            return []

        candidates: list[Candidate] = []
        # Sample from gap midpoints (with replacement if needed)
        indices = self._rng.choice(len(gap_midpoints), size=min(n, len(gap_midpoints) * 3), replace=True)
        seen_positions: set[int] = set()

        for idx in indices:
            if len(candidates) >= n:
                break
            if idx in seen_positions and len(gap_midpoints) >= n:
                continue
            seen_positions.add(idx)

            midpoint = gap_midpoints[idx]
            start = midpoint - window_ms / 2.0
            end = start + window_ms

            # Clamp to recording bounds
            if start < 0:
                start = 0.0
                end = window_ms
            if end > recording_dur_ms:
                end = recording_dur_ms
                start = max(0.0, end - window_ms)

            cid = f"{source_stem}_neg_gap_{len(candidates):04d}"
            candidates.append(Candidate(
                source_file=Path(source_name),
                candidate_id=cid,
                start_ms=start,
                end_ms=end,
                duration_ms=window_ms,
                context_start_ms=max(0.0, start - cfg.jitter_context_padding_ms),
                context_end_ms=min(recording_dur_ms, end + cfg.jitter_context_padding_ms),
                peak_freq_hz=0.0,
                peak_energy_db=0.0,
            ))

        if len(candidates) < n:
            logger.debug(
                "Only generated %d/%d gap negatives for %s",
                len(candidates), n, source_stem,
            )
        return candidates

    def _low_energy_negatives(
        self,
        source_name: str,
        source_stem: str,
        wav_path: Path,
        detections_ms: list[tuple[float, float]],
        n: int,
    ) -> list[Candidate]:
        """Generate negatives from low-energy regions in the USV frequency band."""
        if n <= 0 or not wav_path.exists():
            return []

        cfg = self.config
        window_ms = cfg.jitter_window_ms
        buffer = self.DETECTION_BUFFER_MS

        # Load WAV and compute band-limited energy
        samples, sr = load_wav_mono(wav_path)
        if sr != self.SAMPLE_RATE:
            logger.warning(
                "Skipping low-energy negatives for %s: sample rate %d != expected %d",
                source_stem, sr, self.SAMPLE_RATE,
            )
            return []

        recording_dur_ms = len(samples) / sr * 1000.0

        # STFT (ADR-002 compliant: n_fft=512, hop=128, sr=300000)
        freqs, times, Zxx = scipy_signal.stft(
            samples, fs=sr, nperseg=self.N_FFT, noverlap=self.N_FFT - self.HOP_LENGTH,
            window="hann",
        )

        # USV band mask (20-120 kHz)
        band_mask = (freqs >= 20_000) & (freqs <= 120_000)
        magnitude = np.abs(Zxx[band_mask, :])

        # Peak energy per frame in USV band
        frame_energy = magnitude.max(axis=0)

        # Find frames below 20th percentile AND outside detection buffer zones
        threshold = np.percentile(frame_energy, 20)
        low_energy_mask = frame_energy <= threshold

        frame_times_ms = times * 1000.0

        # Exclude frames within buffer of any detection (mask at frame level)
        safe_mask = np.ones(len(frame_times_ms), dtype=bool)
        for det_start, det_end in detections_ms:
            safe_mask &= ~(
                (frame_times_ms >= det_start - buffer)
                & (frame_times_ms <= det_end + buffer)
            )

        eligible_mask = low_energy_mask & safe_mask

        # Group contiguous eligible frames into regions
        regions: list[tuple[float, float]] = []
        in_region = False
        region_start = 0.0

        for i, is_eligible in enumerate(eligible_mask):
            t_ms = frame_times_ms[i]
            if is_eligible and not in_region:
                region_start = t_ms
                in_region = True
            elif not is_eligible and in_region:
                regions.append((region_start, t_ms))
                in_region = False
        if in_region:
            regions.append((region_start, frame_times_ms[-1]))

        # Filter: regions must be wide enough for a window
        eligible: list[tuple[float, float]] = []
        for reg_start, reg_end in regions:
            if reg_end - reg_start >= window_ms:
                eligible.append((reg_start, reg_end))

        if not eligible:
            logger.debug("No eligible low-energy regions in %s", source_stem)
            return []

        # Sample windows from eligible regions
        candidates: list[Candidate] = []
        max_attempts = n * 10
        for _ in range(max_attempts):
            if len(candidates) >= n:
                break

            # Pick a random eligible region (weighted by region length)
            lengths = np.array([r[1] - r[0] for r in eligible])
            probs = lengths / lengths.sum()
            reg_idx = self._rng.choice(len(eligible), p=probs)
            reg_start, reg_end = eligible[reg_idx]

            start = self._rng.uniform(reg_start, max(reg_start, reg_end - window_ms))
            end = start + window_ms

            cid = f"{source_stem}_neg_lowE_{len(candidates):04d}"
            candidates.append(Candidate(
                source_file=Path(source_name),
                candidate_id=cid,
                start_ms=start,
                end_ms=end,
                duration_ms=window_ms,
                context_start_ms=max(0.0, start - cfg.jitter_context_padding_ms),
                context_end_ms=min(recording_dur_ms, end + cfg.jitter_context_padding_ms),
                peak_freq_hz=0.0,
                peak_energy_db=0.0,
            ))

        if len(candidates) < n:
            logger.debug(
                "Only generated %d/%d low-energy negatives for %s",
                len(candidates), n, source_stem,
            )
        return candidates

    # ── Global-MAD Pipeline (new, matched to inference) ─────────────

    def _load_global_spectrogram(
        self, wav_path: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load WAV and compute globally MAD-normalized spectrogram.

        This matches AudioLoader._compute_spectrogram() + SlidingInference
        ._apply_mad_normalization() exactly, so training windows are
        identical to what the CNN sees at inference time.

        Returns
        -------
        spec_norm : ndarray, shape (n_freq_bins, n_frames)
            Spectrogram normalized to [0, 1] via global MAD.
        times_s : ndarray, shape (n_frames,)
            Center time of each STFT frame in seconds.
        """
        cfg = self.config

        # Step 1: Load WAV
        samples, sr = load_wav_mono(wav_path)
        if sr != self.SAMPLE_RATE:
            raise ValueError(
                f"Sample rate mismatch: {wav_path.name} has sr={sr}, "
                f"expected {self.SAMPLE_RATE}"
            )

        # Step 2: Extract frames (matches AudioLoader)
        frames = extract_frames(samples, self.N_FFT, self.HOP_LENGTH)
        if frames.shape[0] == 0:
            raise ValueError(f"No frames extracted from {wav_path.name}")

        # Step 3: Window function
        window = scipy_signal.get_window("hann", self.N_FFT, fftbins=True)

        # Step 4: Frequency band mask (20-120 kHz)
        freqs_hz = np.fft.rfftfreq(self.N_FFT, d=1.0 / self.SAMPLE_RATE)
        band_mask = (freqs_hz >= 20_000) & (freqs_hz <= 120_000)

        # Step 5: STFT → dB (normalize_magnitude=True matches inference)
        spec_db = compute_stft_frames_db(
            frames, window, self.N_FFT, band_mask,
            eps=1e-10, normalize_magnitude=True,
        )
        # spec_db shape: (n_freq_bins, n_frames)

        # Step 6: Global MAD normalization (matches SlidingInference._apply_mad_normalization)
        median = np.median(spec_db)
        mad = np.median(np.abs(spec_db - median))
        vmin = median - cfg.mad_vmin_scale * mad
        vmax = median + cfg.mad_vmax_scale * mad

        spec_clipped = np.clip(spec_db, vmin, vmax)
        if vmax > vmin:
            spec_norm = (spec_clipped - vmin) / (vmax - vmin + 1e-12)
        else:
            spec_norm = np.zeros_like(spec_db)

        # Step 7: Time bins (matches AudioLoader)
        n_frames = frames.shape[0]
        times_s = (
            (np.arange(n_frames) * self.HOP_LENGTH) + self.N_FFT / 2.0
        ) / self.SAMPLE_RATE

        return spec_norm, times_s

    def _render_window_png(
        self, window: np.ndarray, output_path: Path
    ) -> None:
        """Render a single STFT window to PNG, matching inference pipeline.

        Steps match SlidingInference._prepare_batch() exactly:
        1. Apply magma colormap
        2. Flip vertically (low freq at bottom)
        3. Convert to uint8
        4. Resize height to 256 (width stays at window_columns)
        5. Save as PNG
        """
        cfg = self.config

        # Colormap → RGB
        cmap = plt.get_cmap(cfg.colormap)
        rgb = cmap(window)[:, :, :3]  # drop alpha

        # Flip vertically
        rgb = np.flipud(rgb)

        # uint8
        rgb_uint8 = (rgb * 255).astype(np.uint8)

        # Resize height only to match inference (_prepare_batch resizes height,
        # not width). Width should already be window_columns from the slice.
        img = Image.fromarray(rgb_uint8)
        current_w, current_h = img.size
        target_h = cfg.image_height_px
        if current_h != target_h:
            img = img.resize((current_w, target_h), Image.LANCZOS)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Saved as RGB; data loader applies grayscale via PIL convert('L'),
        # matching the manual conversion in SlidingInference._prepare_batch.
        img.save(str(output_path))

    def _time_to_col(self, time_s: float) -> int:
        """Convert a time in seconds to the nearest STFT column index.

        Accounts for the n_fft/2 centering offset used in the time axis
        computation in _load_global_spectrogram.
        """
        sample_idx = time_s * self.SAMPLE_RATE
        return max(0, int(round((sample_idx - self.N_FFT / 2.0) / self.HOP_LENGTH)))

    def _create_positive_windows_global(
        self,
        labels_by_recording: dict[str, list[dict]],
        wav_path_index: dict[str, Path],
        output_dir: Path,
        dry_run: bool = False,
    ) -> list[Sample]:
        """Create positive training windows using global-MAD pipeline.

        For each recording:
        - Short USVs (≤ window_columns): 1 centered window + jittered windows
        - Long USVs (> window_columns): rolling windows with stride
        """
        cfg = self.config
        wc = cfg.window_columns
        stride_cols = max(1, self._time_to_col(cfg.rolling_stride_ms / 1000.0))
        samples: list[Sample] = []

        for recording_stem, labels in labels_by_recording.items():
            wav_path = wav_path_index.get(recording_stem)
            if wav_path is None or not wav_path.exists():
                logger.warning("WAV not found for %s (skipping %d labels)", recording_stem, len(labels))
                continue

            # Load global spectrogram (once per recording)
            try:
                spec_norm, times_s = self._load_global_spectrogram(wav_path)
            except ValueError as e:
                logger.warning("Skipping %s: %s", recording_stem, e)
                continue

            n_cols = spec_norm.shape[1]

            for label in labels:
                start_col = self._time_to_col(label["start_s"])
                end_col = self._time_to_col(label["end_s"])
                usv_span = end_col - start_col

                if usv_span <= wc:
                    # Short USV: centered window + jitter
                    windows = self._jitter_windows_for_short_usv(
                        start_col, end_col, n_cols, wc
                    )
                else:
                    # Long USV: rolling windows
                    windows = self._rolling_windows_for_long_usv(
                        start_col, end_col, n_cols, wc, stride_cols
                    )

                for i, (win_start, win_end) in enumerate(windows):
                    window_data = spec_norm[:, win_start:win_end]
                    if window_data.shape[1] != wc:
                        continue  # Skip edge windows that don't fit

                    suffix = f"_jit{i:02d}" if usv_span <= wc and i > 0 else ""
                    if usv_span > wc:
                        suffix = f"_roll{win_start:05d}"
                    sample_id = f"{recording_stem}_{start_col:06d}{suffix}"

                    spec_path = ""
                    if not dry_run:
                        png_path = output_dir / "spectrograms" / f"{sample_id}.png"
                        self._render_window_png(window_data, png_path)
                        spec_path = str(png_path)

                    samples.append(Sample(
                        candidate_id=sample_id,
                        source_file=f"{recording_stem}.wav",
                        label="USV",
                        spectrogram_path=spec_path,
                    ))

        return samples

    def _jitter_windows_for_short_usv(
        self,
        start_col: int,
        end_col: int,
        n_cols: int,
        wc: int,
    ) -> list[tuple[int, int]]:
        """Generate centered + jittered windows for a short USV.

        Returns list of (win_start, win_end) tuples.
        """
        cfg = self.config
        usv_center = (start_col + end_col) // 2
        usv_span = end_col - start_col
        min_overlap_cols = max(1, int(usv_span * cfg.jitter_min_overlap))

        # Centered window (original)
        centered_start = max(0, usv_center - wc // 2)
        centered_start = min(centered_start, max(0, n_cols - wc))
        windows = [(centered_start, centered_start + wc)]

        # Jittered windows: must contain ≥ min_overlap_cols of USV
        # Valid range for window start:
        #   earliest: USV end - wc + min_overlap_cols (window catches the tail)
        #   latest: USV start + usv_span - min_overlap_cols (window catches the head)
        earliest = max(0, end_col - wc)
        latest = min(n_cols - wc, start_col + usv_span - min_overlap_cols)

        if latest > earliest:
            for _ in range(cfg.jitter_n_samples):
                jit_start = int(self._rng.integers(earliest, latest + 1))
                windows.append((jit_start, jit_start + wc))

        return windows

    def _rolling_windows_for_long_usv(
        self,
        start_col: int,
        end_col: int,
        n_cols: int,
        wc: int,
        stride_cols: int,
    ) -> list[tuple[int, int]]:
        """Generate rolling windows for a long USV.

        Returns list of (win_start, win_end) tuples covering the USV.
        """
        windows = []
        pos = max(0, start_col)
        max_start = min(end_col - wc, n_cols - wc)

        while pos <= max_start:
            windows.append((pos, pos + wc))
            pos += stride_cols

        # Ensure we don't miss the tail
        if windows and windows[-1][1] < end_col and max_start >= 0:
            last_start = min(max_start, n_cols - wc)
            if last_start >= 0:
                windows.append((last_start, last_start + wc))

        return windows

    def _create_negative_windows_global(
        self,
        labels_by_recording: dict[str, list[dict]],
        wav_path_index: dict[str, Path],
        output_dir: Path,
        n_positives: int,
        spec_cache: dict[str, tuple[np.ndarray, np.ndarray]],
        dry_run: bool = False,
    ) -> list[Sample]:
        """Create negative windows from 3 sources using global-MAD pipeline.

        Uses cached spectrograms from positive window creation.
        """
        cfg = self.config
        wc = cfg.window_columns
        n_total = int(n_positives * cfg.neg_ratio)
        if n_total == 0:
            return []

        samples: list[Sample] = []
        buffer_cols = self._time_to_col(self.DETECTION_BUFFER_MS / 1000.0)
        total_labels = sum(len(v) for v in labels_by_recording.values())

        for recording_stem, labels in labels_by_recording.items():
            wav_path = wav_path_index.get(recording_stem)

            # Load or use cached spectrogram
            if recording_stem in spec_cache:
                spec_norm, times_s = spec_cache[recording_stem]
            elif wav_path and wav_path.exists():
                try:
                    spec_norm, times_s = self._load_global_spectrogram(wav_path)
                except ValueError:
                    continue
            else:
                continue

            n_cols = spec_norm.shape[1]
            if n_cols < wc:
                continue

            # Proportion of negatives for this recording
            proportion = len(labels) / total_labels
            n_rec = max(1, round(n_total * proportion))

            # Allocate across 3 sources
            n_random = max(1, round(n_rec * cfg.neg_random_frac))
            n_gap = max(0, round(n_rec * cfg.neg_inter_usv_frac))
            n_low_e = max(0, n_rec - n_random - n_gap)

            # Detection column ranges (for exclusion)
            det_ranges = [
                (self._time_to_col(l["start_s"]), self._time_to_col(l["end_s"]))
                for l in labels
            ]

            def overlaps_detection(start: int, end: int) -> bool:
                return any(
                    start < d_end + buffer_cols and end > d_start - buffer_cols
                    for d_start, d_end in det_ranges
                )

            # Source 1: Random negatives
            neg_count = 0
            for attempt in range(n_random * 20):
                if neg_count >= n_random:
                    break
                win_start = int(self._rng.integers(0, max(1, n_cols - wc)))
                win_end = win_start + wc
                if win_end > n_cols or overlaps_detection(win_start, win_end):
                    continue
                window_data = spec_norm[:, win_start:win_end]
                sid = f"{recording_stem}_neg_rand_{neg_count:04d}"
                spec_path = ""
                if not dry_run:
                    png_path = output_dir / "spectrograms" / f"{sid}.png"
                    self._render_window_png(window_data, png_path)
                    spec_path = str(png_path)
                samples.append(Sample(
                    candidate_id=sid,
                    source_file=f"{recording_stem}.wav",
                    label="Not USV",
                    spectrogram_path=spec_path,
                ))
                neg_count += 1

            # Source 2: Gap negatives (between consecutive detections)
            sorted_dets = sorted(det_ranges, key=lambda d: d[0])
            gap_centers: list[int] = []
            min_gap_cols = self._time_to_col(self.MIN_GAP_MS / 1000.0)
            for i in range(len(sorted_dets) - 1):
                gap_start = sorted_dets[i][1]
                gap_end = sorted_dets[i + 1][0]
                if gap_end - gap_start >= min_gap_cols:
                    gap_centers.append((gap_start + gap_end) // 2)

            neg_count = 0
            if gap_centers and n_gap > 0:
                indices = self._rng.choice(
                    len(gap_centers),
                    size=min(n_gap * 3, len(gap_centers) * 3),
                    replace=True,
                )
                for idx in indices:
                    if neg_count >= n_gap:
                        break
                    center = gap_centers[idx]
                    win_start = max(0, min(center - wc // 2, n_cols - wc))
                    win_end = win_start + wc
                    if win_end > n_cols:
                        continue
                    window_data = spec_norm[:, win_start:win_end]
                    sid = f"{recording_stem}_neg_gap_{neg_count:04d}"
                    spec_path = ""
                    if not dry_run:
                        png_path = output_dir / "spectrograms" / f"{sid}.png"
                        self._render_window_png(window_data, png_path)
                        spec_path = str(png_path)
                    samples.append(Sample(
                        candidate_id=sid,
                        source_file=f"{recording_stem}.wav",
                        label="Not USV",
                        spectrogram_path=spec_path,
                    ))
                    neg_count += 1

            # Source 3: Low-energy negatives
            if n_low_e > 0:
                # Find frames with low energy in USV band
                frame_max = spec_norm.max(axis=0)  # max per column
                threshold = np.percentile(frame_max, 20)
                low_mask = frame_max <= threshold

                # Exclude detection buffer zones
                safe_mask = np.ones(n_cols, dtype=bool)
                for d_start, d_end in det_ranges:
                    lo = max(0, d_start - buffer_cols)
                    hi = min(n_cols, d_end + buffer_cols)
                    safe_mask[lo:hi] = False

                eligible = low_mask & safe_mask
                eligible_indices = np.where(eligible)[0]

                neg_count = 0
                if len(eligible_indices) >= wc:
                    for _ in range(n_low_e * 10):
                        if neg_count >= n_low_e:
                            break
                        idx = int(self._rng.choice(eligible_indices))
                        win_start = max(0, min(idx - wc // 2, n_cols - wc))
                        win_end = win_start + wc
                        if win_end > n_cols:
                            continue
                        window_data = spec_norm[:, win_start:win_end]
                        sid = f"{recording_stem}_neg_lowE_{neg_count:04d}"
                        spec_path = ""
                        if not dry_run:
                            png_path = output_dir / "spectrograms" / f"{sid}.png"
                            self._render_window_png(window_data, png_path)
                            spec_path = str(png_path)
                        samples.append(Sample(
                            candidate_id=sid,
                            source_file=f"{recording_stem}.wav",
                            label="Not USV",
                            spectrogram_path=spec_path,
                        ))
                        neg_count += 1

        return samples

    def _create_noise_recording_negatives(
        self,
        noise_recordings: list[dict],
        wav_path_index: dict[str, Path],
        output_dir: Path,
        max_samples: int,
        dry_run: bool = False,
    ) -> list[Sample]:
        """Create negative windows by systematically slicing noise recordings."""
        cfg = self.config
        wc = cfg.window_columns
        stride_cols = max(1, self._time_to_col(cfg.noise_stride_ms / 1000.0))
        samples: list[Sample] = []

        for rec in noise_recordings:
            if len(samples) >= max_samples:
                break

            stem = rec["recording_stem"]
            wav_path = wav_path_index.get(stem)
            if wav_path is None or not wav_path.exists():
                logger.debug("Noise WAV not found for: %s", stem)
                continue

            try:
                spec_norm, times_s = self._load_global_spectrogram(wav_path)
            except ValueError as e:
                logger.debug("Skipping noise recording %s: %s", stem, e)
                continue

            n_cols = spec_norm.shape[1]
            if n_cols < wc:
                continue

            # Systematic slicing
            win_idx = 0
            pos = 0
            while pos + wc <= n_cols and len(samples) < max_samples:
                window_data = spec_norm[:, pos:pos + wc]
                sid = f"{stem}_noise_{win_idx:04d}"
                spec_path = ""
                if not dry_run:
                    png_path = output_dir / "spectrograms" / f"{sid}.png"
                    self._render_window_png(window_data, png_path)
                    spec_path = str(png_path)
                samples.append(Sample(
                    candidate_id=sid,
                    source_file=f"{stem}.wav",
                    label="Not USV",
                    spectrogram_path=spec_path,
                ))
                pos += stride_cols
                win_idx += 1

        logger.info(
            "Noise recording negatives: %d samples from %d recordings",
            len(samples), len(noise_recordings),
        )
        return samples

    # ── Spectrogram Extraction (legacy pipeline) ─────────────────────

    def _extract_spectrograms(self, candidates: list[Candidate]) -> None:
        """Extract spectrogram images for all candidates."""
        cfg = self.config
        extraction_config = ExtractionConfig(default_render_mode="training")
        extractor = SpectrogramExtractor(extraction_config)
        output_dir = cfg.output_dir / "spectrograms"

        for i, candidate in enumerate(candidates):
            result = extractor.extract_single(candidate, cfg.wav_dir, output_dir)
            if result is not None:
                candidate.spectrogram_path = result
            else:
                logger.warning("Failed to extract spectrogram for %s", candidate.candidate_id)

            if (i + 1) % 50 == 0:
                logger.info("Extracted %d / %d spectrograms", i + 1, len(candidates))

        logger.info("Spectrogram extraction complete: %d candidates", len(candidates))

    # ── Split Creation ────────────────────────────────────────────────

    def _candidates_to_samples(
        self, candidates: list[Candidate], label: str
    ) -> list[Sample]:
        """Convert candidates to Sample objects for splitting."""
        samples: list[Sample] = []
        for c in candidates:
            spec_path = str(c.spectrogram_path) if c.spectrogram_path else ""
            samples.append(Sample(
                candidate_id=c.candidate_id,
                source_file=c.source_file.name,
                label=label,
                spectrogram_path=spec_path,
            ))
        return samples

    def _create_splits(self, samples: list[Sample]) -> dict[str, list[Sample]]:
        """Split samples by recording into train/val/test."""
        cfg = self.config

        by_recording = group_samples_by_recording(samples)
        all_recordings = list(by_recording.keys())

        split_config = SplitConfig(
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
            random_seed=cfg.seed,
            stratify_by_population=False,  # No metadata available from labels
        )

        train_recs, val_recs, test_recs = split_recordings(
            all_recordings, split_config
        )

        splits: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
        for rec in train_recs:
            splits["train"].extend(by_recording[rec])
        for rec in val_recs:
            splits["val"].extend(by_recording[rec])
        for rec in test_recs:
            splits["test"].extend(by_recording[rec])

        return splits

    # ── Validation ────────────────────────────────────────────────────

    def _validate(
        self, splits: dict[str, list[Sample]], dry_run: bool = False
    ) -> list[str]:
        """Run quality checks and return warning strings."""
        cfg = self.config
        warnings: list[str] = []

        checks = [
            check_no_recording_leakage(splits),
            check_class_balance(splits),
            check_no_duplicate_ids(splits),
            check_split_sizes(
                splits, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio
            ),
        ]

        if not dry_run:
            checks.append(check_spectrogram_files_exist(splits))

        for check in checks:
            if not check.passed:
                msg = f"[FAIL] {check.name}: {check.message}"
                if check.details:
                    msg += " — " + "; ".join(check.details[:3])
                warnings.append(msg)

        return warnings

    # ── Output Writing ────────────────────────────────────────────────

    def _write_csvs(self, splits: dict[str, list[Sample]]) -> None:
        """Write train.csv, val.csv, test.csv."""
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, samples in splits.items():
            csv_path = cfg.output_dir / f"{split_name}.csv"
            _write_split_csv(samples, csv_path)
            logger.info("Wrote %s: %d samples", csv_path.name, len(samples))

    def _write_report(self, report: AssemblyReport, config: AssemblyConfig) -> None:
        """Write assembly_report.json with run metadata."""
        report_data = {
            "total_positives": report.total_positives,
            "total_negatives": report.total_negatives,
            "train_count": report.train_count,
            "val_count": report.val_count,
            "test_count": report.test_count,
            "n_recordings": report.n_recordings,
            "warnings": report.warnings,
            "config": {
                "labels_dir": str(config.labels_dir) if config.labels_dir else None,
                "wav_dir": str(config.wav_dir),
                "use_global_mad": config.use_global_mad,
                "unified_labels_path": str(config.unified_labels_path) if config.unified_labels_path else None,
                "window_columns": config.window_columns,
                "image_height_px": config.image_height_px,
                "colormap": config.colormap,
                "mad_vmin_scale": config.mad_vmin_scale,
                "mad_vmax_scale": config.mad_vmax_scale,
                "rolling_stride_ms": config.rolling_stride_ms,
                "noise_wav_dir": str(config.noise_wav_dir) if config.noise_wav_dir else None,
                "noise_stride_ms": config.noise_stride_ms,
                "jitter_n_samples": config.jitter_n_samples,
                "jitter_window_ms": config.jitter_window_ms,
                "jitter_context_padding_ms": config.jitter_context_padding_ms,
                "jitter_min_overlap": config.jitter_min_overlap,
                "neg_random_frac": config.neg_random_frac,
                "neg_inter_usv_frac": config.neg_inter_usv_frac,
                "neg_low_energy_frac": config.neg_low_energy_frac,
                "neg_ratio": config.neg_ratio,
                "train_ratio": config.train_ratio,
                "val_ratio": config.val_ratio,
                "test_ratio": config.test_ratio,
                "seed": config.seed,
                "output_dir": str(config.output_dir),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        report_path = config.output_dir / "assembly_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        logger.info("Wrote report: %s", report_path)
