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

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal as scipy_signal

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

    labels_dir: Path  # Dir with LabelStorage JSON files
    wav_dir: Path  # WAV file directory

    # Jitter augmentation
    jitter_n_samples: int = 5  # Jittered versions per positive
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
        object.__setattr__(self, "labels_dir", Path(self.labels_dir))
        object.__setattr__(self, "wav_dir", Path(self.wav_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))


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
        report = AssemblyReport(output_dir=cfg.output_dir)

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

    # ── Spectrogram Extraction ────────────────────────────────────────

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
                "labels_dir": str(config.labels_dir),
                "wav_dir": str(config.wav_dir),
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
