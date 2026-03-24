"""Quality checks for the prepared dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .metadata import RecordingMetadata, load_metadata
from .splits import Sample, load_splits


@dataclass
class CheckResult:
    """Result of a single quality check."""

    name: str
    passed: bool
    message: str
    details: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Complete quality check report."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def summary(self) -> str:
        lines = [
            f"Quality Check Report: {self.n_passed}/{len(self.checks)} passed",
            "=" * 50,
        ]
        for check in self.checks:
            status = "PASS" if check.passed else "FAIL"
            lines.append(f"[{status}] {check.name}: {check.message}")
            if not check.passed and check.details:
                for detail in check.details[:5]:  # Limit details shown
                    lines.append(f"       - {detail}")
                if len(check.details) > 5:
                    lines.append(f"       ... and {len(check.details) - 5} more")
        return "\n".join(lines)


def check_no_recording_leakage(
    splits: dict[str, list[Sample]],
) -> CheckResult:
    """Check that no recording appears in multiple splits."""
    recordings_by_split: dict[str, set[str]] = {}
    for split_name, samples in splits.items():
        recordings_by_split[split_name] = {s.source_file for s in samples}

    # Find any overlap
    overlap_issues = []
    split_names = list(recordings_by_split.keys())

    for i, split1 in enumerate(split_names):
        for split2 in split_names[i + 1:]:
            overlap = recordings_by_split[split1] & recordings_by_split[split2]
            if overlap:
                for rec in overlap:
                    overlap_issues.append(f"{rec} in both {split1} and {split2}")

    if overlap_issues:
        return CheckResult(
            name="No Recording Leakage",
            passed=False,
            message=f"Found {len(overlap_issues)} recordings in multiple splits",
            details=overlap_issues,
        )

    return CheckResult(
        name="No Recording Leakage",
        passed=True,
        message="No recordings appear in multiple splits",
    )


def check_spectrogram_files_exist(
    splits: dict[str, list[Sample]],
) -> CheckResult:
    """Check that all spectrogram files exist."""
    missing = []

    for split_name, samples in splits.items():
        for sample in samples:
            spec_path = Path(sample.spectrogram_path)
            if not spec_path.exists():
                missing.append(f"{split_name}: {sample.candidate_id} -> {spec_path}")

    if missing:
        return CheckResult(
            name="Spectrogram Files Exist",
            passed=False,
            message=f"{len(missing)} spectrogram files missing",
            details=missing,
        )

    total_samples = sum(len(s) for s in splits.values())
    return CheckResult(
        name="Spectrogram Files Exist",
        passed=True,
        message=f"All {total_samples} spectrogram files exist",
    )


def check_class_balance(
    splits: dict[str, list[Sample]],
    max_ratio: float = 3.0,
) -> CheckResult:
    """Check class balance per split (warn if > max_ratio)."""
    imbalanced = []

    for split_name, samples in splits.items():
        if not samples:
            continue

        n_usv = sum(1 for s in samples if s.label == "USV")
        n_not_usv = sum(1 for s in samples if s.label == "Not USV")

        if n_usv == 0 or n_not_usv == 0:
            imbalanced.append(f"{split_name}: USV={n_usv}, Not USV={n_not_usv} (missing class)")
            continue

        ratio = max(n_usv / n_not_usv, n_not_usv / n_usv)
        if ratio > max_ratio:
            imbalanced.append(
                f"{split_name}: USV={n_usv}, Not USV={n_not_usv} (ratio={ratio:.2f})"
            )

    if imbalanced:
        return CheckResult(
            name="Class Balance",
            passed=False,
            message=f"Imbalanced classes in {len(imbalanced)} split(s)",
            details=imbalanced,
        )

    # Build summary
    details = []
    for split_name, samples in splits.items():
        n_usv = sum(1 for s in samples if s.label == "USV")
        n_not_usv = sum(1 for s in samples if s.label == "Not USV")
        total = len(samples)
        if total > 0:
            usv_pct = n_usv / total * 100
            details.append(f"{split_name}: {n_usv} USV ({usv_pct:.0f}%), {n_not_usv} Not USV")

    return CheckResult(
        name="Class Balance",
        passed=True,
        message="Class balance acceptable in all splits",
        details=details,
    )


def check_no_duplicate_ids(
    splits: dict[str, list[Sample]],
) -> CheckResult:
    """Check for duplicate candidate_ids across all samples."""
    all_ids = []
    for samples in splits.values():
        all_ids.extend(s.candidate_id for s in samples)

    seen = set()
    duplicates = []
    for cid in all_ids:
        if cid in seen:
            duplicates.append(cid)
        seen.add(cid)

    if duplicates:
        return CheckResult(
            name="No Duplicate IDs",
            passed=False,
            message=f"{len(duplicates)} duplicate candidate_ids found",
            details=list(set(duplicates)),
        )

    return CheckResult(
        name="No Duplicate IDs",
        passed=True,
        message=f"All {len(all_ids)} candidate_ids are unique",
    )


def check_population_coverage(
    splits: dict[str, list[Sample]],
    metadata: Optional[dict[str, RecordingMetadata]] = None,
) -> CheckResult:
    """Check that test set includes samples from all populations."""
    if not metadata:
        return CheckResult(
            name="Population Coverage",
            passed=True,
            message="No metadata provided (skipped)",
        )

    # Get populations in each split
    pops_by_split: dict[str, set[str]] = {}
    for split_name, samples in splits.items():
        pops = set()
        for sample in samples:
            meta = metadata.get(sample.source_file)
            if meta:
                pops.add(meta.population)
        pops_by_split[split_name] = pops

    # Check if all populations are in train set are also in test
    all_pops = set()
    for pops in pops_by_split.values():
        all_pops.update(pops)

    # Skip if only unknown population
    if all_pops == {"unknown"}:
        return CheckResult(
            name="Population Coverage",
            passed=True,
            message="All recordings have unknown population (stratification not applicable)",
        )

    test_pops = pops_by_split.get("test", set())
    missing_in_test = all_pops - test_pops - {"unknown"}

    if missing_in_test:
        return CheckResult(
            name="Population Coverage",
            passed=False,
            message=f"Test set missing populations: {missing_in_test}",
            details=[f"Train has: {pops_by_split.get('train', set())}",
                     f"Test has: {test_pops}"],
        )

    return CheckResult(
        name="Population Coverage",
        passed=True,
        message=f"Test set covers all populations: {test_pops}",
    )


def check_split_sizes(
    splits: dict[str, list[Sample]],
    expected_train_ratio: float = 0.70,
    expected_val_ratio: float = 0.15,
    expected_test_ratio: float = 0.15,
    tolerance: float = 0.10,
) -> CheckResult:
    """Check that split sizes are close to expected ratios."""
    total = sum(len(s) for s in splits.values())
    if total == 0:
        return CheckResult(
            name="Split Sizes",
            passed=False,
            message="No samples in any split",
        )

    actual_ratios = {
        name: len(samples) / total
        for name, samples in splits.items()
    }

    expected = {
        "train": expected_train_ratio,
        "val": expected_val_ratio,
        "test": expected_test_ratio,
    }

    issues = []
    for split_name, expected_ratio in expected.items():
        actual = actual_ratios.get(split_name, 0)
        if abs(actual - expected_ratio) > tolerance:
            issues.append(
                f"{split_name}: expected {expected_ratio:.0%}, got {actual:.1%}"
            )

    if issues:
        return CheckResult(
            name="Split Sizes",
            passed=False,
            message="Split sizes deviate from expected ratios",
            details=issues,
        )

    details = [
        f"{name}: {len(samples)} samples ({len(samples)/total:.1%})"
        for name, samples in splits.items()
    ]
    return CheckResult(
        name="Split Sizes",
        passed=True,
        message=f"Split sizes within tolerance (total={total})",
        details=details,
    )


def run_quality_checks(
    splits_dir: Path,
    metadata_csv: Optional[Path] = None,
) -> QualityReport:
    """Run all quality checks on the prepared dataset.

    Parameters
    ----------
    splits_dir : Path
        Directory containing train.csv, val.csv, test.csv.
    metadata_csv : Path, optional
        Path to recordings_metadata.csv.

    Returns
    -------
    QualityReport with all check results.
    """
    splits_dir = Path(splits_dir)
    report = QualityReport()

    # Load splits
    splits = load_splits(splits_dir)

    # Check if splits exist
    if not any(splits.values()):
        report.checks.append(CheckResult(
            name="Splits Loaded",
            passed=False,
            message="No splits found in " + str(splits_dir),
        ))
        return report

    report.checks.append(CheckResult(
        name="Splits Loaded",
        passed=True,
        message=f"Loaded {sum(len(s) for s in splits.values())} total samples",
    ))

    # Load metadata if available
    metadata = None
    if metadata_csv and Path(metadata_csv).exists():
        metadata = load_metadata(metadata_csv)

    # Run all checks
    report.checks.append(check_no_recording_leakage(splits))
    report.checks.append(check_spectrogram_files_exist(splits))
    report.checks.append(check_class_balance(splits))
    report.checks.append(check_no_duplicate_ids(splits))
    report.checks.append(check_population_coverage(splits, metadata))
    report.checks.append(check_split_sizes(splits))

    return report
