"""Tests for training cycle report generation and CLI argument parsing.

Tests are lightweight — no I/O-heavy operations, no mocking. Report tests
create CycleMetrics with hardcoded values and verify markdown output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from usv_spectrogram.training.cycle_report import CycleMetrics, generate_cycle_report


# ── CycleMetrics dataclass ────────────────────────────────────────────


class TestCycleMetrics:
    """Test CycleMetrics dataclass behavior."""

    def test_defaults(self):
        m = CycleMetrics()
        assert m.cycle_name == ""
        assert m.label_count == 0
        assert m.f1 == 0.0
        assert m.optimal_threshold == 0.5
        assert m.mining_skipped is False
        assert m.previous_f1 is None
        assert m.failed_step is None
        assert m.completed_steps == []

    def test_incremental_population(self):
        """Verify fields can be set incrementally (non-frozen)."""
        m = CycleMetrics()
        m.cycle_name = "test_cycle"
        m.label_count = 100
        m.train_samples = 600
        m.f1 = 0.85
        assert m.cycle_name == "test_cycle"
        assert m.label_count == 100
        assert m.f1 == 0.85

    def test_to_dict(self):
        m = CycleMetrics(cycle_name="test", label_count=50, f1=0.9)
        d = m.to_dict()
        assert isinstance(d, dict)
        assert d["cycle_name"] == "test"
        assert d["label_count"] == 50
        assert d["f1"] == 0.9
        # All fields should be present
        assert "optimal_threshold" in d
        assert "previous_f1" in d
        assert "completed_steps" in d

    def test_to_json_roundtrip(self, tmp_path):
        m = CycleMetrics(
            cycle_name="roundtrip",
            label_count=200,
            f1=0.88,
            precision=0.90,
            recall=0.86,
            optimal_threshold=0.35,
            model_size="medium",
            completed_steps=["ASSEMBLE", "TRAIN"],
        )
        json_path = tmp_path / "metrics.json"
        m.to_json(json_path)

        loaded = CycleMetrics.from_json(json_path)
        assert loaded.cycle_name == "roundtrip"
        assert loaded.label_count == 200
        assert loaded.f1 == 0.88
        assert loaded.optimal_threshold == 0.35
        assert loaded.model_size == "medium"
        assert loaded.completed_steps == ["ASSEMBLE", "TRAIN"]

    def test_from_json_ignores_unknown_fields(self, tmp_path):
        """Future-proofing: extra keys in JSON should not crash loading."""
        json_path = tmp_path / "metrics.json"
        data = CycleMetrics(cycle_name="test").to_dict()
        data["future_field"] = "should_be_ignored"
        with open(json_path, "w") as f:
            json.dump(data, f)

        loaded = CycleMetrics.from_json(json_path)
        assert loaded.cycle_name == "test"

    def test_completed_steps_not_shared(self):
        """Each instance gets its own completed_steps list."""
        m1 = CycleMetrics()
        m2 = CycleMetrics()
        m1.completed_steps.append("ASSEMBLE")
        assert m2.completed_steps == []


# ── Report generation ─────────────────────────────────────────────────


def _make_full_metrics() -> CycleMetrics:
    """Create a fully-populated CycleMetrics for testing."""
    return CycleMetrics(
        cycle_name="milestone_1",
        timestamp="2026-02-21T12:00:00+00:00",
        label_count=500,
        train_samples=3000,
        val_samples=600,
        test_samples=600,
        model_size="small",
        epochs_trained=35,
        best_val_loss=0.1234,
        training_time_s=180.0,
        precision=0.92,
        recall=0.88,
        f1=0.90,
        optimal_threshold=0.35,
        optimal_f1=0.91,
        high_recall_threshold=0.15,
        hard_negatives_found=42,
        mining_skipped=False,
        completed_steps=[
            "ASSEMBLE", "TRAIN", "EVALUATE", "OPTIMIZE", "MINE", "REPORT"
        ],
    )


class TestCycleReport:
    """Test report generation with known metrics."""

    def test_report_contains_all_sections(self, tmp_path):
        metrics = _make_full_metrics()
        report_path = tmp_path / "cycle_report.md"
        content = generate_cycle_report(metrics, report_path)

        assert report_path.exists()
        assert "# Training Cycle Report: milestone_1" in content
        assert "## Data Summary" in content
        assert "## Training Summary" in content
        assert "## Evaluation Results" in content
        assert "## Threshold Optimization" in content
        assert "## Hard Negative Mining" in content
        assert "## Output Artifacts" in content

    def test_report_data_values(self, tmp_path):
        metrics = _make_full_metrics()
        content = generate_cycle_report(metrics, tmp_path / "r.md")

        assert "500" in content  # label_count
        assert "3000" in content  # train_samples
        assert "4200" in content  # total (3000+600+600)
        assert "0.92" in content or "0.9200" in content  # precision

    def test_report_training_values(self, tmp_path):
        metrics = _make_full_metrics()
        content = generate_cycle_report(metrics, tmp_path / "r.md")

        assert "35" in content  # epochs
        assert "0.1234" in content  # best val loss
        assert "3.0 min" in content  # 180s -> 3.0 min

    def test_report_threshold_values(self, tmp_path):
        metrics = _make_full_metrics()
        content = generate_cycle_report(metrics, tmp_path / "r.md")

        assert "0.35" in content  # optimal threshold
        assert "0.15" in content  # high-recall threshold
        assert "0.9100" in content  # optimal F1

    def test_report_mining_count(self, tmp_path):
        metrics = _make_full_metrics()
        content = generate_cycle_report(metrics, tmp_path / "r.md")

        assert "42" in content  # hard negatives found

    def test_report_no_comparison_section_by_default(self, tmp_path):
        metrics = _make_full_metrics()
        content = generate_cycle_report(metrics, tmp_path / "r.md")

        # No previous_f1 set, so no comparison section
        assert "## Model Comparison" not in content


class TestCycleReportComparison:
    """Test report when comparison data is present."""

    def test_comparison_section_appears(self, tmp_path):
        metrics = _make_full_metrics()
        metrics.previous_f1 = 0.82
        metrics.f1_delta = 0.08

        content = generate_cycle_report(metrics, tmp_path / "r.md")

        assert "## Model Comparison" in content
        assert "0.8200" in content  # previous F1
        assert "+0.0800" in content  # delta

    def test_negative_delta(self, tmp_path):
        metrics = _make_full_metrics()
        metrics.previous_f1 = 0.95
        metrics.f1_delta = -0.05

        content = generate_cycle_report(metrics, tmp_path / "r.md")

        assert "-0.0500" in content


class TestCycleReportMiningSkipped:
    """Test report when mining was skipped."""

    def test_mining_skipped_message(self, tmp_path):
        metrics = _make_full_metrics()
        metrics.mining_skipped = True
        metrics.completed_steps = [
            s for s in metrics.completed_steps if s != "MINE"
        ]

        content = generate_cycle_report(metrics, tmp_path / "r.md")

        assert "Mining was **skipped**" in content

    def test_no_hard_negatives_dir_in_artifacts(self, tmp_path):
        metrics = _make_full_metrics()
        metrics.mining_skipped = True

        content = generate_cycle_report(metrics, tmp_path / "r.md")

        assert "hard_negatives/" not in content


class TestCycleReportFailure:
    """Test report when pipeline failed partway."""

    def test_failure_banner(self, tmp_path):
        metrics = CycleMetrics(
            cycle_name="failed_run",
            failed_step="TRAIN",
            completed_steps=["ASSEMBLE"],
            label_count=100,
            train_samples=500,
            val_samples=100,
            test_samples=100,
        )
        content = generate_cycle_report(metrics, tmp_path / "r.md")

        assert "WARNING" in content
        assert "TRAIN" in content
        assert "ASSEMBLE" in content


# ── Model size configs ────────────────────────────────────────────────


class TestModelSizeConfigs:
    """Test that MODEL_CONFIGS create valid CNN instances."""

    def test_all_sizes_instantiate(self):
        from run_training_cycle import MODEL_CONFIGS, create_model

        for size_name in ["small", "medium", "large"]:
            model = create_model(size_name)
            config = MODEL_CONFIGS[size_name]
            assert model.num_filters == config["filters"]
            assert model.dense_units == config["dense_units"]

    def test_small_param_count(self):
        from run_training_cycle import create_model

        model = create_model("small")
        n_params = sum(p.numel() for p in model.parameters())
        # small ~101K params
        assert 50_000 < n_params < 200_000, f"Small model has {n_params} params"

    def test_medium_param_count(self):
        from run_training_cycle import create_model

        model = create_model("medium")
        n_params = sum(p.numel() for p in model.parameters())
        # medium ~400K params
        assert 200_000 < n_params < 800_000, f"Medium model has {n_params} params"

    def test_large_param_count(self):
        from run_training_cycle import create_model

        model = create_model("large")
        n_params = sum(p.numel() for p in model.parameters())
        # large ~1.6M params
        assert 800_000 < n_params < 3_000_000, f"Large model has {n_params} params"

    def test_model_sizes_are_ordered(self):
        from run_training_cycle import create_model

        sizes = {}
        for name in ["small", "medium", "large"]:
            m = create_model(name)
            sizes[name] = sum(p.numel() for p in m.parameters())

        assert sizes["small"] < sizes["medium"] < sizes["large"]


# ── CLI argument parsing ──────────────────────────────────────────────


class TestParseArgs:
    """Test CLI argument parsing."""

    def test_required_args(self):
        from run_training_cycle import parse_args

        args = parse_args([
            "--labels-dir", "data/labels",
            "--wav-dir", "data/wav",
            "--cycle-name", "test_cycle",
            "--output-dir", "runs/test",
        ])
        assert args.labels_dir == Path("data/labels")
        assert args.wav_dir == Path("data/wav")
        assert args.cycle_name == "test_cycle"
        assert args.output_dir == Path("runs/test")

    def test_defaults(self):
        from run_training_cycle import parse_args

        args = parse_args([
            "--labels-dir", "x",
            "--wav-dir", "y",
            "--cycle-name", "z",
            "--output-dir", "w",
        ])
        assert args.model_size == "small"
        assert args.epochs == 50
        assert args.patience == 10
        assert args.seed == 42
        assert args.skip_mining is False
        assert args.previous_model is None
        assert args.target_recall == 0.90
        assert args.device == "auto"

    def test_all_optional_args(self):
        from run_training_cycle import parse_args

        args = parse_args([
            "--labels-dir", "data/labels",
            "--wav-dir", "data/wav",
            "--cycle-name", "test",
            "--output-dir", "runs/test",
            "--model-size", "large",
            "--epochs", "100",
            "--patience", "20",
            "--seed", "123",
            "--skip-mining",
            "--previous-model", "prev.pt",
            "--previous-model-size", "medium",
            "--target-recall", "0.95",
            "--device", "cpu",
        ])
        assert args.model_size == "large"
        assert args.epochs == 100
        assert args.patience == 20
        assert args.seed == 123
        assert args.skip_mining is True
        assert args.previous_model == Path("prev.pt")
        assert args.previous_model_size == "medium"
        assert args.target_recall == 0.95
        assert args.device == "cpu"

    def test_invalid_model_size(self):
        from run_training_cycle import parse_args

        with pytest.raises(SystemExit):
            parse_args([
                "--labels-dir", "x",
                "--wav-dir", "y",
                "--cycle-name", "z",
                "--output-dir", "w",
                "--model-size", "xlarge",
            ])

    def test_missing_required(self):
        from run_training_cycle import parse_args

        with pytest.raises(SystemExit):
            parse_args([])


# ── Integration test: step ordering ───────────────────────────────────


class TestMainStepOrder:
    """Test that main() executes steps in correct order."""

    def test_step_order_and_completed_steps(self, tmp_path):
        from unittest.mock import patch

        call_order = []

        def mock_assemble(args, metrics):
            call_order.append("ASSEMBLE")
            metrics.label_count = 50
            metrics.train_samples = 300
            metrics.val_samples = 50
            metrics.test_samples = 50
            return tmp_path / "data"

        def mock_train(args, metrics, data_dir, device):
            call_order.append("TRAIN")
            metrics.epochs_trained = 5

        def mock_evaluate(args, metrics, data_dir, device):
            call_order.append("EVALUATE")
            metrics.f1 = 0.85

        def mock_optimize(args, metrics, data_dir, device):
            call_order.append("OPTIMIZE")
            metrics.optimal_threshold = 0.3

        def mock_mine(args, metrics):
            call_order.append("MINE")
            metrics.mining_skipped = True

        def mock_compare(args, metrics, data_dir, device):
            call_order.append("COMPARE")

        def mock_report(args, metrics):
            call_order.append("REPORT")
            # Write a minimal file so the step doesn't fail
            report_path = args.output_dir / "cycle_report.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("# Report\n")

        with (
            patch("run_training_cycle.step_assemble", mock_assemble),
            patch("run_training_cycle.step_train", mock_train),
            patch("run_training_cycle.step_evaluate", mock_evaluate),
            patch("run_training_cycle.step_optimize_threshold", mock_optimize),
            patch("run_training_cycle.step_mine_hard_negatives", mock_mine),
            patch("run_training_cycle.step_compare", mock_compare),
            patch("run_training_cycle.step_report", mock_report),
        ):
            from run_training_cycle import main

            result = main([
                "--labels-dir", str(tmp_path),
                "--wav-dir", str(tmp_path),
                "--cycle-name", "test",
                "--output-dir", str(tmp_path / "out"),
                "--skip-mining",
            ])

        assert result == 0
        assert call_order == [
            "ASSEMBLE", "TRAIN", "EVALUATE", "OPTIMIZE",
            "MINE", "COMPARE", "REPORT",
        ]

    def test_failure_skips_to_report(self, tmp_path):
        from unittest.mock import patch

        call_order = []

        def mock_assemble(args, metrics):
            call_order.append("ASSEMBLE")
            metrics.train_samples = 100
            metrics.val_samples = 20
            metrics.test_samples = 20
            return tmp_path / "data"

        def mock_train(args, metrics, data_dir, device):
            call_order.append("TRAIN")
            raise RuntimeError("Simulated training failure")

        def mock_report(args, metrics):
            call_order.append("REPORT")
            report_path = args.output_dir / "cycle_report.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("# Partial Report\n")

        with (
            patch("run_training_cycle.step_assemble", mock_assemble),
            patch("run_training_cycle.step_train", mock_train),
            patch("run_training_cycle.step_report", mock_report),
        ):
            from run_training_cycle import main

            result = main([
                "--labels-dir", str(tmp_path),
                "--wav-dir", str(tmp_path),
                "--cycle-name", "fail_test",
                "--output-dir", str(tmp_path / "out"),
            ])

        assert result == 1
        # ASSEMBLE ran, TRAIN failed, then REPORT ran for partial results
        assert "ASSEMBLE" in call_order
        assert "TRAIN" in call_order
        assert "REPORT" in call_order
        # Steps after TRAIN should NOT have been called
        assert "EVALUATE" not in call_order


# ── Report content fixes ─────────────────────────────────────────────


class TestReportContentFixes:
    """Test fixes from review: comparison caveat, label approx, next steps."""

    def test_comparison_caveat_note(self, tmp_path):
        metrics = _make_full_metrics()
        metrics.previous_f1 = 0.82
        metrics.f1_delta = 0.08

        content = generate_cycle_report(metrics, tmp_path / "r.md")

        assert "Previous model evaluated on current cycle" in content

    def test_label_count_shows_approx(self, tmp_path):
        metrics = _make_full_metrics()
        content = generate_cycle_report(metrics, tmp_path / "r.md")

        assert "Unique labels (approx)" in content
        assert "~500" in content

    def test_next_steps_with_hard_negatives(self, tmp_path):
        metrics = _make_full_metrics()
        content = generate_cycle_report(metrics, tmp_path / "r.md")

        assert "## Recommended Next Steps" in content
        assert "hard negative candidates" in content

    def test_next_steps_low_f1(self, tmp_path):
        metrics = _make_full_metrics()
        metrics.f1 = 0.70
        content = generate_cycle_report(metrics, tmp_path / "r.md")

        assert "F1 below 0.85" in content

    def test_no_next_steps_when_nothing_actionable(self, tmp_path):
        metrics = CycleMetrics(
            cycle_name="clean",
            f1=0.95,
            mining_skipped=True,
            completed_steps=["ASSEMBLE", "TRAIN", "EVALUATE"],
        )
        content = generate_cycle_report(metrics, tmp_path / "r.md")

        assert "## Recommended Next Steps" not in content
