"""Tests for the Module 18.2a real-data loader in ``scripts/cnn_cleaning_validation.py``.

The loader functions are private helpers (``_load_real_cohorts``,
``_png_to_luminance``, ``_wav_to_spectrograms``, ``_resize_2d``) that
unlock Module 18.1's deferred real-data exit criteria. These tests cover
shape correctness, dtype, [0,1] range for PNG, finiteness for WAVs, and
the empty-cohort error path — without modifying the existing
``test_cleaning_pipeline.py`` / ``test_diagnostics.py`` spec tests.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_SCRIPT_PATH = REPO_ROOT / "scripts" / "cnn_cleaning_validation.py"
_spec = importlib.util.spec_from_file_location("cv_real", _SCRIPT_PATH)
assert _spec and _spec.loader
cv = importlib.util.module_from_spec(_spec)
sys.modules["cv_real"] = cv
_spec.loader.exec_module(cv)

# Canonical 300 kHz source rate — imported from corpus, not redeclared.
from usv_spectrogram.corpus import SAMPLE_RATE_HZ  # noqa: E402


# ---------------------------------------------------------------------------
# _resize_2d
# ---------------------------------------------------------------------------


def test_resize_2d_identity_when_shape_matches():
    """If input already matches target_shape, _resize_2d returns same shape."""
    a = np.random.RandomState(0).randn(8, 8).astype(np.float32)
    out = cv._resize_2d(a, (8, 8))
    assert out.shape == (8, 8)
    assert out.dtype == np.float32


def test_resize_2d_downsample_shape():
    """Downsample 257×430 → 227×227 (the lab/wild STFT → target case)."""
    a = np.random.RandomState(1).randn(257, 430).astype(np.float32)
    out = cv._resize_2d(a, (227, 227))
    assert out.shape == (227, 227)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


def test_resize_2d_constant_array_stays_constant():
    """Resizing a constant array yields a constant array of the same value."""
    a = np.full((50, 50), 3.14, dtype=np.float32)
    out = cv._resize_2d(a, (20, 20))
    assert out.shape == (20, 20)
    assert np.allclose(out, 3.14, atol=1e-5)


# ---------------------------------------------------------------------------
# _png_to_luminance
# ---------------------------------------------------------------------------


def _write_rgb_png(path: Path, size: tuple[int, int] = (227, 227)) -> None:
    """Create a synthetic RGB PNG so we don't depend on the OSF download."""
    from PIL import Image
    rng = np.random.RandomState(42)
    arr = (rng.rand(size[1], size[0], 3) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def test_png_to_luminance_shape_dtype_range(tmp_path):
    """PNG → luminance returns (227, 227) float32 in [0, 1]."""
    p = tmp_path / "fake.png"
    _write_rgb_png(p)
    out = cv._png_to_luminance(p, (227, 227))
    assert out.shape == (227, 227)
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_png_to_luminance_handles_non_target_size(tmp_path):
    """PNGs not at 227×227 get resized into target shape."""
    p = tmp_path / "oddsize.png"
    _write_rgb_png(p, size=(100, 80))
    out = cv._png_to_luminance(p, (227, 227))
    assert out.shape == (227, 227)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# _wav_to_spectrograms
# ---------------------------------------------------------------------------


def _write_test_wav(path: Path, duration_s: float = 5.0,
                    sr: int = SAMPLE_RATE_HZ) -> None:
    """Write a short synthetic WAV at the given sr. Mono white noise +
    a faint 60 kHz tone so the STFT has structure, not just noise.
    """
    import soundfile as sf
    rng = np.random.RandomState(123)
    n = int(round(duration_s * sr))
    t = np.arange(n) / sr
    samples = (0.01 * rng.randn(n) + 0.05 * np.sin(2 * np.pi * 60_000 * t))
    sf.write(str(path), samples.astype(np.float32), sr)


def test_wav_to_spectrograms_shape_and_finiteness(tmp_path):
    """Loader returns (n, 227, 227) float32, no NaN / inf."""
    wav = tmp_path / "synthetic.wav"
    _write_test_wav(wav, duration_s=3.0, sr=SAMPLE_RATE_HZ)

    rng = np.random.default_rng(7)
    out = cv._wav_to_spectrograms(
        wav_paths=[wav],
        n_windows=4,
        target_sr=cv.TARGET_SAMPLE_RATE_HZ,
        target_shape=cv._REAL_TARGET_SHAPE,
        window_seconds=cv._REAL_WINDOW_DURATION_S,
        n_fft=cv.STFT_N_FFT,
        hop=cv.STFT_HOP,
        rng=rng,
    )
    assert out.shape == (4, *cv._REAL_TARGET_SHAPE)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


def test_wav_to_spectrograms_empty_path_list_raises(tmp_path):
    """Empty wav_paths is a programmer error → raise."""
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="no WAV files"):
        cv._wav_to_spectrograms(
            wav_paths=[],
            n_windows=4,
            target_sr=cv.TARGET_SAMPLE_RATE_HZ,
            target_shape=cv._REAL_TARGET_SHAPE,
            window_seconds=cv._REAL_WINDOW_DURATION_S,
            n_fft=cv.STFT_N_FFT,
            hop=cv.STFT_HOP,
            rng=rng,
        )


def test_wav_to_spectrograms_too_short_wav_raises(tmp_path):
    """WAVs shorter than window_seconds cannot yield windows → raises."""
    wav = tmp_path / "tiny.wav"
    # 50 ms is shorter than the 220 ms window; loader can't sample it.
    _write_test_wav(wav, duration_s=0.05, sr=SAMPLE_RATE_HZ)

    rng = np.random.default_rng(7)
    with pytest.raises(RuntimeError, match="Failed to produce"):
        cv._wav_to_spectrograms(
            wav_paths=[wav],
            n_windows=4,
            target_sr=cv.TARGET_SAMPLE_RATE_HZ,
            target_shape=cv._REAL_TARGET_SHAPE,
            window_seconds=cv._REAL_WINDOW_DURATION_S,
            n_fft=cv.STFT_N_FFT,
            hop=cv.STFT_HOP,
            rng=rng,
        )


# ---------------------------------------------------------------------------
# _load_real_cohorts (full integration, all 3 cohorts)
# ---------------------------------------------------------------------------


def _build_fake_vocalmat(root: Path, n_pngs: int = 20) -> Path:
    """Lay out a VocalMat-like directory with n_pngs PNGs across 2 classes."""
    for i in range(n_pngs):
        cls = "noise" if i % 2 == 0 else "step_up"
        (root / cls).mkdir(parents=True, exist_ok=True)
        _write_rgb_png(root / cls / f"png_{i:03d}.png")
    return root


def test_load_real_cohorts_returns_three_cohorts(tmp_path):
    """All three cohorts present, uniform shape, finite values."""
    vm_dir = _build_fake_vocalmat(tmp_path / "vm", n_pngs=10)
    lab_dir = tmp_path / "lab"
    wild_dir = tmp_path / "wild"
    lab_dir.mkdir()
    wild_dir.mkdir()
    _write_test_wav(lab_dir / "lab.wav", duration_s=2.0, sr=SAMPLE_RATE_HZ)
    _write_test_wav(wild_dir / "wild.wav", duration_s=2.0, sr=SAMPLE_RATE_HZ)

    out = cv._load_real_cohorts(
        vocalmat_dir=vm_dir,
        lab_wav_dir=lab_dir,
        wild_wav_dir=wild_dir,
        sample_size=4,
    )
    assert set(out.keys()) == {"vocalmat", "lab_131204", "wild_5970"}
    for cid, arr in out.items():
        assert arr.shape == (4, *cv._REAL_TARGET_SHAPE), \
            f"cohort {cid!r} shape mismatch: {arr.shape}"
        assert arr.dtype == np.float32
        assert np.all(np.isfinite(arr))


def test_load_real_cohorts_missing_vocalmat_raises(tmp_path):
    """An empty VocalMat dir raises before any WAV processing."""
    empty_vm = tmp_path / "empty_vm"
    empty_vm.mkdir()
    lab_dir = tmp_path / "lab"
    wild_dir = tmp_path / "wild"
    lab_dir.mkdir()
    wild_dir.mkdir()
    _write_test_wav(lab_dir / "lab.wav", duration_s=2.0, sr=SAMPLE_RATE_HZ)
    _write_test_wav(wild_dir / "wild.wav", duration_s=2.0, sr=SAMPLE_RATE_HZ)

    with pytest.raises(FileNotFoundError, match="no PNG files"):
        cv._load_real_cohorts(
            vocalmat_dir=empty_vm,
            lab_wav_dir=lab_dir,
            wild_wav_dir=wild_dir,
            sample_size=4,
        )


def test_load_real_cohorts_missing_lab_wavs_raises(tmp_path):
    """An empty lab WAV dir raises FileNotFoundError."""
    vm_dir = _build_fake_vocalmat(tmp_path / "vm", n_pngs=10)
    lab_dir = tmp_path / "lab"
    wild_dir = tmp_path / "wild"
    lab_dir.mkdir()  # empty
    wild_dir.mkdir()
    _write_test_wav(wild_dir / "wild.wav", duration_s=2.0, sr=SAMPLE_RATE_HZ)

    with pytest.raises(FileNotFoundError, match="No .wav"):
        cv._load_real_cohorts(
            vocalmat_dir=vm_dir,
            lab_wav_dir=lab_dir,
            wild_wav_dir=wild_dir,
            sample_size=4,
        )


def test_load_real_cohorts_deterministic_under_seed(tmp_path):
    """Same seed → identical VocalMat selection across runs."""
    vm_dir = _build_fake_vocalmat(tmp_path / "vm", n_pngs=20)
    lab_dir = tmp_path / "lab"
    wild_dir = tmp_path / "wild"
    lab_dir.mkdir()
    wild_dir.mkdir()
    _write_test_wav(lab_dir / "lab.wav", duration_s=2.0, sr=SAMPLE_RATE_HZ)
    _write_test_wav(wild_dir / "wild.wav", duration_s=2.0, sr=SAMPLE_RATE_HZ)

    a = cv._load_real_cohorts(
        vocalmat_dir=vm_dir, lab_wav_dir=lab_dir, wild_wav_dir=wild_dir,
        sample_size=4, seed=1729,
    )
    b = cv._load_real_cohorts(
        vocalmat_dir=vm_dir, lab_wav_dir=lab_dir, wild_wav_dir=wild_dir,
        sample_size=4, seed=1729,
    )
    # VocalMat is deterministic (rng.choice on sorted paths with same seed).
    np.testing.assert_array_equal(a["vocalmat"], b["vocalmat"])
