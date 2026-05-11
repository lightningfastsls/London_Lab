"""Pre-implementation spec tests for the adaptive soft-notch filter.

These 15 tests are the contract for ``src/usv_spectrogram/app/core/notch.py``
and ``scripts/calibrate_lab_tonal_lines.py``. They are written BEFORE the
implementation. Until implementation lands, every test fails with
``NotImplementedError`` (or downstream propagation from a stubbed call).

Tests 1-7   discovery + filtering, pure auto-detect mode
Tests 8-13  library mode + reconciliation
Tests 14-15 calibration script end-to-end

Spec: ``docs/handoffs/2026-05-11_adaptive-soft-notch.md``

Do not modify test expectations to make implementation pass — they are SPEC.
If a test expectation looks wrong, flag it for discussion (CLAUDE.md
Test Protocol).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# ---------------------------------------------------------------------------
# Path bootstrap: src/ for the notch module, scripts/ for the calibration script
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for _p in (SRC_ROOT, SCRIPTS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from usv_spectrogram.app.core import notch  # noqa: E402
from usv_spectrogram.corpus import SAMPLE_RATE_HZ  # noqa: E402
import calibrate_lab_tonal_lines  # noqa: E402

REF_WAV = REPO_ROOT / "USV_lab_131204_chunked_2s_hot" / "131204_1400_m3fm3_chunk_243.wav"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _power_at_freq(audio: np.ndarray, target_hz: float, fs_hz: float) -> float:
    """Sinusoid-amplitude estimate at ``target_hz`` via Goertzel-like dot product.

    Used to assert pre/post filter attenuation at specific frequencies
    without depending on a particular PSD/STFT configuration.
    """
    n = len(audio)
    t = np.arange(n) / fs_hz
    ref = np.exp(2j * np.pi * target_hz * t)
    return float(np.abs(np.dot(audio, ref)) / n)


def _db(ratio: float) -> float:
    return 20.0 * np.log10(max(ratio, 1e-30))


def _fm_sweep(t: np.ndarray, f0_hz: float, f1_hz: float) -> np.ndarray:
    """Linear FM sweep covering ``[t[0], t[-1]]``."""
    duration = t[-1] - t[0]
    if duration <= 0:
        return np.zeros_like(t)
    phase = 2 * np.pi * (f0_hz * (t - t[0]) + (f1_hz - f0_hz) / (2 * duration) * (t - t[0]) ** 2)
    return np.sin(phase)


# ===========================================================================
# Tests 1-7  —  discovery + filtering, pure auto-detect mode
# ===========================================================================

def test_01_two_tone_synthetic_survives():
    """30 kHz tone survives, 50 kHz tone reduced by >=10 dB (the discovery threshold).

    The spec prose says "30 + 50 kHz sines at equal amplitude," but the
    assertion below requires the 30 kHz tone to NOT be flagged by
    ``discover_tonals``. Two equal-amplitude pure sines both produce PSD peaks
    ~100 dB above the local-median floor, so both would be detected and both
    filtered. Fixture corrected: 30 kHz at low SNR (~+6 dB elevation, below
    the 10 dB discovery threshold) on a noise floor; 50 kHz still prominent.
    Assertions unchanged — they embody the spec's intent (benign in-band signal
    survives, equipment line is cut).
    """
    fs = SAMPLE_RATE_HZ
    duration_s = 2.0
    rng = np.random.default_rng(1)
    t = np.arange(int(duration_s * fs)) / fs
    audio = (
        0.005 * np.sin(2 * np.pi * 30_000 * t)
        + np.sin(2 * np.pi * 50_000 * t)
        + 0.1 * rng.standard_normal(t.shape)
    ).astype(np.float64)

    cleaned, _ = notch.auto_soft_notch(audio, fs_hz=fs, library=None)

    p30_in = _power_at_freq(audio, 30_000, fs)
    p30_out = _power_at_freq(cleaned, 30_000, fs)
    p50_in = _power_at_freq(audio, 50_000, fs)
    p50_out = _power_at_freq(cleaned, 50_000, fs)

    assert abs(_db(p30_out / p30_in)) < 1.0, \
        f"30 kHz tone must survive within 1 dB; got {_db(p30_out/p30_in):+.2f} dB"
    assert _db(p50_out / p50_in) < -10.0, \
        f"50 kHz tone must be reduced by >=10 dB; got {_db(p50_out/p50_in):+.2f} dB"


def test_02_pure_noise_no_tonals_detected():
    """White noise: discover_tonals returns []; audio passes through unchanged."""
    fs = SAMPLE_RATE_HZ
    rng = np.random.default_rng(42)
    audio = rng.standard_normal(2 * fs).astype(np.float64)

    tonals = notch.discover_tonals(audio, fs_hz=fs)
    assert tonals == [], f"Expected no tonals in white noise; got {tonals}"

    cleaned, recon = notch.auto_soft_notch(audio, fs_hz=fs, library=None)
    assert recon.unmatched_detections == []
    np.testing.assert_array_equal(cleaned, audio)


def test_03_usv_burst_not_flagged_as_tonal():
    """A 50 ms FM sweep across 45-55 kHz is transient; not picked up as a tonal."""
    fs = SAMPLE_RATE_HZ
    duration_s = 2.0
    rng = np.random.default_rng(3)
    t = np.arange(int(duration_s * fs)) / fs
    # Background noise so the local-median baseline is realistic, not -inf.
    audio = (0.1 * rng.standard_normal(t.shape)).astype(np.float64)
    burst_start = int(0.5 * fs)
    burst_end = int(0.55 * fs)
    audio[burst_start:burst_end] += _fm_sweep(t[burst_start:burst_end], 45_000, 55_000)

    tonals = notch.discover_tonals(audio, fs_hz=fs)
    in_band = [tt for tt in tonals if 44_000 < tt.center_hz < 56_000]
    assert in_band == [], \
        f"Transient FM sweep at 45-55 kHz must not register as a tonal; got {in_band}"


def test_04_tonal_on_usv_preserves_usv():
    """USV sweep + 51 kHz continuous tone: tonal cut, sweep largely preserved."""
    fs = SAMPLE_RATE_HZ
    duration_s = 1.0
    t = np.arange(int(duration_s * fs)) / fs
    tone = np.sin(2 * np.pi * 51_000 * t)
    sweep = _fm_sweep(t, 40_000, 60_000)
    audio = (tone + sweep).astype(np.float64)

    cleaned, recon = notch.auto_soft_notch(audio, fs_hz=fs, library=None)

    all_detected = [m[1] for m in recon.matched] + list(recon.unmatched_detections)
    centers = sorted(round(d.center_hz) for d in all_detected)
    assert any(50_500 < c < 51_500 for c in centers), \
        f"Expected tonal near 51 kHz; got centers {centers}"

    # 51 kHz line attenuated
    assert _db(_power_at_freq(cleaned, 51_000, fs) / _power_at_freq(audio, 51_000, fs)) < -8.0

    # Sweep energy at 45 kHz (well outside the notch kill zone) survives within ~3 dB
    p45_ratio_db = _db(_power_at_freq(cleaned, 45_000, fs) / _power_at_freq(audio, 45_000, fs))
    assert abs(p45_ratio_db) < 3.0, \
        f"USV sweep energy at 45 kHz over-attenuated: {p45_ratio_db:+.2f} dB"


def test_05_multi_tonal_cascade():
    """Two tones at 46 and 51 kHz: both detected, both cut."""
    fs = SAMPLE_RATE_HZ
    duration_s = 2.0
    t = np.arange(int(duration_s * fs)) / fs
    rng = np.random.default_rng(5)
    audio = (
        np.sin(2 * np.pi * 46_000 * t)
        + np.sin(2 * np.pi * 51_000 * t)
        + 0.05 * rng.standard_normal(t.shape)
    ).astype(np.float64)

    cleaned, recon = notch.auto_soft_notch(audio, fs_hz=fs, library=None)
    all_detected = [m[1] for m in recon.matched] + list(recon.unmatched_detections)
    centers = sorted(round(d.center_hz) for d in all_detected)
    assert any(45_500 < c < 46_500 for c in centers), centers
    assert any(50_500 < c < 51_500 for c in centers), centers

    assert _db(_power_at_freq(cleaned, 46_000, fs) / _power_at_freq(audio, 46_000, fs)) < -8.0
    assert _db(_power_at_freq(cleaned, 51_000, fs) / _power_at_freq(audio, 51_000, fs)) < -8.0


@pytest.mark.skipif(
    not REF_WAV.exists(),
    reason=f"Reference WAV {REF_WAV} not present; required for regression test #6.",
)
def test_06_reference_wav_self_consistency():
    """Real lab WAV: tonal at 51.09 +/- 0.2 kHz, above_median_db in [13, 18]."""
    audio, fs = sf.read(str(REF_WAV), dtype="float64")
    if audio.ndim > 1:
        audio = audio[:, 0]
    tonals = notch.discover_tonals(audio, fs_hz=fs)
    near_51 = [tt for tt in tonals if abs(tt.center_hz - 51_090) < 200]
    assert near_51, (
        f"Expected tonal near 51.09 kHz in reference WAV; got centers "
        f"{[tt.center_hz for tt in tonals]}"
    )
    tt = near_51[0]
    assert 13.0 <= tt.above_median_db <= 18.0, (
        f"above_median_db {tt.above_median_db:.2f} dB outside expected [13, 18] window"
    )


def test_07_wild_data_default_off_is_identity():
    """The default-off path must not perturb audio.

    Two angles:
      (a) Direct: ``auto_soft_notch`` on a no-tonal signal returns audio bit-identical.
      (b) Indirect: ``AudioLoader(auto_soft_notch=False)`` (default) does not call notch.
          We assert the kwarg exists with default False — the actual end-to-end
          byte-equivalence is GATE C of the spec, not a unit test.
    """
    # (a) Direct invariant
    fs = SAMPLE_RATE_HZ
    rng = np.random.default_rng(7)
    audio = (0.05 * rng.standard_normal(2 * fs)).astype(np.float64)
    cleaned, recon = notch.auto_soft_notch(audio, fs_hz=fs, library=None)
    assert recon.matched == []
    assert recon.unmatched_detections == []
    np.testing.assert_array_equal(cleaned, audio)

    # (b) AudioLoader contract
    from usv_spectrogram.app.core.audio_loader import AudioLoader
    loader = AudioLoader()
    assert getattr(loader, "auto_soft_notch", None) is False, (
        "AudioLoader must expose auto_soft_notch with default False so wild-mouse "
        "runs are byte-identical."
    )


# ===========================================================================
# Tests 8-13  —  library mode + reconciliation
# ===========================================================================

def _build_library(*entries: tuple[float, float, float, float]) -> notch.TonalLibrary:
    """Helper: build a TonalLibrary from (center_hz, width_hz, mean_above, stdev) tuples."""
    return notch.TonalLibrary(
        rig_id="test_rig",
        calibrated_at="2026-05-11T00:00:00",
        n_chunks_sampled=10,
        sample_files=["synthetic"],
        entries=[
            notch.LibraryEntry(
                center_hz=c, width_hz=w,
                mean_above_median_db=m, stdev_above_median_db=s,
                n_chunks_seen=10, detection_rate=1.0,
            )
            for (c, w, m, s) in entries
        ],
    )


def test_08_library_hit_matched():
    """Library entry at 51 kHz; signal contains 30 + 51 kHz tones. 51 cut, 30 survives.

    Spec assertion is ``unmatched_detections == []`` — i.e., the ONLY detection
    is the library-matched 51 kHz tonal. Same fixture interpretation as test 01:
    30 kHz at low SNR so ``discover_tonals`` doesn't flag it. 51 kHz prominent
    so it triggers and matches the library entry.
    """
    fs = SAMPLE_RATE_HZ
    duration_s = 2.0
    rng = np.random.default_rng(8)
    t = np.arange(int(duration_s * fs)) / fs
    audio = (
        0.005 * np.sin(2 * np.pi * 30_000 * t)
        + np.sin(2 * np.pi * 51_000 * t)
        + 0.1 * rng.standard_normal(t.shape)
    ).astype(np.float64)
    library = _build_library((51_000.0, 400.0, 15.0, 1.0))

    cleaned, recon = notch.auto_soft_notch(audio, fs_hz=fs, library=library)

    assert len(recon.matched) == 1
    assert recon.unmatched_detections == []
    assert recon.unmatched_library_entries == []
    assert _db(_power_at_freq(cleaned, 51_000, fs) / _power_at_freq(audio, 51_000, fs)) < -8.0
    assert abs(_db(_power_at_freq(cleaned, 30_000, fs) / _power_at_freq(audio, 30_000, fs))) < 1.0


def test_09_library_miss_unmatched_detection_not_filtered():
    """Library has 51 kHz; signal has 51 + 73. 51 matched & filtered; 73 logged but NOT cut."""
    fs = SAMPLE_RATE_HZ
    duration_s = 2.0
    t = np.arange(int(duration_s * fs)) / fs
    audio = (np.sin(2 * np.pi * 51_000 * t) + np.sin(2 * np.pi * 73_000 * t)).astype(np.float64)
    library = _build_library((51_000.0, 400.0, 15.0, 1.0))

    cleaned, recon = notch.auto_soft_notch(audio, fs_hz=fs, library=library)

    assert len(recon.matched) == 1
    assert any(72_500 < d.center_hz < 73_500 for d in recon.unmatched_detections), (
        f"Expected unmatched detection near 73 kHz; got "
        f"{[d.center_hz for d in recon.unmatched_detections]}"
    )
    # 51 kHz (library) cut
    assert _db(_power_at_freq(cleaned, 51_000, fs) / _power_at_freq(audio, 51_000, fs)) < -8.0
    # 73 kHz (audit-only) NOT cut — library is source of truth
    p73_ratio = _db(_power_at_freq(cleaned, 73_000, fs) / _power_at_freq(audio, 73_000, fs))
    assert abs(p73_ratio) < 1.0, (
        f"Library audit must not auto-filter unmatched detections; got {p73_ratio:+.2f} dB at 73 kHz"
    )


def test_10_library_expects_signal_lacks():
    """Library has 51 + 46 kHz; signal has only 51 kHz. 46 in unmatched_library_entries."""
    fs = SAMPLE_RATE_HZ
    duration_s = 2.0
    t = np.arange(int(duration_s * fs)) / fs
    audio = np.sin(2 * np.pi * 51_000 * t).astype(np.float64)
    library = _build_library(
        (51_000.0, 400.0, 15.0, 1.0),
        (46_000.0, 400.0, 12.0, 1.0),
    )

    cleaned, recon = notch.auto_soft_notch(audio, fs_hz=fs, library=library)

    assert len(recon.matched) == 1
    assert len(recon.unmatched_library_entries) == 1
    assert recon.unmatched_library_entries[0].center_hz == pytest.approx(46_000.0)
    assert _db(_power_at_freq(cleaned, 51_000, fs) / _power_at_freq(audio, 51_000, fs)) < -8.0


def test_11_intensity_drift_triggers_warning():
    """Library mean=+15 dB stdev=1 dB; signal has +35 dB tonal. Matched + intensity_drifts non-empty."""
    fs = SAMPLE_RATE_HZ
    duration_s = 2.0
    t = np.arange(int(duration_s * fs)) / fs
    rng = np.random.default_rng(11)
    # Tone amplitude 10x larger than the noise -> sits ~+40 dB above median
    audio = (
        10.0 * np.sin(2 * np.pi * 51_000 * t)
        + 0.1 * rng.standard_normal(t.shape)
    ).astype(np.float64)
    library = _build_library((51_000.0, 400.0, 15.0, 1.0))

    _, recon = notch.auto_soft_notch(audio, fs_hz=fs, library=library)

    assert len(recon.matched) == 1
    assert len(recon.intensity_drifts) >= 1, (
        "Expected an intensity_drifts entry when measured above_median_db deviates >2 sigma"
    )
    _, sigma = recon.intensity_drifts[0]
    assert sigma > 2.0, f"Drift sigma should exceed 2.0; got {sigma:.2f}"


def test_12_tonal_library_roundtrip(tmp_path):
    """Build a TonalLibrary, save to JSON, reload — all fields equal."""
    library = notch.TonalLibrary(
        rig_id="lab_test",
        calibrated_at="2026-05-11T12:00:00",
        n_chunks_sampled=42,
        sample_files=["a.wav", "b.wav"],
        entries=[
            notch.LibraryEntry(51_090.0, 220.0, 15.9, 1.2, 42, 0.98),
            notch.LibraryEntry(46_580.0, 200.0, 11.8, 0.7, 40, 0.93),
        ],
    )
    path = tmp_path / "lib.json"
    library.save(path)
    loaded = notch.TonalLibrary.load(path)
    assert loaded == library


def test_13_tonal_library_schema_validation(tmp_path):
    """Malformed JSON and out-of-range field values raise clear errors."""
    # (a) Missing required 'entries' field
    bad1 = tmp_path / "missing_entries.json"
    bad1.write_text(json.dumps({"rig_id": "x", "calibrated_at": "2026-05-11T00:00:00"}))
    with pytest.raises((ValueError, KeyError, TypeError)):
        notch.TonalLibrary.load(bad1)

    # (b) detection_rate > 1.0 (invalid by definition)
    bad2 = tmp_path / "rate_out_of_range.json"
    bad2.write_text(json.dumps({
        "rig_id": "x",
        "calibrated_at": "2026-05-11T00:00:00",
        "n_chunks_sampled": 10,
        "sample_files": [],
        "entries": [{
            "center_hz": 51_000.0, "width_hz": 200.0,
            "mean_above_median_db": 15.0, "stdev_above_median_db": 1.0,
            "n_chunks_seen": 10, "detection_rate": 1.5,
        }],
    }))
    with pytest.raises((ValueError, AssertionError)):
        notch.TonalLibrary.load(bad2)


# ===========================================================================
# Tests 14-15  —  calibration script integration
# ===========================================================================

def _write_synthetic_chunks(
    wav_dir: Path,
    n_chunks: int,
    fs: int,
    duration_s: float,
    tone_freqs_amps,
    noise_amp: float,
    rng: np.random.Generator,
    sporadic_freq_in_chunks: tuple[float, list[int]] | None = None,
) -> None:
    """Helper for tests 14 and 15. Writes ``n_chunks`` WAVs.

    ``tone_freqs_amps``: iterable of (freq_hz, base_amplitude) — applied to every chunk
        with a small per-chunk amplitude jitter for realism.
    ``sporadic_freq_in_chunks``: if not None, (freq_hz, list_of_chunk_indices) — added
        only to the listed chunks (used in test 15 to verify rejection of rare tonals).
    """
    wav_dir.mkdir(parents=True, exist_ok=True)
    n_samples = int(duration_s * fs)
    t = np.arange(n_samples) / fs
    for i in range(n_chunks):
        audio = noise_amp * rng.standard_normal(n_samples)
        for freq, base_amp in tone_freqs_amps:
            jitter = 1.0 + 0.15 * rng.standard_normal()
            audio += base_amp * jitter * np.sin(2 * np.pi * freq * t)
        if sporadic_freq_in_chunks is not None:
            spor_freq, spor_chunks = sporadic_freq_in_chunks
            if i in spor_chunks:
                audio += np.sin(2 * np.pi * spor_freq * t)
        sf.write(str(wav_dir / f"chunk_{i:03d}.wav"), audio.astype(np.float32), fs)


def test_14_calibration_end_to_end(tmp_path):
    """20 synthetic 2 s chunks each carry a 51 kHz tone of varying intensity.

    Expected library: exactly one entry near 51 kHz with ``detection_rate ~= 1.0``.
    """
    fs = SAMPLE_RATE_HZ
    rng = np.random.default_rng(14)
    wav_dir = tmp_path / "synthetic_chunks"
    _write_synthetic_chunks(
        wav_dir, n_chunks=20, fs=fs, duration_s=2.0,
        tone_freqs_amps=[(51_000.0, 2.0)],
        noise_amp=0.1,
        rng=rng,
    )

    library = calibrate_lab_tonal_lines.calibrate(
        wav_dir=wav_dir,
        rig_id="test_rig",
        sample_size=20,
        min_detection_rate=0.5,
        random_seed=0,
    )

    near_51 = [e for e in library.entries if abs(e.center_hz - 51_000) < 200]
    assert len(near_51) == 1, (
        f"Expected exactly one library entry near 51 kHz; "
        f"got {[e.center_hz for e in library.entries]}"
    )
    assert near_51[0].detection_rate >= 0.95, (
        f"detection_rate {near_51[0].detection_rate:.2f} should be ~= 1.0 for a "
        "tonal present in every chunk"
    )


def test_15_calibration_rejects_sporadic_tonals(tmp_path):
    """Same as 14, plus a 73 kHz tone in only 2/20 chunks. 73 kHz must NOT be in library."""
    fs = SAMPLE_RATE_HZ
    rng = np.random.default_rng(15)
    wav_dir = tmp_path / "synthetic_sporadic"
    _write_synthetic_chunks(
        wav_dir, n_chunks=20, fs=fs, duration_s=2.0,
        tone_freqs_amps=[(51_000.0, 2.0)],
        noise_amp=0.1,
        rng=rng,
        sporadic_freq_in_chunks=(73_000.0, [3, 11]),
    )

    library = calibrate_lab_tonal_lines.calibrate(
        wav_dir=wav_dir,
        rig_id="test_rig",
        sample_size=20,
        min_detection_rate=0.5,
        random_seed=0,
    )

    centers = [e.center_hz for e in library.entries]
    assert any(50_500 < c < 51_500 for c in centers), centers
    assert not any(72_500 < c < 73_500 for c in centers), (
        f"Sporadic 73 kHz tonal (2/20 chunks = detection_rate 0.10) must NOT be "
        f"promoted to a library entry under min_detection_rate=0.5; got centers {centers}"
    )
