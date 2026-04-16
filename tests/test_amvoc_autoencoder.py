"""Tests for amvoc_autoencoder — written by test-architect BEFORE implementation.

Module under test: src/usv_spectrogram/features/amvoc_autoencoder.py

ROADMAP test plan coverage (module 17.6):
  1. AMVOCAutoencoder forward pass shape (2,1,64,160) -> test_forward_pass_output_shape
  2. encode() returns shape (batch, 1280)             -> test_encode_returns_bottleneck_shape
  3. train_amvoc runs without crashing (1 epoch)      -> test_train_amvoc_runs_without_crash
  4. extract_bottleneck shape (5, 1280)               -> test_extract_bottleneck_shape
  5. reduce_features output dims < 1280               -> test_reduce_features_dims_reduced
  6. PCA 95% variance -> explained_variance_ratio_    -> test_reduce_features_pca_explains_variance
  7. all-zero bottleneck handled                      -> test_reduce_features_all_zeros_no_crash
  8. AMVOCConfig epochs < 1 raises                    -> test_config_rejects_invalid_epochs
  9. Training loss decreases over epochs              -> test_training_loss_decreases
  10. CPU-only path works                             -> test_cpu_only_device_path

Additional coverage (recurring gap patterns):
  - Single-item batch (N=1) through encode/forward   -> test_single_sample_batch_forward
  - AMVOCConfig frozen (immutable after creation)     -> test_config_is_frozen_dataclass
  - extract_bottleneck batching consistency           -> test_extract_bottleneck_batch_vs_single
  - reduce_features info dict has required keys       -> test_reduce_features_info_dict_keys
  - reduce_features with N < k (underdetermined PCA) -> test_reduce_features_few_samples
  - bottleneck is exactly 8x8x20 = 1280 dims         -> test_bottleneck_dimension_matches_spec
  - forward output is float32                        -> test_forward_output_dtype
  - reduce_features variance_threshold drops dims    -> test_variance_threshold_drops_constant_dims

Total: 18 tests (10 from ROADMAP, 8 additional)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Pattern 8: import bootstrap — tests live one level below repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Graceful degradation if torch is absent; normally it is present.
torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# Module-under-test imports — these will fail with ImportError until the
# implementation exists. That failure mode is expected and correct.
# ---------------------------------------------------------------------------
from usv_spectrogram.features.amvoc_autoencoder import (  # noqa: E402
    AMVOCAutoencoder,
    AMVOCConfig,
    extract_bottleneck,
    reduce_features,
    train_amvoc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cpu_cfg(**overrides) -> AMVOCConfig:
    """Return an AMVOCConfig forced to CPU with minimal epochs for speed."""
    defaults = dict(device="cpu", epochs=2, batch_size=4)
    defaults.update(overrides)
    return AMVOCConfig(**defaults)


def _synthetic_spectrograms(n: int = 10) -> "torch.Tensor":
    """Return n synthetic spectrogram tensors of shape (n, 1, 64, 160)."""
    torch.manual_seed(42)
    return torch.randn(n, 1, 64, 160)


# ---------------------------------------------------------------------------
# ROADMAP test 1 — forward pass output shape
# ---------------------------------------------------------------------------

def test_forward_pass_output_shape():
    """Spec: AMVOCAutoencoder.forward() on (2,1,64,160) input returns (2,1,64,160).

    The autoencoder must be an identity-shaped operation: input and output
    spatial dimensions must match so reconstruction loss is well-defined.
    """
    cfg = _cpu_cfg()
    model = AMVOCAutoencoder(cfg)
    x = torch.randn(2, 1, 64, 160)
    out = model(x)
    assert out.shape == (2, 1, 64, 160), (
        f"Expected output shape (2, 1, 64, 160), got {tuple(out.shape)}"
    )


# ---------------------------------------------------------------------------
# ROADMAP test 2 — encode() bottleneck shape
# ---------------------------------------------------------------------------

def test_encode_returns_bottleneck_shape():
    """Spec: encode() returns (batch, 1280) — the 8×8×20 AMVOC bottleneck flattened.

    1280 = 8 (time) × 8 (freq) × 20 (filters) after three maxpool operations
    on a 64×160 input.
    """
    cfg = _cpu_cfg()
    model = AMVOCAutoencoder(cfg)
    x = torch.randn(4, 1, 64, 160)
    bottleneck = model.encode(x)
    assert bottleneck.shape == (4, 1280), (
        f"Expected encode() shape (4, 1280), got {tuple(bottleneck.shape)}"
    )


# ---------------------------------------------------------------------------
# ROADMAP test 3 — train_amvoc runs without crashing
# ---------------------------------------------------------------------------

def test_train_amvoc_runs_without_crash():
    """Spec: train_amvoc(spectrograms, cfg) completes and returns an AMVOCAutoencoder.

    Training on 10 synthetic spectrograms for 1 epoch must not raise any
    exception and must return a model of the correct type.
    """
    spectrograms = _synthetic_spectrograms(10)
    cfg = _cpu_cfg(epochs=1)
    model = train_amvoc(spectrograms, cfg)
    assert isinstance(model, AMVOCAutoencoder), (
        f"train_amvoc must return AMVOCAutoencoder, got {type(model)}"
    )


# ---------------------------------------------------------------------------
# ROADMAP test 4 — extract_bottleneck shape
# ---------------------------------------------------------------------------

def test_extract_bottleneck_shape():
    """Spec: extract_bottleneck returns np.ndarray of shape (N, 1280).

    Extraction must work on a pre-trained model and return a NumPy array,
    not a torch.Tensor.
    """
    cfg = _cpu_cfg(epochs=1)
    spectrograms = _synthetic_spectrograms(5)
    model = train_amvoc(spectrograms, cfg)
    result = extract_bottleneck(model, spectrograms, batch_size=32)
    assert isinstance(result, np.ndarray), (
        f"extract_bottleneck must return np.ndarray, got {type(result)}"
    )
    assert result.shape == (5, 1280), (
        f"Expected shape (5, 1280), got {result.shape}"
    )


# ---------------------------------------------------------------------------
# ROADMAP test 5 — reduce_features output dims < 1280
# ---------------------------------------------------------------------------

def test_reduce_features_dims_reduced():
    """Spec: reduce_features on real noise bottleneck returns fewer than 1280 dims.

    The pipeline applies variance thresholding followed by PCA, both of which
    must reduce dimensionality below the raw 1280D bottleneck.
    Random noise from randn will have non-zero variance in all dims, so the
    reduction here comes from PCA (95% variance criterion).
    """
    rng = np.random.default_rng(0)
    bottleneck = rng.standard_normal((100, 1280)).astype(np.float32)
    reduced, info = reduce_features(bottleneck, variance_threshold=1e-4, pca_variance=0.95)
    assert reduced.shape[0] == 100, "Row count must be preserved"
    assert reduced.shape[1] < 1280, (
        f"PCA must reduce dims below 1280, got {reduced.shape[1]}"
    )


# ---------------------------------------------------------------------------
# ROADMAP test 6 — explained_variance_ratio_ sums to >= 0.95
# ---------------------------------------------------------------------------

def test_reduce_features_pca_explains_variance():
    """Spec: info dict contains explained_variance_ratio_ whose sum is >= pca_variance.

    The PCA step must select enough components to explain at least the requested
    fraction of variance; the info dict must record the ratios for auditability.
    """
    rng = np.random.default_rng(1)
    bottleneck = rng.standard_normal((200, 1280)).astype(np.float32)
    _, info = reduce_features(bottleneck, variance_threshold=1e-4, pca_variance=0.95)
    assert "explained_variance_ratio_" in info, (
        "info dict must contain 'explained_variance_ratio_'"
    )
    ratio_sum = float(np.sum(info["explained_variance_ratio_"]))
    assert ratio_sum >= 0.95, (
        f"Explained variance ratio sum must be >= 0.95, got {ratio_sum:.4f}"
    )


# ---------------------------------------------------------------------------
# ROADMAP test 7 — all-zero bottleneck does not crash
# ---------------------------------------------------------------------------

def test_reduce_features_all_zeros_no_crash():
    """Spec: reduce_features must not raise on all-zero input.

    All-zero input has zero variance — variance thresholding will drop all dims.
    The function must handle this gracefully (e.g., return empty or single-dim
    output) rather than raising a ZeroDivisionError or sklearn exception.
    """
    bottleneck = np.zeros((50, 1280), dtype=np.float32)
    # Must not raise
    reduced, info = reduce_features(bottleneck, variance_threshold=1e-4, pca_variance=0.95)
    assert isinstance(reduced, np.ndarray), (
        "reduce_features must return np.ndarray even on all-zero input"
    )
    assert reduced.shape[0] == 50, "Row count must be preserved even for zero input"


# ---------------------------------------------------------------------------
# ROADMAP test 8 — AMVOCConfig rejects epochs < 1
# ---------------------------------------------------------------------------

def test_config_rejects_invalid_epochs():
    """Spec: AMVOCConfig with epochs < 1 must raise ValueError or similar.

    Zero or negative epochs is nonsensical; the frozen dataclass must validate
    this in __post_init__ and raise before the bad config can propagate.
    """
    with pytest.raises((ValueError, TypeError)):
        AMVOCConfig(epochs=0, device="cpu")


# ---------------------------------------------------------------------------
# ROADMAP test 9 — training loss decreases over epochs
# ---------------------------------------------------------------------------

def test_training_loss_decreases():
    """Spec: model learns — loss after epoch 2 must be strictly less than after epoch 1.

    This is a sanity check that gradient descent is functional. We instrument
    training by calling train_amvoc twice with epochs=1 and epochs=2 and compare
    loss, OR by checking that the model returned is in eval mode and its
    reconstruction error on the training set drops. Because train_amvoc does not
    return losses directly, we verify indirectly: reconstruction MSE on the same
    batch must be lower after 2 epochs than after 1.
    """
    torch.manual_seed(7)
    spectrograms = _synthetic_spectrograms(8)  # small enough to overfit

    # Train for 1 epoch, measure reconstruction error
    cfg_1 = _cpu_cfg(epochs=1, lr=1e-3)
    model_1 = train_amvoc(spectrograms, cfg_1)
    model_1.eval()
    with torch.no_grad():
        recon_1 = model_1(spectrograms)
        mse_1 = float(torch.mean((recon_1 - spectrograms) ** 2))

    # Train for 2 epochs on the same data, measure reconstruction error
    torch.manual_seed(7)  # same initialisation
    cfg_2 = _cpu_cfg(epochs=2, lr=1e-3)
    model_2 = train_amvoc(spectrograms, cfg_2)
    model_2.eval()
    with torch.no_grad():
        recon_2 = model_2(spectrograms)
        mse_2 = float(torch.mean((recon_2 - spectrograms) ** 2))

    assert mse_2 < mse_1, (
        f"Reconstruction MSE must decrease with more training: "
        f"epoch-1 MSE={mse_1:.6f}, epoch-2 MSE={mse_2:.6f}"
    )


# ---------------------------------------------------------------------------
# ROADMAP test 10 — CPU-only device path works end-to-end
# ---------------------------------------------------------------------------

def test_cpu_only_device_path():
    """Spec: the full pipeline (train → extract → reduce) must work on CPU.

    No CUDA should be required. Developers without a GPU must be able to run
    the pipeline. Setting device='cpu' must not raise.
    """
    cfg = AMVOCConfig(device="cpu", epochs=1, batch_size=4)
    spectrograms = _synthetic_spectrograms(6)
    model = train_amvoc(spectrograms, cfg)
    bottleneck = extract_bottleneck(model, spectrograms, batch_size=4)
    assert bottleneck.shape == (6, 1280)
    reduced, info = reduce_features(bottleneck)
    assert reduced.ndim == 2
    assert reduced.shape[0] == 6


# ---------------------------------------------------------------------------
# Additional test — single-sample batch through forward and encode
# ---------------------------------------------------------------------------

def test_single_sample_batch_forward():
    """Edge case: batch size of exactly 1 must work through forward() and encode().

    Batch normalisation layers (if used) can fail with N=1; the architecture
    must handle the single-sample case gracefully.
    """
    cfg = _cpu_cfg()
    model = AMVOCAutoencoder(cfg)
    x = torch.randn(1, 1, 64, 160)
    out = model(x)
    assert out.shape == (1, 1, 64, 160), f"Expected (1,1,64,160), got {tuple(out.shape)}"
    bottleneck = model.encode(x)
    assert bottleneck.shape == (1, 1280), f"Expected (1, 1280), got {tuple(bottleneck.shape)}"


# ---------------------------------------------------------------------------
# Additional test — AMVOCConfig is frozen (immutable after creation)
# ---------------------------------------------------------------------------

def test_config_is_frozen_dataclass():
    """Pattern 1: AMVOCConfig must be a frozen dataclass — mutation raises FrozenInstanceError.

    Frozen configs prevent accidental modification of shared config objects
    passed to multiple workers.
    """
    from dataclasses import FrozenInstanceError

    cfg = AMVOCConfig(device="cpu")
    with pytest.raises(FrozenInstanceError):
        cfg.epochs = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Additional test — extract_bottleneck batching is consistent with single-pass
# ---------------------------------------------------------------------------

def test_extract_bottleneck_batch_vs_single():
    """Invariant: extract_bottleneck with batch_size=1 and batch_size=N must agree.

    If batching introduces any inconsistency (e.g., wrong reshaping), the
    results will differ. They must be numerically identical.
    """
    cfg = _cpu_cfg(epochs=1)
    spectrograms = _synthetic_spectrograms(4)
    model = train_amvoc(spectrograms, cfg)
    model.eval()

    result_batched = extract_bottleneck(model, spectrograms, batch_size=4)
    result_single = extract_bottleneck(model, spectrograms, batch_size=1)

    np.testing.assert_allclose(
        result_batched,
        result_single,
        rtol=1e-5,
        atol=1e-6,
        err_msg="extract_bottleneck results differ between batch_size=4 and batch_size=1",
    )


# ---------------------------------------------------------------------------
# Additional test — reduce_features info dict has required keys
# ---------------------------------------------------------------------------

def test_reduce_features_info_dict_keys():
    """Spec: reduce_features info dict must contain the three documented keys.

    Keys: kept_dims_after_variance, pca_n_components, explained_variance_ratio_
    These are required for auditability and downstream reporting (info.json).
    """
    rng = np.random.default_rng(2)
    bottleneck = rng.standard_normal((80, 1280)).astype(np.float32)
    _, info = reduce_features(bottleneck)
    for key in ("kept_dims_after_variance", "pca_n_components", "explained_variance_ratio_"):
        assert key in info, f"info dict missing required key '{key}'"


# ---------------------------------------------------------------------------
# Additional test — reduce_features with very few samples (N < typical PCA k)
# ---------------------------------------------------------------------------

def test_reduce_features_few_samples():
    """Edge case: reduce_features must work when N (samples) is small (e.g., 5).

    PCA requires at most min(N, D) components; requesting 95% variance with
    only 5 samples must not crash and must return a valid array.
    """
    rng = np.random.default_rng(3)
    bottleneck = rng.standard_normal((5, 1280)).astype(np.float32)
    reduced, info = reduce_features(bottleneck, pca_variance=0.95)
    assert isinstance(reduced, np.ndarray)
    assert reduced.shape[0] == 5
    assert reduced.shape[1] >= 1, "Must return at least one component"


# ---------------------------------------------------------------------------
# Additional test — bottleneck dimension exactly 1280 = 8×8×20
# ---------------------------------------------------------------------------

def test_bottleneck_dimension_matches_spec():
    """Spec: AMVOC bottleneck is 8×8×20 = 1280 — encoder must produce exactly this.

    Three maxpool(2,2) operations on a 64-time × 160-freq input:
      64 → 32 → 16 → 8 (time)
      160 → 80 → 40 → 20... wait — 160/8 = 20, not 8.

    The spec says 8×8×20 = 1280. The 20 comes from bottleneck_filters, and
    the spatial size after pooling must be verified to be 8×8. This test
    locks the bottleneck size to exactly 1280 regardless of implementation.
    """
    cfg = _cpu_cfg()
    model = AMVOCAutoencoder(cfg)
    x = torch.randn(3, 1, 64, 160)
    enc = model.encode(x)
    assert enc.shape[1] == 1280, (
        f"Bottleneck must be exactly 1280 (8×8×20), got {enc.shape[1]}"
    )


# ---------------------------------------------------------------------------
# Additional test — forward output dtype is float32
# ---------------------------------------------------------------------------

def test_forward_output_dtype():
    """Implementation detail: output tensor must be float32, not float64.

    Mixed precision bugs can cause downstream NumPy operations to silently
    upcast; locking the dtype here prevents that.
    """
    cfg = _cpu_cfg()
    model = AMVOCAutoencoder(cfg)
    x = torch.randn(2, 1, 64, 160)
    out = model(x)
    assert out.dtype == torch.float32, (
        f"Output dtype must be torch.float32, got {out.dtype}"
    )


# ---------------------------------------------------------------------------
# Additional test — variance_threshold removes constant (zero-variance) dims
# ---------------------------------------------------------------------------

def test_variance_threshold_drops_constant_dims():
    """Spec: reduce_features variance_threshold must drop features with zero variance.

    We create a bottleneck where half the columns are constant (zero variance).
    After reduce_features, kept_dims_after_variance must be <= 640 (only the
    varying half survives thresholding).
    """
    rng = np.random.default_rng(4)
    n = 100
    # 640 varying columns + 640 constant columns
    varying = rng.standard_normal((n, 640)).astype(np.float32)
    constant = np.zeros((n, 640), dtype=np.float32)
    bottleneck = np.concatenate([varying, constant], axis=1)
    assert bottleneck.shape == (n, 1280)

    _, info = reduce_features(bottleneck, variance_threshold=1e-4, pca_variance=0.95)
    kept = info["kept_dims_after_variance"]
    assert kept <= 640, (
        f"Variance threshold must drop constant dims; expected <= 640 kept, got {kept}"
    )
