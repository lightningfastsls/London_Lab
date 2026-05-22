"""Statistical diagnostics for the cleaning-validation gate (Module 18.1).

Four falsifiable diagnostics measure how much cage-confound signal survives
each cleaning ablation. All four return a :class:`DiagnosticResult` with a
documented pass threshold so the gate decision is mechanical.

Diagnostics
-----------
- :func:`notch_injection_test` — train a small diagnostic VAE on the
  combined (A + B) spectrograms, inject a synthetic cage tone into a
  copy of cohort B, measure the K-NN migration rate of injected B
  samples toward cohort A in the embedding space. Pass: migration < 30%.

- :func:`per_band_cohens_d` — Cohen's d between cohorts on mean spectral
  power per 10 kHz sub-band. Pass: max ``|d|`` < 0.3.

- :func:`knn_same_cohort_rate` — for each sample, fraction of k-NN that
  share its cohort label. Pass: < 0.85.

- :func:`raw_pixel_pca_d` — Cohen's d on PC1 of flattened spectrograms.
  Pass: ``|d|`` < 1.5.

A small CPU-runnable VAE (``train_diagnostic_vae``) backs the K-NN-based
diagnostics. The architecture is intentionally tiny — 32-dim latent,
2-layer encoder/decoder. **Epoch budget scales with input feature count.**
For the 32×32 smoke-test cohorts 4-8 epochs suffices; for real 227×227
data (51,529 input features, ~50× the smoke regime) use ≥32 epochs.
Under-training on real data produces degenerate K-NN measurements — in
the Module 18.2a real-data run, 4 epochs caused
``notch_injection_migration = 1.0`` on the ``all_layers`` ablation
(false NO-GO); 32 epochs gave the correct 0.0. See
``docs/handoffs/cleaning-validation-report.md`` Interpretation section.

Cohen's d formula::

    d = (mean_A - mean_B) / sqrt((var_A + var_B) / 2)
"""
from __future__ import annotations

import math
import warnings
from collections import namedtuple
from typing import Callable, Optional

import numpy as np

# Sentinel default for `_inject_cage_tone.notch_depth_db`. The parameter
# is preserved for backward compatibility (2026-05-21 locked methodology)
# but its value is no longer consumed by the function body — the
# injection magnitude scales to INJECTION_SIGMA * local_std instead.
# Passing any non-default value triggers a DeprecationWarning so callers
# relying on the legacy fixed-dB semantics get a clear signal that those
# semantics are gone.
_NOTCH_DEPTH_DB_LEGACY_DEFAULT: float = 20.0


# ---------------------------------------------------------------------------
# Thresholds — bound to the spec in ROADMAP §18.1
# ---------------------------------------------------------------------------

_THRESHOLD_NOTCH_INJECTION: float = 0.30      # migration rate
_THRESHOLD_PER_BAND_COHENS_D: float = 0.30    # max |d|
_THRESHOLD_KNN_SAME_COHORT: float = 0.85      # fraction
_THRESHOLD_PCA_D: float = 1.50                # |d| on PC1

# Cage-tone injection scaling. The injection magnitude is set to
# ``INJECTION_SIGMA * local_std`` over the notch band so the perturbation
# is comparable across all 6 ablations (raw dB, baseline_only,
# mad_only, zscore_only, all_layers, soft_notch_only) — see
# ``_inject_cage_tone``. A fixed-dB offset (the previous behaviour) was
# fine for ``raw``/``baseline_only`` (dB-scale input) but completely
# saturated the band on the normalised ablations (input in ~[0, 1]),
# producing false-FAIL migration on the most important "all_layers"
# configuration.
INJECTION_SIGMA: float = 2.0

# Fallback injection magnitude used when ``local_std`` is too small to
# define a sensible perturbation scale (constant input over the notch
# band). 0.1 is large relative to a normalised [0, 1] cell but small
# relative to the historic 20 dB shift on dB-scale data — it cannot
# saturate either domain. The docstring of ``_inject_cage_tone`` records
# the trade-off.
_INJECTION_FALLBACK: float = 0.1
_INJECTION_STD_EPS: float = 1e-9


# ---------------------------------------------------------------------------
# DiagnosticResult dataclass
# ---------------------------------------------------------------------------


_DiagnosticResultBase = namedtuple(
    "_DiagnosticResultBase",
    ["name", "value", "threshold", "threshold_direction", "passed", "details"],
)


class DiagnosticResult(_DiagnosticResultBase):
    """Result of one diagnostic test with pass/fail against a threshold.

    Attributes
    ----------
    name:
        Human-readable identifier for the metric (e.g.
        ``"notch_injection_migration"``).
    value:
        Measured value of the diagnostic.
    threshold:
        Pass threshold; interpretation depends on ``threshold_direction``.
    threshold_direction:
        ``"less_than"`` (lower is better) or ``"greater_than"``.
    passed:
        ``True`` iff the measured value satisfies the threshold.
    details:
        Method-specific metrics (intermediate measurements, band info,
        etc.) for the generated report. Defaults to an empty dict.

    Implementation note
    -------------------
    namedtuple-based (not frozen dataclass) so that ``object.__setattr__``
    raises on existing field names -- the test
    ``test_diagnostic_result_dataclass_fields_and_types`` probes this
    contract. See cleaning_pipeline.CleaningConfig for the same pattern.
    """

    __slots__ = ()

    def __new__(
        cls,
        name: str,
        value: float,
        threshold: float,
        threshold_direction: str,
        passed: bool,
        details: Optional[dict] = None,
    ) -> "DiagnosticResult":
        return super().__new__(
            cls,
            name,
            value,
            threshold,
            threshold_direction,
            passed,
            details if details is not None else {},
        )


# ---------------------------------------------------------------------------
# Helpers — Cohen's d, pooled std, KNN
# ---------------------------------------------------------------------------


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled standard deviation (equal-variance formula).

    d = (mean_a - mean_b) / sqrt((var_a + var_b) / 2)

    Uses unbiased variance (ddof=1) so the formula matches the spec hand
    computation in ``test_cohens_d_formula_hand_computed_values``.
    Returns 0.0 when both samples are constant (degenerate denominator).
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2:
        return 0.0
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = math.sqrt(max((var_a + var_b) / 2.0, 0.0))
    if pooled <= 0.0:
        return 0.0
    return (float(np.mean(a)) - float(np.mean(b))) / pooled


def _default_band_edges_khz() -> list[tuple[float, float]]:
    """10 kHz bands from 20 to 120 kHz (10 bands)."""
    edges_khz = list(range(20, 130, 10))  # 20, 30, ..., 120
    return [(float(lo), float(hi)) for lo, hi in zip(edges_khz[:-1], edges_khz[1:])]


def _khz_to_bin_range(
    band_lo_khz: float,
    band_hi_khz: float,
    n_freq: int,
    sample_rate_hz: int = 250_000,
) -> tuple[int, int]:
    """Map a frequency band in kHz to inclusive freq-bin indices.

    The frequency axis is assumed linear from 0 to Nyquist over ``n_freq``
    bins. Returns (lo_bin, hi_bin) where ``lo_bin <= hi_bin``. Clipped to
    ``[0, n_freq)``.
    """
    nyq_khz = (sample_rate_hz / 2.0) / 1000.0
    if nyq_khz <= 0:
        return (0, 0)
    # Map directly to bin index. Each bin covers nyq_khz / (n_freq-1) kHz.
    bin_per_khz = (n_freq - 1) / nyq_khz if n_freq > 1 else 0.0
    lo_bin = max(0, int(math.floor(band_lo_khz * bin_per_khz)))
    hi_bin = min(n_freq - 1, int(math.ceil(band_hi_khz * bin_per_khz)))
    if hi_bin < lo_bin:
        hi_bin = lo_bin
    return (lo_bin, hi_bin)


def _knn_majority_label(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_labels: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return majority-vote nearest-neighbor labels for each query.

    Uses ``sklearn.neighbors.NearestNeighbors`` if available, else a
    vectorised numpy fallback. Reference can be the same as query as long
    as the caller has not de-duplicated (we sort distances and take the k
    nearest in order; self-match handling is the caller's responsibility).
    """
    n_ref = reference_embeddings.shape[0]
    k_eff = min(k, n_ref)
    if k_eff <= 0:
        raise ValueError(f"k must be >= 1 and reference must be non-empty (k_eff={k_eff})")

    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
        nn.fit(reference_embeddings)
        _, idx = nn.kneighbors(query_embeddings, n_neighbors=k_eff)
    except Exception:
        # Numpy fallback — pairwise L2 then argpartition.
        diff = query_embeddings[:, None, :] - reference_embeddings[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        idx = np.argpartition(dists, k_eff - 1, axis=1)[:, :k_eff]

    neighbour_labels = reference_labels[idx]  # (n_query, k_eff)
    # Majority vote per row.
    majority = np.empty(query_embeddings.shape[0], dtype=reference_labels.dtype)
    for i in range(query_embeddings.shape[0]):
        vals, counts = np.unique(neighbour_labels[i], return_counts=True)
        majority[i] = vals[np.argmax(counts)]
    return majority


# ---------------------------------------------------------------------------
# train_diagnostic_vae — tiny CPU-runnable VAE
# ---------------------------------------------------------------------------


def _train_diagnostic_vae_with_encoder(
    spectrograms: np.ndarray,
    latent_dim: int = 32,
    n_epochs: int = 6,
    device: str = "cuda",
    seed: int = 0,
):
    """Internal: train a small VAE and return (embeddings, encode_fn, stats).

    ``encode_fn`` accepts a new ``(n, n_freq, n_time)`` array and returns
    its latent embeddings using the SAME encoder weights — used by
    :func:`notch_injection_test` to project injected samples through the
    encoder trained on (A + B).
    """
    import torch
    from torch import nn

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)

    # Deterministic init for the notch-injection re-encode path so the
    # baseline VAE and the injected VAE share the same weights.
    torch.manual_seed(int(seed))
    np_rng = np.random.default_rng(int(seed))

    n_input, n_freq, n_time = spectrograms.shape
    input_dim = n_freq * n_time
    hidden_dim = max(64, latent_dim * 2)

    class _DiagnosticVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc1 = nn.Linear(input_dim, hidden_dim)
            self.enc_mu = nn.Linear(hidden_dim, latent_dim)
            self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
            self.dec1 = nn.Linear(latent_dim, hidden_dim)
            self.dec_out = nn.Linear(hidden_dim, input_dim)

        def encode(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            h = torch.relu(self.enc1(x))
            return self.enc_mu(h), self.enc_logvar(h)

        def reparam(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z: "torch.Tensor") -> "torch.Tensor":
            h = torch.relu(self.dec1(z))
            return self.dec_out(h)

        def forward(self, x: "torch.Tensor") -> tuple[
            "torch.Tensor", "torch.Tensor", "torch.Tensor"
        ]:
            mu, logvar = self.encode(x)
            z = self.reparam(mu, logvar)
            return self.decode(z), mu, logvar

    # Standardise inputs to zero mean / unit variance so the VAE has a
    # well-conditioned target. Stats stored so encode_fn can reuse them
    # for projecting external (e.g. injected) samples.
    flat = spectrograms.reshape(n_input, -1).astype(np.float32)
    mean = float(flat.mean()) if flat.size else 0.0
    std = float(flat.std()) if flat.size else 1.0
    if std < 1e-8:
        std = 1.0
    flat_std = (flat - mean) / std

    x_all = torch.from_numpy(flat_std).to(torch_device)
    model = _DiagnosticVAE().to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = max(1, min(16, n_input))

    model.train()
    for _ in range(max(1, int(n_epochs))):
        perm = torch.randperm(n_input, device=torch_device)
        for start in range(0, n_input, batch_size):
            idx = perm[start:start + batch_size]
            xb = x_all[idx]
            recon, mu, logvar = model(xb)
            recon_loss = nn.functional.mse_loss(recon, xb, reduction="mean")
            logvar_c = torch.clamp(logvar, min=-10.0, max=10.0)
            kl = -0.5 * torch.mean(1 + logvar_c - mu.pow(2) - logvar_c.exp())
            loss = recon_loss + 1e-3 * kl
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(x_all)
    embeddings = mu.cpu().numpy().astype(np.float32)
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    def encode_fn(new_specs: np.ndarray) -> np.ndarray:
        """Embed new spectrograms with the SAME encoder weights/normalisation."""
        n_new = new_specs.shape[0]
        nflat = new_specs.reshape(n_new, -1).astype(np.float32)
        nflat = (nflat - mean) / std
        with torch.no_grad():
            xt = torch.from_numpy(nflat).to(torch_device)
            mu_new, _ = model.encode(xt)
        out = mu_new.cpu().numpy().astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    return embeddings, encode_fn, {"mean": mean, "std": std}


def train_diagnostic_vae(
    spectrograms: np.ndarray,
    latent_dim: int = 32,
    n_epochs: int = 6,
    device: str = "cuda",
) -> np.ndarray:
    """Train a small diagnostic VAE and return latent embeddings.

    A 2-layer MLP encoder/decoder over flattened spectrograms with a
    32-dim Gaussian latent (mean+logvar).

    **Critical: ``n_epochs`` must scale with input feature count.**
    The smoke-test data (32×32 = 1,024 features) converges in 4-8
    epochs. Real-data spectrograms (227×227 = 51,529 features) need
    ≥32 epochs; the Module 18.2a real-data run found that 4 epochs
    silently produced ``notch_injection_migration = 1.0`` on the
    ``all_layers`` ablation — a false NO-GO. 32 epochs gave 0.0. The
    default value below is the smoke-test default; callers using
    real data MUST override it.

    Parameters
    ----------
    spectrograms:
        Shape ``(n_input, n_freq, n_time)`` float array.
    latent_dim:
        Latent dimensionality. Defaults to 32 (spec).
    n_epochs:
        Number of training epochs. Smoke-test default (synthetic 32×32):
        4-8. Real 227×227 data: ≥32 (under-training produces degenerate
        diagnostic outputs — see module docstring + Module 18.2a report).
    device:
        ``"cuda"`` or ``"cpu"``. Falls back to CPU when CUDA unavailable.

    Returns
    -------
    embeddings:
        Shape ``(n_input, latent_dim)`` float32 latent means.
    """
    embeddings, _encode_fn, _stats = _train_diagnostic_vae_with_encoder(
        spectrograms, latent_dim=latent_dim, n_epochs=n_epochs, device=device,
    )
    return embeddings


# ---------------------------------------------------------------------------
# notch_injection_test
# ---------------------------------------------------------------------------


def _inject_cage_tone(
    spectrograms: np.ndarray,
    notch_band_khz: tuple[float, float],
    notch_depth_db: float,
    sample_rate_hz: int = 250_000,
) -> np.ndarray:
    """Inject a synthetic cage tone in the notch band of every spectrogram.

    The injection magnitude is scaled to the local distribution of the
    band: ``offset = INJECTION_SIGMA * local_std`` where ``local_std`` is
    computed across all ``(sample, freq, time)`` cells inside the notch
    band. This produces a consistent perturbation across all 6 ablations
    rather than a fixed-magnitude shift that would saturate the band on
    normalized-input ablations (``mad_only``, ``zscore_only``,
    ``all_layers`` — input in ~[0, 1]) and under-perturb the dB-scale
    ablations (``raw``, ``baseline_only`` — input in dB).

    **DEPRECATED parameter: ``notch_depth_db``.** Retained in the
    function signature for backward compatibility with callers from the
    locked methodology (2026-05-21), but its value is **not consumed**:
    scaling to the local distribution preserves the migration
    measurement semantics on every ablation. The original fixed-dB
    behaviour was the source of false-FAIL migration on the
    ``all_layers`` configuration — the gate's most important
    measurement. Passing any non-default value emits a
    ``DeprecationWarning``.

    When ``local_std`` is below ``_INJECTION_STD_EPS`` (constant input
    over the notch band), the function falls back to a small fixed
    offset ``_INJECTION_FALLBACK`` (0.1) — large relative to a [0, 1]
    cell, small relative to the legacy +20 dB shift, and safe on both
    domains.
    """
    if notch_depth_db != _NOTCH_DEPTH_DB_LEGACY_DEFAULT:
        warnings.warn(
            "_inject_cage_tone: `notch_depth_db` is deprecated and has no "
            f"effect (received {notch_depth_db!r}; legacy default was "
            f"{_NOTCH_DEPTH_DB_LEGACY_DEFAULT}). The injection magnitude "
            "is now INJECTION_SIGMA * local_std for consistency across "
            "ablations — see function docstring.",
            DeprecationWarning,
            stacklevel=2,
        )

    n, n_freq, n_time = spectrograms.shape
    lo, hi = notch_band_khz
    lo_bin, hi_bin = _khz_to_bin_range(lo, hi, n_freq, sample_rate_hz=sample_rate_hz)
    if hi_bin < lo_bin:
        return spectrograms.copy()

    injected = spectrograms.copy()
    band_slice = injected[:, lo_bin:hi_bin + 1, :]
    local_std = float(np.std(band_slice.ravel()))
    if local_std < _INJECTION_STD_EPS:
        injection_offset = _INJECTION_FALLBACK
    else:
        injection_offset = INJECTION_SIGMA * local_std
    injected[:, lo_bin:hi_bin + 1, :] = band_slice + injection_offset
    return injected


def notch_injection_test(
    spectrograms_by_cohort: dict[str, np.ndarray],
    notch_band_khz: tuple[float, float] = (50.4, 51.0),
    notch_depth_db: float = 20.0,
    n_epochs: int = 4,
    k: int = 5,
) -> DiagnosticResult:
    """K-NN migration rate test using a per-pair 32-dim diagnostic VAE.

    Methodology (locked 2026-05-21):
        1. Take the first two cohorts (A, B) from ``spectrograms_by_cohort``.
        2. Train a small 32-dim VAE on the **combined (A + B)**
           spectrograms. Training on A only would bias the latent space
           toward A's features; combined training gives a neutral
           embedding for migration measurement.
        3. Embed both cohorts in the same VAE -> baseline embedding space.
        4. Inject a synthetic cage tone into a COPY of cohort B's
           spectrograms.
        5. Embed the injected cohort B samples in the same VAE.
        6. For each injected sample, find its k nearest neighbours among
           (original cohort A + original cohort B) embeddings. Majority
           label = predicted cohort.
        7. Migration rate = fraction of injected B samples whose majority
           label is cohort A.

    Pass: migration rate < 30% (raw baseline 91.7% on our VAE, 58.5% on
    DeepSqueak's per PLAN §"Phase 1.0").
    """
    cohort_ids = list(spectrograms_by_cohort.keys())
    if len(cohort_ids) < 2:
        return DiagnosticResult(
            name="notch_injection_migration",
            value=0.0,
            threshold=_THRESHOLD_NOTCH_INJECTION,
            threshold_direction="less_than",
            passed=True,
            details={"reason": "fewer than 2 cohorts; test skipped"},
        )

    a_id, b_id = cohort_ids[0], cohort_ids[1]
    specs_a = np.asarray(spectrograms_by_cohort[a_id], dtype=np.float32)
    specs_b = np.asarray(spectrograms_by_cohort[b_id], dtype=np.float32)

    if specs_a.ndim != 3 or specs_b.ndim != 3:
        raise ValueError(
            "spectrograms must be 3-D (n_specs, n_freq, n_time); "
            f"got A={specs_a.shape}, B={specs_b.shape}"
        )

    # Step 4 — inject cage tone into cohort B copy
    specs_b_inj = _inject_cage_tone(
        specs_b, notch_band_khz, notch_depth_db,
    )

    # Step 2/3 — train ONE VAE on the baseline (A + B) pair and embed
    # every sample (including the injected copy) through the SAME encoder
    # weights. This is the locked methodology — re-training on the
    # injected data would leak the cage tone into the latent geometry and
    # bury the migration signal.
    n_a = specs_a.shape[0]
    n_b = specs_b.shape[0]
    train_specs = np.concatenate([specs_a, specs_b], axis=0)
    train_embed, encode_fn, _stats = _train_diagnostic_vae_with_encoder(
        train_specs, latent_dim=32, n_epochs=n_epochs, device="cpu",
    )
    embed_a = train_embed[:n_a]
    embed_b = train_embed[n_a:]

    # Step 5 — project injected samples through the same encoder
    embed_b_inj = encode_fn(specs_b_inj)

    # Step 6 — K-NN majority label for each injected sample, reference
    # set = original A + original B embeddings (NOT the injected samples).
    reference = np.concatenate([embed_a, embed_b], axis=0)
    reference_labels = np.array(
        [a_id] * n_a + [b_id] * n_b
    )
    majority = _knn_majority_label(embed_b_inj, reference, reference_labels, k=k)
    migration_rate = float(np.mean(majority == a_id))

    return DiagnosticResult(
        name="notch_injection_migration",
        value=migration_rate,
        threshold=_THRESHOLD_NOTCH_INJECTION,
        threshold_direction="less_than",
        passed=migration_rate < _THRESHOLD_NOTCH_INJECTION,
        details={
            "cohort_a": a_id,
            "cohort_b": b_id,
            "notch_band_khz": notch_band_khz,
            "notch_depth_db": notch_depth_db,
            "k": k,
            "n_epochs": n_epochs,
            "n_injected_samples": int(n_b),
            "baseline_a_embed_norm": float(np.linalg.norm(embed_a)),
            "baseline_b_embed_norm": float(np.linalg.norm(embed_b)),
        },
    )


# ---------------------------------------------------------------------------
# per_band_cohens_d
# ---------------------------------------------------------------------------


def per_band_cohens_d(
    spectrograms_by_cohort: dict[str, np.ndarray],
    band_edges_khz: Optional[list[tuple[float, float]]] = None,
    sample_rate_hz: int = 250_000,
) -> DiagnosticResult:
    """Max Cohen's d between cohorts on per-pixel spectral power per 10 kHz band.

    Flatten all ``(sample, freq_bin, time_frame)`` cells inside the band
    into a per-pixel distribution, then compute Cohen's d between cohort
    distributions. Per-sample-mean pooling would inflate ``|d|`` ~10x by
    underestimating variance (the per-sample mean variance shrinks by a
    factor of ``1/(n_freq * n_time)``), so the diagnostic uses raw
    per-pixel pooling instead. Pass: max ``|d|`` < 0.3.
    """
    if band_edges_khz is None:
        band_edges_khz = _default_band_edges_khz()

    cohort_ids = list(spectrograms_by_cohort.keys())
    if len(cohort_ids) < 2:
        return DiagnosticResult(
            name="per_band_cohens_d",
            value=0.0,
            threshold=_THRESHOLD_PER_BAND_COHENS_D,
            threshold_direction="less_than",
            passed=True,
            details={
                "reason": "fewer than 2 cohorts; test skipped",
                "band_edges_khz": band_edges_khz,
                "n_bands": len(band_edges_khz),
            },
        )

    max_abs_d = 0.0
    signed_max_d = 0.0
    best_band: tuple[float, float] = band_edges_khz[0]
    best_pair: tuple[str, str] = (cohort_ids[0], cohort_ids[1])

    # Per-pixel band slice per cohort — flatten ALL (sample, freq, time)
    # pixels inside the band into a 1-D array. Cohen's d is then computed
    # on raw pixel-level distributions, NOT on per-sample band means.
    # This matches ``test_cohens_d_formula_hand_computed_values`` which
    # asserts |d| ~= 4 when the raw per-pixel std is 5 and the mean shift
    # is 20 — per-sample-mean pooling would give |d| ~= 80 (variance
    # shrinks by 1/(n_freq*n_time)).
    per_cohort_band_pixels: dict[str, list[np.ndarray]] = {cid: [] for cid in cohort_ids}
    for cid, specs in spectrograms_by_cohort.items():
        specs_arr = np.asarray(specs, dtype=np.float64)
        if specs_arr.ndim != 3:
            raise ValueError(
                f"spectrograms for cohort {cid!r} must be 3-D; got {specs_arr.shape}"
            )
        n_freq = specs_arr.shape[1]
        for b_idx, (lo, hi) in enumerate(band_edges_khz):
            lo_bin, hi_bin = _khz_to_bin_range(lo, hi, n_freq, sample_rate_hz)
            band_slice = specs_arr[:, lo_bin:hi_bin + 1, :]
            per_cohort_band_pixels[cid].append(band_slice.ravel())

    # Pairwise Cohen's d per band — pick max over (pair, band)
    for i, a_id in enumerate(cohort_ids):
        for b_id in cohort_ids[i + 1:]:
            for b_idx, band in enumerate(band_edges_khz):
                d = _cohens_d(
                    per_cohort_band_pixels[a_id][b_idx],
                    per_cohort_band_pixels[b_id][b_idx],
                )
                if abs(d) > max_abs_d:
                    max_abs_d = abs(d)
                    signed_max_d = d
                    best_band = band
                    best_pair = (a_id, b_id)

    return DiagnosticResult(
        name="per_band_cohens_d",
        value=signed_max_d,
        threshold=_THRESHOLD_PER_BAND_COHENS_D,
        threshold_direction="less_than",
        passed=abs(signed_max_d) < _THRESHOLD_PER_BAND_COHENS_D,
        details={
            "band_edges_khz": band_edges_khz,
            "n_bands": len(band_edges_khz),
            "max_d_band_khz": best_band,
            "max_d_pair": best_pair,
            "max_abs_d": max_abs_d,
        },
    )


# ---------------------------------------------------------------------------
# knn_same_cohort_rate
# ---------------------------------------------------------------------------


def knn_same_cohort_rate(
    embeddings_by_cohort: dict[str, np.ndarray],
    k: int = 5,
) -> DiagnosticResult:
    """For each sample, fraction of k-NN that share its cohort label.

    Pass: < 0.85 (raw baseline 0.98-1.0). High values indicate the cohort
    is linearly separable in the embedding space -> classifier is at risk
    of learning cohort identity instead of biology.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    cohort_ids = list(embeddings_by_cohort.keys())
    if len(cohort_ids) < 2:
        return DiagnosticResult(
            name="knn_same_cohort_rate",
            value=0.0,
            threshold=_THRESHOLD_KNN_SAME_COHORT,
            threshold_direction="less_than",
            passed=True,
            details={"reason": "fewer than 2 cohorts; test skipped"},
        )

    all_embeds = []
    all_labels = []
    for cid, embeds in embeddings_by_cohort.items():
        e = np.asarray(embeds, dtype=np.float32)
        if e.ndim != 2:
            raise ValueError(
                f"embeddings for cohort {cid!r} must be 2-D (n_samples, embed_dim); "
                f"got {e.shape}"
            )
        all_embeds.append(e)
        all_labels.extend([cid] * e.shape[0])
    all_embeds_arr = np.concatenate(all_embeds, axis=0)
    all_labels_arr = np.array(all_labels)

    # For each query, the k nearest neighbours in the COMBINED set, EXCLUDING
    # the query itself (use k+1 neighbours and drop the self-match).
    k_eff = k + 1
    n_total = all_embeds_arr.shape[0]
    k_eff = min(k_eff, n_total)
    if k_eff <= 1:
        raise ValueError(
            f"insufficient samples for KNN (n_total={n_total}, k={k})"
        )

    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
        nn.fit(all_embeds_arr)
        _, idx = nn.kneighbors(all_embeds_arr, n_neighbors=k_eff)
    except Exception:
        diff = all_embeds_arr[:, None, :] - all_embeds_arr[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        idx = np.argpartition(dists, k_eff - 1, axis=1)[:, :k_eff]

    # Drop the self-match (which has distance 0) — guaranteed to be first
    # for sklearn auto-algorithm; otherwise filter.
    n_query = all_embeds_arr.shape[0]
    self_idx = np.arange(n_query)
    same_cohort_fractions = np.empty(n_query, dtype=np.float64)
    for i in range(n_query):
        neigh = idx[i]
        neigh = neigh[neigh != self_idx[i]][:k]
        if len(neigh) == 0:
            same_cohort_fractions[i] = 0.0
            continue
        neigh_labels = all_labels_arr[neigh]
        same_cohort_fractions[i] = float(np.mean(neigh_labels == all_labels_arr[i]))

    mean_rate = float(np.mean(same_cohort_fractions))

    return DiagnosticResult(
        name="knn_same_cohort_rate",
        value=mean_rate,
        threshold=_THRESHOLD_KNN_SAME_COHORT,
        threshold_direction="less_than",
        passed=mean_rate < _THRESHOLD_KNN_SAME_COHORT,
        details={
            "k": k,
            "n_total_samples": int(n_total),
            "cohort_sizes": {cid: int(np.asarray(embeddings_by_cohort[cid]).shape[0])
                             for cid in cohort_ids},
        },
    )


# ---------------------------------------------------------------------------
# raw_pixel_pca_d
# ---------------------------------------------------------------------------


def raw_pixel_pca_d(
    spectrograms_by_cohort: dict[str, np.ndarray],
    n_components: int = 1,
) -> DiagnosticResult:
    """Cohen's d on PC1 scores of flattened spectrograms between cohorts.

    Pass: ``|d|`` < 1.5 (raw observation +5.83 on our VAE data per
    PLAN §"Phase 1.0").
    """
    cohort_ids = list(spectrograms_by_cohort.keys())
    if len(cohort_ids) < 2:
        return DiagnosticResult(
            name="raw_pixel_pca_d",
            value=0.0,
            threshold=_THRESHOLD_PCA_D,
            threshold_direction="less_than",
            passed=True,
            details={"reason": "fewer than 2 cohorts; test skipped"},
        )

    flat_chunks: list[np.ndarray] = []
    labels: list[str] = []
    for cid, specs in spectrograms_by_cohort.items():
        s = np.asarray(specs, dtype=np.float32)
        if s.ndim != 3:
            raise ValueError(
                f"spectrograms for {cid!r} must be 3-D; got {s.shape}"
            )
        flat_chunks.append(s.reshape(s.shape[0], -1))
        labels.extend([cid] * s.shape[0])
    flat = np.concatenate(flat_chunks, axis=0).astype(np.float64)
    labels_arr = np.array(labels)

    try:
        from sklearn.decomposition import PCA
        n_pc = max(1, min(int(n_components), min(flat.shape) - 1))
        pca = PCA(n_components=n_pc)
        scores = pca.fit_transform(flat)  # (n_total, n_pc)
        pc1 = scores[:, 0]
    except Exception:
        # numpy fallback — centre + SVD, take PC1
        centred = flat - flat.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(centred, full_matrices=False)
        pc1 = (u[:, 0] * s[0]).astype(np.float64)

    # Cohen's d between first two cohorts on PC1
    a_id, b_id = cohort_ids[0], cohort_ids[1]
    mask_a = labels_arr == a_id
    mask_b = labels_arr == b_id
    d = _cohens_d(pc1[mask_a], pc1[mask_b])

    return DiagnosticResult(
        name="raw_pixel_pca_d",
        value=d,
        threshold=_THRESHOLD_PCA_D,
        threshold_direction="less_than",
        passed=abs(d) < _THRESHOLD_PCA_D,
        details={
            "cohort_pair": (a_id, b_id),
            "n_components": int(n_components),
            "n_samples_total": int(flat.shape[0]),
            "n_features_per_sample": int(flat.shape[1]),
        },
    )


# ---------------------------------------------------------------------------
# Diagnostic registry (used by the CLI)
# ---------------------------------------------------------------------------


def run_all_diagnostics(
    spectrograms_by_cohort: dict[str, np.ndarray],
    *,
    sample_rate_hz: int = 250_000,
    n_epochs: int = 4,
    knn_k: int = 5,
) -> list[DiagnosticResult]:
    """Run all four diagnostics and return them in canonical order.

    The KNN diagnostic operates on a small diagnostic VAE embedding rather
    than raw pixels (it is more sensitive). The VAE is trained on the
    concatenated cohort spectrograms.
    """
    notch = notch_injection_test(
        spectrograms_by_cohort, n_epochs=n_epochs,
    )
    bands = per_band_cohens_d(
        spectrograms_by_cohort, sample_rate_hz=sample_rate_hz,
    )
    pca = raw_pixel_pca_d(spectrograms_by_cohort)

    # KNN on VAE embeddings
    embeddings_by_cohort: dict[str, np.ndarray] = {}
    all_specs = []
    cohort_sizes: dict[str, int] = {}
    for cid, specs in spectrograms_by_cohort.items():
        s = np.asarray(specs, dtype=np.float32)
        all_specs.append(s)
        cohort_sizes[cid] = s.shape[0]
    combined = np.concatenate(all_specs, axis=0)
    combined_embed = train_diagnostic_vae(
        combined, latent_dim=32, n_epochs=n_epochs, device="cpu",
    )
    cursor = 0
    for cid, n in cohort_sizes.items():
        embeddings_by_cohort[cid] = combined_embed[cursor:cursor + n]
        cursor += n
    knn = knn_same_cohort_rate(embeddings_by_cohort, k=knn_k)

    return [notch, bands, knn, pca]
