"""Module 18.4 — VAE falsifiable cage test, re-run on v2 ENCODER features.

ROADMAP §18.4 file 4. Module 18.1 ran four falsifiable cage-confound tests on
*raw spectrograms* (and on a small diagnostic VAE's latents). This script
re-runs the same four criteria on the **v2 DANN encoder's features** — stronger
evidence than the raw-pixel pass, because it asks whether cage identity survives
*after* the encoder has been adversarially trained to remove it.

The four criteria (thresholds imported from ``diagnostics`` so they never drift):

  1. **k-NN same-cohort rate** (< 0.85) — for each feature vector, the fraction
     of its k nearest neighbours that share its cohort. Cage-invariant features
     ⇒ neighbours are cohort-mixed ⇒ rate near 0.5. Reused directly from 18.1
     (it is cohort-agnostic about the input space).
  2. **PCA PC1 Cohen's d** (< 1.50) — separation between cohorts along the first
     principal component of the features. 18.1's ``raw_pixel_pca_d`` requires
     3-D spectrogram inputs, so for 512-d feature vectors we compute PC1 inline
     (SVD on the centred features) and reuse the project's ``_cohens_d`` on the
     PC1 scores for formula consistency.
  3. **Per-dimension max Cohen's d** (< 0.30) — the feature-space analogue of
     18.1's per-band Cohen's d. 18.1 measured per-10-kHz-band pixel power; in
     feature space there are no frequency bands, so we measure the maximum
     per-feature-dimension Cohen's d between cohorts. Implemented inline and
     labelled as the analogue.
  4. **Notch-injection migration** (< 0.30) — inject a frequency-band notch into
     target patches, push them through the encoder, and measure the k-NN
     migration rate toward the source cohort in *feature* space. A cage-invariant
     encoder should not let a low-level pixel perturbation flip cohort identity.

All four must pass for the 18.4 VAE gate. Output: ``cage_invariance_probe.md``.

This script needs a trained v2 ``best.pt`` and the patch pool, so it is run on
the rig after GPU training (step 7). Locally it is import/compile-validated.

Reference: Ganin & Lempitsky 2015; Module 18.1 diagnostics
(``src/usv_spectrogram/classifier/diagnostics.py``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
_SCRIPTS = Path(__file__).resolve().parent
for _p in (str(_SRC), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from usv_spectrogram.classifier.dann import ResNet18DANN
from usv_spectrogram.classifier.diagnostics import (
    DiagnosticResult,
    knn_same_cohort_rate,
    _cohens_d,
    _THRESHOLD_KNN_SAME_COHORT,
    _THRESHOLD_NOTCH_INJECTION,
    _THRESHOLD_PCA_D,
    _THRESHOLD_PER_BAND_COHENS_D,
)
from train_lab_classifier import _DEFAULT_IMAGE_SIZE

_TRANSFORM = transforms.Compose([
    transforms.Resize((_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


# ---------------------------------------------------------------------------
# Patch loading + encoder feature extraction
# ---------------------------------------------------------------------------

def _resolve(raw: str, roots: list[Path]) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    for r in roots:
        if (r / p).exists():
            return r / p
    return roots[0] / p


def _load_patches(paths: list[str], roots: list[Path], notch_frac: float = 0.0) -> torch.Tensor:
    """Load patches → (N, 3, H, W). If ``notch_frac`` > 0, zero a horizontal band
    (the patch's vertical axis is frequency) to simulate a tonal notch."""
    tensors = []
    for raw in paths:
        with Image.open(_resolve(raw, roots)) as im:
            t = _TRANSFORM(im.convert("L"))
        if notch_frac > 0.0:
            h = t.shape[1]
            band = max(1, int(h * notch_frac))
            start = (h - band) // 2
            t[:, start:start + band, :] = 0.0
        tensors.append(t)
    return torch.stack(tensors)


@torch.no_grad()
def _encode(encoder: torch.nn.Module, images: torch.Tensor, device: str, batch: int = 64) -> np.ndarray:
    encoder.eval()
    feats = []
    for i in range(0, images.shape[0], batch):
        out = encoder(images[i:i + batch].to(device))
        feats.append(out.reshape(out.shape[0], -1).cpu().numpy())
    return np.concatenate(feats, axis=0) if feats else np.empty((0, 0))


def _sample_paths(csv_path: Path, cohort: str | None, max_n: int, seed: int = 0) -> list[str]:
    df = pd.read_csv(csv_path)
    if cohort is not None and "cohort" in df.columns:
        df = df[df["cohort"].astype(str) == cohort]
    if len(df) > max_n:
        df = df.sample(n=max_n, random_state=seed)
    return df["path"].astype(str).tolist()


# ---------------------------------------------------------------------------
# Feature-space test 3: per-dimension max Cohen's d (per_band analogue)
# ---------------------------------------------------------------------------

def pc1_cohens_d_features(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    """|Cohen's d| between cohorts on PC1 of the combined features.

    Feature-space version of 18.1's ``raw_pixel_pca_d`` (which only accepts 3-D
    spectrograms). PC1 is the top right-singular vector of the centred combined
    feature matrix; we project each cohort onto it and reuse ``_cohens_d``.
    """
    if len(feats_a) == 0 or len(feats_b) == 0:
        return 0.0
    X = np.concatenate([feats_a, feats_b], axis=0).astype(np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    scores = Xc @ vt[0]
    return abs(_cohens_d(scores[:len(feats_a)], scores[len(feats_a):]))


def per_dimension_max_cohens_d(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    """Max |Cohen's d| across feature dimensions between two cohorts.

    Feature-space analogue of 18.1's per-band Cohen's d. For each of the D
    feature dimensions, compute the standardised mean difference; return the
    maximum absolute value. Cage-invariant features ⇒ no single dimension
    strongly separates cohorts ⇒ low max d.
    """
    ma, mb = feats_a.mean(0), feats_b.mean(0)
    va, vb = feats_a.var(0, ddof=1), feats_b.var(0, ddof=1)
    na, nb = len(feats_a), len(feats_b)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(1, (na + nb - 2)))
    pooled = np.where(pooled <= 0, np.nan, pooled)
    d = np.abs(ma - mb) / pooled
    return float(np.nanmax(d)) if np.isfinite(d).any() else 0.0


# ---------------------------------------------------------------------------
# Feature-space test 4: notch-injection migration
# ---------------------------------------------------------------------------

def notch_migration_in_feature_space(
    source_feats: np.ndarray, target_feats_clean: np.ndarray,
    target_feats_notched: np.ndarray, k: int = 5,
) -> float:
    """Fraction of notched target features whose k-NN are majority source.

    Baseline = {source clean} ∪ {target clean}. For each notched-target feature,
    find its k nearest neighbours in the baseline; "migrated" if the majority
    are source. Cage-invariant encoder ⇒ a notch should not flip a target patch
    into the source neighbourhood ⇒ low migration.
    """
    if len(target_feats_notched) == 0:
        return 0.0
    base = np.concatenate([source_feats, target_feats_clean], axis=0)
    base_is_source = np.concatenate([
        np.ones(len(source_feats), bool), np.zeros(len(target_feats_clean), bool)])
    migrated = 0
    for f in target_feats_notched:
        dist = np.linalg.norm(base - f[None, :], axis=1)
        nn = np.argsort(dist)[:k]
        if base_is_source[nn].mean() > 0.5:
            migrated += 1
    return migrated / len(target_feats_notched)


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def load_v2_encoder(ckpt_path: Path, num_classes: int, num_domains: int, device: str):
    model = ResNet18DANN(num_classes=num_classes, num_domains=num_domains, pretrained=False)
    blob = torch.load(ckpt_path, map_location="cpu")
    sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    model.load_state_dict(sd, strict=False)
    return model.encoder.to(device)


def run_diagnostic(
    encoder, source_paths, target_paths, source_roots, target_roots,
    device, k, notch_frac,
) -> list[DiagnosticResult]:
    src_clean = _encode(encoder, _load_patches(source_paths, source_roots), device)
    tgt_clean = _encode(encoder, _load_patches(target_paths, target_roots), device)
    tgt_notched = _encode(
        encoder, _load_patches(target_paths, target_roots, notch_frac=notch_frac), device)

    emb_by_cohort = {"vocalmat": src_clean, "lab": tgt_clean}

    knn = knn_same_cohort_rate(emb_by_cohort, k=k)
    pc1_d = pc1_cohens_d_features(src_clean, tgt_clean)
    per_dim = per_dimension_max_cohens_d(src_clean, tgt_clean)
    migration = notch_migration_in_feature_space(src_clean, tgt_clean, tgt_notched, k=k)

    def _less_than(name, value, threshold, details):
        # All four 18.4 feature-space criteria are "lower is better".
        return DiagnosticResult(
            name=name, value=float(value), threshold=float(threshold),
            threshold_direction="less_than", passed=bool(value < threshold),
            details={"note": details})

    results = [
        knn,   # DiagnosticResult from 18.1 (knn_same_cohort_rate), accepts 2-D embeddings
        _less_than("pca_pc1_cohens_d", pc1_d, _THRESHOLD_PCA_D,
                   "feature-space analogue of 18.1 raw_pixel_pca_d (PC1 Cohen's d)"),
        _less_than("per_dimension_max_cohens_d", per_dim, _THRESHOLD_PER_BAND_COHENS_D,
                   "feature-space analogue of 18.1 per-band Cohen's d"),
        _less_than("notch_injection_migration_features", migration, _THRESHOLD_NOTCH_INJECTION,
                   f"k-NN migration of notched target features toward source (notch_frac={notch_frac})"),
    ]
    return results


def _write_report(out_dir: Path, results: list[DiagnosticResult], meta: dict) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_pass = all(r.passed for r in results)
    lines = [
        "# Module 18.4 — VAE falsifiable cage test on v2 encoder features",
        "",
        f"- Encoder checkpoint: `{meta['ckpt']}`",
        f"- Source (vocalmat) patches: {meta['n_source']} · Target (lab) patches: {meta['n_target']}",
        f"- k (k-NN): {meta['k']} · notch fraction: {meta['notch_frac']}",
        "",
        "| Test | Value | Threshold | Direction | Pass |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r.name} | {r.value:.4f} | {r.threshold} | {r.threshold_direction} | "
                     f"{'✅' if r.passed else '❌'} |")
    lines += [
        "",
        f"## Overall: {'✅ ALL 4 PASS — feature-level cage invariance confirmed' if all_pass else '❌ FAIL — at least one criterion failed'}",
        "",
        "This is the 18.4 VAE gate (stronger than 18.1's raw-spectrogram pass: it "
        "tests invariance of the *learned* features after adversarial training). "
        "Combined with the linear cage probe (<0.65) and the collapse tripwire "
        "(syllable F1 ≥ v1−0.05), all three must pass to ship v2.",
    ]
    (out_dir / "cage_invariance_probe.md").write_text("\n".join(lines))
    return all_pass


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Re-run the 18.1 VAE cage test on v2 encoder features (Module 18.4)")
    p.add_argument("--v2-checkpoint", required=True, type=Path)
    p.add_argument("--source-val-csv", required=True, type=Path, help="VocalMat val manifest (source)")
    p.add_argument("--domain-unlabeled-csv", required=True, type=Path, help="lab patches (target)")
    p.add_argument("--target-cohort", default="lab")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--num-classes", type=int, default=12)
    p.add_argument("--num-domains", type=int, default=2)
    p.add_argument("--max-per-cohort", type=int, default=1000,
                   help="Patches per cohort. MUST be >= 500: the per-dimension max "
                        "Cohen's d criterion takes a max over 512 feature dims, so its "
                        "0.30 threshold (calibrated for 18.1's ~10 bands) suffers "
                        "multiple-comparisons inflation at small n — expected max|d| "
                        "~0.34 at n=200 (false FAIL) vs ~0.15 at n=1000 (safe).")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--notch-frac", type=float, default=0.15,
                   help="fraction of the patch height (frequency axis) to zero as a notch")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    for flag, path in [("--v2-checkpoint", args.v2_checkpoint),
                       ("--source-val-csv", args.source_val_csv),
                       ("--domain-unlabeled-csv", args.domain_unlabeled_csv)]:
        if not path.exists():
            print(f"Error: {flag} does not exist: {path}", file=sys.stderr)
            return 1

    device = "cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    encoder = load_v2_encoder(args.v2_checkpoint, args.num_classes, args.num_domains, device)

    source_paths = _sample_paths(args.source_val_csv, cohort=None, max_n=args.max_per_cohort)
    target_paths = _sample_paths(args.domain_unlabeled_csv, cohort=args.target_cohort,
                                 max_n=args.max_per_cohort)
    # MAJOR-1 guard: the per-dimension max Cohen's d criterion takes a max over
    # 512 dims; its 0.30 threshold is only calibrated for large n. Below ~500
    # per cohort it false-FAILs on a genuinely invariant encoder.
    min_n = min(len(source_paths), len(target_paths))
    if min_n < 500:
        print(f"WARNING: only {min_n} patches in the smaller cohort (< 500). "
              "The 'per_dimension_max_cohens_d' criterion is unreliable at this n "
              "(multiple-comparisons inflation → likely false FAIL). Increase "
              "--max-per-cohort or ensure the val/target splits have >= 500 patches.",
              file=sys.stderr)

    source_roots = [REPO_ROOT, args.source_val_csv.resolve().parent,
                    args.source_val_csv.resolve().parent.parent]
    target_roots = [REPO_ROOT, args.domain_unlabeled_csv.resolve().parent,
                    args.domain_unlabeled_csv.resolve().parent.parent]

    results = run_diagnostic(encoder, source_paths, target_paths, source_roots, target_roots,
                             device=device, k=args.k, notch_frac=args.notch_frac)
    all_pass = _write_report(
        args.output_dir, results,
        meta={"ckpt": args.v2_checkpoint, "n_source": len(source_paths),
              "n_target": len(target_paths), "k": args.k, "notch_frac": args.notch_frac})
    for r in results:
        print(f"{r.name}: {r.value:.4f} ({'PASS' if r.passed else 'FAIL'})")
    print(f"VAE gate overall: {'ALL PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 3


if __name__ == "__main__":
    sys.exit(main())
