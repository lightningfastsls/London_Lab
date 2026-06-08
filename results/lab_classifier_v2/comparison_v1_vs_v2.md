# Module 18.4 — v1 vs v2 (DANN) comparison

| Metric | v1 (18.3) | v2 (DANN) | Gate |
|---|---|---|---|
| Syllable macro-F1 (test) | 0.7668754899942165 | 0.6359 | ≥ v1−0.05 → FAIL (collapse) |
| Syllable macro-F1 (val) | 0.7692520159010385 | 0.6649 | — |
| Linear cage probe acc | n/a | 1.0000 | < 0.65 → FAIL |

- **F1 drop v1→v2:** 0.1310
- **Collapse tripwire (drop > 0.05):** TRIGGERED — STOP
- **Cage gate (probe < 0.65):** FAIL — encoder still cage-decodable

## Verdict
**DO NOT SHIP** — encoder collapsed (F1 drop > 0.05). cage probe ≥ 0.65 (cage still decodable). Surface λ-schedule alternatives per ROADMAP §18.4 exit criteria.

_Note: the VAE falsifiable re-run on encoder features (`run_vae_diagnostic_on_encoder.py` → `cage_invariance_probe.md`) is the third gate and must also pass; it is computed separately._