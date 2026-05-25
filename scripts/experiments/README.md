# scripts/experiments/ — rig contour-VAE / shape-representation experiments

These scripts were developed on the GPU rig (`shachar@100.113.224.57`,
`/data/shachar/contour_vae/`) during the 2026-05 shape-representation work and
promoted into `main` for provenance during the 2026-05-25 post-merge
reconciliation. They are **experiment drivers**, not production-pipeline code —
they expect rig-local data (patches / ridges / latents under
`/data/shachar/contour_vae/...`; see `docs/DATA_LOCATIONS.md`) and the rig's
canonical root. Run them on the rig, not the box.

| Script | Role | Status |
|---|---|---|
| `rig_R1_true_ridges.py` | Build the byte-faithful registered-ridge set (`true_registered_ridges*.npz`) — the foundation for the shape alphabet | Keeper (data foundation) |
| `rig_R2_shape_alphabet.py` | Productionized registered-shape K=20 clustering (`models/shape_kmeans/k20.joblib`); the registration→shape η²=0.75 result | Keeper (production path) |
| `rig_M8_contour_vae.py` | Bake-off: contour-masked VAE architecture | Archival (bake-off) |
| `rig_M9_contrastive.py` | Bake-off: contrastive representation | Archival (bake-off) |
| `rig_M10_image_vae.py` | Bake-off: image-VAE architecture | Archival (bake-off) |

The M8/M9/M10 bake-off verdict ("Outcome A": near-disjoint cohort separation
driven by a cage confound, not biology) is recorded in the knowledge graph;
these three are kept for reproducibility, not active use. See
`docs/handoffs/2026-05-18_vae_comparison_memo.md` and
`PLAN_shape_representation_v2.md`.
