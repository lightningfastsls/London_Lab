"""Tests for the shared FPCA↔detections merge helper (WS-D/WS-E Phase 0).

These lock the inherited gotchas from
``docs/handoffs/2026-06-04_ws-de-archetypes-confound-comparison.md`` §1:
the −1 det_index offset, the largest-|amp_pc1| dedupe, and the per-cohort
biology-safe metadata attach. Spec-level: do not weaken expectations to pass.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.experiments import _fpca_merge as fm

# Skip the whole module if the canonical artifacts are absent (CI without data).
pytestmark = pytest.mark.skipif(
    not fm.FPCA_SCORES_PATH.exists(),
    reason="elastic_fpca_scores.parquet not present",
)


def test_dedupe_keeps_largest_abs_amp_pc1():
    df = pd.DataFrame(
        {
            "wav_stem": ["a", "a", "b"],
            "call_id": [1, 1, 1],
            "cohort": ["x", "x", "x"],
            "amp_pc1": [2.0, -5.0, 1.0],
            "det_index": [0, 0, 0],
        }
    )
    out = fm.dedupe_fpca(df)
    assert len(out) == 2
    # (a,1) row kept must be the |amp_pc1|=5 one
    kept = out[(out.wav_stem == "a") & (out.call_id == 1)]
    assert kept["amp_pc1"].iloc[0] == -5.0


def test_det_index_is_call_id_minus_one():
    df = fm.load_fpca_scores(dedupe=False)
    assert (df["det_index"] == df["call_id"].astype("int64") - 1).all()


def test_dedupe_produces_one_row_per_call():
    df = fm.load_fpca_scores(dedupe=True)
    assert not df.duplicated(["wav_stem", "call_id"]).any()
    # 5970 known figure from handoff §1: 7,064 unique calls.
    n5970 = (df["cohort"] == "5970").sum()
    assert n5970 == 7064, f"expected 7064 unique 5970 calls, got {n5970}"


def test_merge_attaches_pitch_and_duration():
    df = fm.load_merged_fpca(cohorts=["5970"], require_meta=True)
    for col in (fm.PITCH_COL, fm.DURATION_COL, fm.LABEL_COL):
        assert col in df.columns
    # pitch is a positive frequency in Hz; duration positive seconds.
    assert (df[fm.PITCH_COL] > 0).all()
    assert (df[fm.DURATION_COL] > 0).all()


def test_merge_is_many_to_one_safe():
    # validate='m:1' inside load_merged_fpca would raise if the join fanned out;
    # reaching here means the (wav_stem, det_index) key is unique on the meta side.
    df = fm.load_merged_fpca(cohorts=["5970"])
    # No row duplication relative to deduped FPCA input.
    fp = fm.load_fpca_scores(dedupe=True)
    assert len(df) == (fp["cohort"] == "5970").sum()
