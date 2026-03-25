#!/usr/bin/env bash
# Vault Search Validation Test Suite
# Phase 1.5: 7 known-answer tests + ground truth + negative tests
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEARCH="node $SCRIPT_DIR/vault-search.mjs"

PASS=0
FAIL=0
TOTAL=0

check() {
    local test_name="$1"
    local query="$2"
    local expected_substr="$3"
    local expected_map="$4"
    local limit="${5:-5}"

    ((TOTAL++)) || true
    local result
    result=$($SEARCH --query "$query" --limit "$limit" 2>/dev/null)

    # Check if expected note substring appears in results
    local note_found=false
    local map_found=false

    if echo "$result" | grep -qi "$expected_substr"; then
        note_found=true
    fi
    if echo "$result" | grep -qi "\"topic_map\":.*$expected_map"; then
        map_found=true
    fi

    if $note_found; then
        echo "  PASS: $test_name (note found)"
        ((PASS++)) || true
    else
        echo "  FAIL: $test_name (expected '$expected_substr' in top $limit)"
        ((FAIL++)) || true
    fi
}

check_negative() {
    local test_name="$1"
    local query="$2"
    local forbidden_substr="$3"
    local limit="${4:-5}"

    ((TOTAL++)) || true
    local result
    result=$($SEARCH --query "$query" --limit "$limit" 2>/dev/null)

    if echo "$result" | grep -qi "$forbidden_substr"; then
        echo "  FAIL: $test_name (forbidden '$forbidden_substr' appeared in top $limit)"
        ((FAIL++)) || true
    else
        echo "  PASS: $test_name (correctly excluded)"
        ((PASS++)) || true
    fi
}

echo "=== Vault Search Validation ==="
echo ""

# --- 7 Known-Answer Tests ---
echo "## Known-Answer Tests (note in top 5)"
check "DeepSqueak Bridge" \
    "DeepSqueak Classification Bridge" \
    "Raven selection table" \
    "classification-tools"

check "Energy Detector" \
    "energy detector threshold" \
    "negative 60 dB" \
    "detection"

check "VQ-VAE Codebook" \
    "VQ-VAE codebook" \
    "codebook" \
    "representation-learning"

check "STFT Parameters" \
    "STFT parameters 300 kHz" \
    "512-point FFT\|STFT" \
    "signal-processing"

check "LMT Integration" \
    "LMT behavioral integration" \
    "LMT\|behavioral" \
    "behavioral-integration"

check "Agent Memory" \
    "agent memory architecture" \
    "memory" \
    "agent-memory"

check "Hook Theory" \
    "hook theory external cognition" \
    "hook\|external cognition" \
    "agent-external-cognition"

echo ""

# --- Ground Truth: DeepSqueak (≥3 of 4 in top 8) ---
echo "## Ground Truth: DeepSqueak (≥3/4 in top 8)"
GT_PASS=0
GT_RESULT=$($SEARCH --query "DeepSqueak Classification Bridge" --context "Phase 2 Raven export Phase 3 MATLAB import" --limit 8 2>/dev/null)

gt_check() {
    local label="$1"
    local substr="$2"
    if echo "$GT_RESULT" | grep -qi "$substr"; then
        echo "  FOUND: $label"
        ((GT_PASS++)) || true
    else
        echo "  MISS:  $label"
    fi
}

gt_check "timestamp proximity matching" "timestamp proximity matching"
gt_check "Raven selection table format" "Raven selection table"
gt_check "DeepSqueak built-in classification" "DeepSqueak built-in classification"
gt_check "Reading DeepSqueak mat outputs" "Reading DeepSqueak mat"

((TOTAL++)) || true
if [[ $GT_PASS -ge 3 ]]; then
    echo "  PASS: Ground truth $GT_PASS/4 (≥3 required)"
    ((PASS++)) || true
else
    echo "  FAIL: Ground truth $GT_PASS/4 (≥3 required)"
    ((FAIL++)) || true
fi
echo ""

# --- Negative Tests (irrelevant notes should NOT appear) ---
echo "## Negative Tests (should not appear in top 5)"
check_negative "DeepSqueak no agent-cognition" \
    "DeepSqueak Classification Bridge" \
    "agent habit system\|basal ganglia\|nudge theory"

check_negative "DeepSqueak no RL-alignment" \
    "DeepSqueak Classification Bridge" \
    "RLHF pipeline\|reward hacking\|policy optimization"

check_negative "Energy no VQ-VAE" \
    "energy detector threshold" \
    "VQ-VAE codebook\|codebook size"

check_negative "Agent memory no signal-processing" \
    "agent memory architecture" \
    "512-point FFT\|300 kHz sample rate\|STFT"

echo ""

# --- Performance ---
echo "## Performance"
START=$(date +%s%N)
$SEARCH --query "DeepSqueak Classification Bridge" --limit 5 > /dev/null 2>&1
END=$(date +%s%N)
ELAPSED_MS=$(( (END - START) / 1000000 ))
echo "  Single query: ${ELAPSED_MS}ms (target: <1000ms)"
((TOTAL++)) || true
if [[ $ELAPSED_MS -lt 1000 ]]; then
    echo "  PASS: Performance"
    ((PASS++)) || true
else
    echo "  FAIL: Performance (${ELAPSED_MS}ms > 1000ms)"
    ((FAIL++)) || true
fi
echo ""

# --- Summary ---
echo "=== Results: $PASS/$TOTAL passed, $FAIL failed ==="
if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
