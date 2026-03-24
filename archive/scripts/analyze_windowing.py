"""Analyze how sliding windows cover the spectrogram."""
import numpy as np

# Simulate sliding window parameters
window_width_px = 100
hop_px = 10
n_times = 1000  # Example spectrogram width

# Generate window centers (same logic as sliding_inference.py)
window_centers = []
col = window_width_px // 2  # = 50
while col + window_width_px // 2 < n_times:
    window_centers.append(col)
    col += hop_px

window_centers = np.array(window_centers)

print("="*60)
print("SLIDING WINDOW ANALYSIS")
print("="*60)
print(f"\nParameters:")
print(f"  Window width: {window_width_px} columns")
print(f"  Hop size: {hop_px} columns")
print(f"  Spectrogram width: {n_times} columns")
print(f"\nWindow generation:")
print(f"  First window center: {window_centers[0]}")
print(f"  Last window center: {window_centers[-1]}")
print(f"  Total windows: {len(window_centers)}")
print(f"  Windows span columns: [{window_centers[0] - window_width_px//2}, {window_centers[-1] + window_width_px//2})")

print(f"\n" + "="*60)
print("COVERAGE ANALYSIS")
print("="*60)

# Analyze how many windows cover each column
coverage = np.zeros(n_times, dtype=int)
for center in window_centers:
    start_col = center - window_width_px // 2
    end_col = start_col + window_width_px
    coverage[start_col:end_col] += 1

print(f"\nHow many windows see each column:")
print(f"  Min coverage: {coverage.min()} windows")
print(f"  Max coverage: {coverage.max()} windows")
print(f"  Typical coverage: {coverage[50:-50].mean():.1f} windows")
print(f"\nColumns with 0 coverage: {(coverage == 0).sum()}")
print(f"  These are at the edges: columns [0-{window_centers[0] - window_width_px//2}) and [{window_centers[-1] + window_width_px//2}-{n_times})")

print(f"\n" + "="*60)
print("PROBABILITY ASSIGNMENT")
print("="*60)

print(f"\nCRITICAL ISSUE:")
print(f"  - Each COLUMN is seen by ~{coverage[50:-50].mean():.0f} different windows")
print(f"  - But we only get ONE probability PER WINDOW (at its center)")
print(f"  - We do NOT average probabilities across overlapping windows")
print(f"\nExample for column 100:")
print(f"  Windows that include column 100:")
windows_covering_100 = []
for center in window_centers:
    start_col = center - window_width_px // 2
    end_col = start_col + window_width_px
    if start_col <= 100 < end_col:
        windows_covering_100.append(center)

for i, center in enumerate(windows_covering_100[:5]):
    print(f"    Window {i+1}: center={center}, spans [{center-50}, {center+50})")

print(f"  ... {len(windows_covering_100)} windows total")
print(f"\n  BUT: We only plot ONE probability point at column 100")
print(f"       (from the window centered at 100)")

print(f"\n" + "="*60)
print("VISUALIZATION OF FIRST 200 COLUMNS")
print("="*60)

# Show window coverage for first 200 columns
print(f"\nColumn | Coverage | Window centers that include this column")
print("-" * 70)
for col in [0, 10, 20, 50, 100, 150, 200]:
    if col >= n_times:
        break
    windows = [c for c in window_centers if c - window_width_px//2 <= col < c + window_width_px//2]
    if len(windows) <= 5:
        centers_str = str(windows)
    else:
        centers_str = f"{windows[:3]} ... {windows[-2:]} ({len(windows)} total)"
    print(f"{col:6d} | {coverage[col]:8d} | {centers_str}")

print(f"\n" + "="*60)
print("IMPLICATIONS FOR INFERENCE")
print("="*60)

print(f"""
The current implementation:
1. Extracts {len(window_centers)} windows of width {window_width_px}
2. Gets ONE probability per window (at window center)
3. Plots probabilities at window center positions

This means:
- Most columns are analyzed by ~{coverage[50:-50].mean():.0f} overlapping windows
- But we only report the probability from ONE window per position
- We're not averaging/aggregating across overlapping windows

This is CORRECT if:
- The CNN learns to predict "is there a USV at the CENTER of this window?"
- We only care about probabilities at discrete positions (every {hop_px} columns)

This might be WRONG if:
- Different windows give different probabilities for the same region
- We want a per-pixel probability (should average overlapping windows)
""")
