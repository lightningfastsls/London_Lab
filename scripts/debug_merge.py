"""Debug NaN values in merge."""
import pandas as pd
from pathlib import Path

# Load data
labels = pd.read_csv('labels_combined.csv')
usv = pd.read_csv('candidates_final.csv')
noise = pd.read_csv('noise_samples/noise_samples_final.csv')

print(f"Labels: {len(labels)} rows")
print(f"USV candidates: {len(usv)} rows")
print(f"Noise candidates: {len(noise)} rows")
print()

# Combine candidates
all_cands = pd.concat([
    usv[['candidate_id', 'source_file']],
    noise[['candidate_id', 'source_file']]
], ignore_index=True)

print(f"Combined candidates: {len(all_cands)} rows")
print(f"Unique candidate_ids in combined: {all_cands['candidate_id'].nunique()}")
print()

# Merge
merged = labels.merge(all_cands, on='candidate_id', how='left')

print(f"Merged rows: {len(merged)}")
print(f"NaN source_file: {merged['source_file'].isna().sum()}")
print()

if merged['source_file'].isna().sum() > 0:
    print("Sample missing candidates:")
    missing = merged[merged['source_file'].isna()]['candidate_id'].head(10)
    for cid in missing:
        print(f"  {cid}")
    print()

    # Check if these are in labels but not in either candidates file
    missing_ids = merged[merged['source_file'].isna()]['candidate_id'].unique()
    print(f"Total missing candidate_ids: {len(missing_ids)}")

    # Check sources
    missing_labels = labels[labels['candidate_id'].isin(missing_ids)]
    print("\nSource distribution of missing candidates:")
    print(missing_labels['source'].value_counts())
