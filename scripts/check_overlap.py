"""Check for overlap between USV and noise candidates."""
import pandas as pd

usv = pd.read_csv('candidates_final.csv')
noise_old = pd.read_csv('noise_samples/noise_samples.csv')
noise_new = pd.read_csv('noise_samples/noise_samples_final.csv')

# Combine and deduplicate noise
noise_all = pd.concat([noise_old, noise_new]).drop_duplicates(subset=['candidate_id'])

# Check for overlap
usv_ids = set(usv['candidate_id'])
noise_ids = set(noise_all['candidate_id'])
overlap = usv_ids & noise_ids

print(f'USV candidates: {len(usv)}')
print(f'Noise candidates (dedup): {len(noise_all)}')
print(f'Overlap between USV and noise: {len(overlap)}')

if len(overlap) > 0:
    print(f'\nFirst 10 overlapping IDs:')
    for cid in list(overlap)[:10]:
        print(f'  {cid}')
