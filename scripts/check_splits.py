"""Check for duplicates in splits."""
import pandas as pd

train = pd.read_csv('splits/train.csv')
val = pd.read_csv('splits/val.csv')
test = pd.read_csv('splits/test.csv')
all_split = pd.concat([train, val, test])

print(f'Total rows in splits: {len(all_split)}')
print(f'Unique candidate_ids: {all_split["candidate_id"].nunique()}')

# Check for duplicates
dups = all_split[all_split.duplicated(subset=['candidate_id'], keep=False)]
if len(dups) > 0:
    print(f'\nDuplicates found: {len(dups)} rows')
    print('\nFirst 10 duplicate candidate_ids:')
    for cid in dups['candidate_id'].unique()[:10]:
        dup_rows = all_split[all_split['candidate_id'] == cid]
        print(f'\n{cid}:')
        print(dup_rows)
else:
    print('\nNo duplicates found')
