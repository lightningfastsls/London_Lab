"""Update split CSVs to point to training mode spectrograms."""

import pandas as pd
from pathlib import Path

for split_name in ['train', 'val', 'test']:
    df = pd.read_csv(f'splits/{split_name}.csv')

    # Update paths - replace review dirs with training dirs
    original_paths = df['spectrogram_path'].copy()

    df['spectrogram_path'] = df['spectrogram_path'].str.replace(
        'spectrograms_review', 'spectrograms_training', regex=False
    )

    # For noise samples specifically in review dir
    mask = df['spectrogram_path'].str.contains('noise_samples')
    df.loc[mask, 'spectrogram_path'] = df.loc[mask, 'spectrogram_path'].str.replace(
        r'noise_samples[/\\]', 'noise_samples_training/', regex=True
    )

    # Verify all exist
    missing_count = 0
    for idx, path_str in enumerate(df['spectrogram_path']):
        path = Path(path_str)
        if not path.exists():
            missing_count += 1
            if missing_count <= 10:
                print(f'{split_name}: Missing [{idx}] {path_str}')
                print(f'  Original: {original_paths.iloc[idx]}')

    # Save
    df.to_csv(f'splits/{split_name}.csv', index=False)

    print(f'\n{split_name}.csv: {len(df)} samples')
    print(f'  USV: {(df["label"] == "USV").sum()}')
    print(f'  Not USV: {(df["label"] == "Not USV").sum()}')
    print(f'  Missing: {missing_count}')

print('\nDone! Updated all splits to use training spectrograms.')
