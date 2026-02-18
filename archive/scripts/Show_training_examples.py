import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

# Load the training records CSV file
Training_folder = '/Users/mikilon/Data/USVtraining/'
records_df = pd.read_csv(Training_folder + "training_records.csv")

# Define the number of records per figure
records_per_figure = 10

# Iterate through the records, creating figures as needed
for i in range(0, len(records_df), records_per_figure):
    fig, axs = plt.subplots(5, 2, figsize=(15, 15))
    for j, ax in enumerate(axs.flat):
        record_index = i + j
        if record_index < len(records_df):
            record = records_df.iloc[record_index]
            file_path = Training_folder + record['filename']
            class_index = record['class_index']
            start_time = record['start_time']
            end_time = record['end_time']

            # Load the audio file
            y, sr = librosa.load(file_path, sr=None, offset=start_time, duration=end_time-start_time)

            # Compute the spectrogram
            D = librosa.stft(y, n_fft=512, hop_length=256)
            spectrogram_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            # Display the spectrogram
            ax.imshow(spectrogram_db, cmap='inferno', origin='lower', aspect='auto')
            ax.set_title(f"Record: {file_path}\nClass: {class_index}")
            ax.axis('off')
    plt.tight_layout()
    plt.show()