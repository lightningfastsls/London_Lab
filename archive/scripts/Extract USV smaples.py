#python file to extract USVs from wav files and save snipest into a training forlder to save a AI training
import os
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
from glob import glob

# Parameters
input_folder = '/Users/mikilon/Data/WAVfiles'
output_folder = '/Users/mikilon/Data/USVtrain'
high_pass_freq = 20000  # High-pass filter frequency threshold (20kHz)
segment_duration = 0.3  # Duration of each extracted segment in seconds

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def high_pass_filter(data, sr, cutoff=high_pass_freq, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def detect_and_extract_segments(file_path):
    # Read the audio file
    data, sr = sf.read(file_path)
    filtered_data = high_pass_filter(data, sr)
    
    # Detect high frequency components
    energy = np.abs(filtered_data)
    threshold = np.mean(energy) * 2  # Basic thresholding approach
    high_freq_indices = np.where(energy > threshold)[0]
    
    if len(high_freq_indices) == 0:
        return  # No high frequency components detected
    
    # Split into segments and save
    segment_samples = int(segment_duration * sr)
    for start_idx in high_freq_indices[::segment_samples]:
        end_idx = start_idx + segment_samples
        if end_idx > len(data):
            break  # Avoid going beyond the file length
        segment_data = data[start_idx:end_idx]
        output_file_path = os.path.join(output_folder, f"{os.path.basename(file_path).split('.')[0]}_{start_idx}.wav")
        sf.write(output_file_path, segment_data, sr)


"""
A function that checks the input folder for a file that contained a list of wav files that were previously processed. 
the function then need to read the file.
Then it has to loop through all the wav files in the input folder and processes each file using the detect_and_extract_segments function but
only if the file is not in the list. 
Then add the file that was processed to the list and save the list. Then repeat for the next wav file.

"""
def main():
    # Check for a file that contains a list of processed files
    processed_files_list_path = os.path.join(input_folder, 'processed_files.txt')
    processed_files = []
    if os.path.exists(processed_files_list_path):
        with open(processed_files_list_path, 'r') as f:
            processed_files = f.read().splitlines()
    
    # Loop through all the wav files in the input folder
    for file_path in glob(os.path.join(input_folder, '*.wav')):
        if file_path in processed_files:
            print(f"Skipping {file_path} as it was already processed")
            continue  # Skip this file
        detect_and_extract_segments(file_path)
        processed_files.append(file_path)
    
    # Save the list of processed files
    with open(processed_files_list_path, 'w') as f:
        f.write('\n'.join(processed_files))


main()