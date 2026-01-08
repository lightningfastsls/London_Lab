import os
import numpy as np
import pandas as pd


def load_all_files(folder_path):
    """
    Loads all files inside a given folder.

    Supported types:
        - .npy  (loaded with numpy.load)
        - .csv  (loaded with pandas.read_csv)
        - .txt  (read as text)

    Returns:
        A dictionary: {filename: data}
    """

    files_data = {}

    # List all files in folder
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)

        # Ignore subfolders
        if os.path.isdir(full_path):
            continue

        # Handle file types
        if filename.endswith(".npy"):
            files_data[filename] = np.load(full_path, allow_pickle=True)

        elif filename.endswith(".csv"):
            files_data[filename] = pd.read_csv(full_path)

        elif filename.endswith(".txt"):
            with open(full_path, "r") as f:
                files_data[filename] = f.read()

        else:
            print(f"Skipping unsupported file: {filename}")

    return files_data


# ---------------------
# Example usage:
# ---------------------
path = r"C:\Users\shach\OneDrive\שולחן העבודה\Miki London Lab\Mice learning analysis\mouse_1\mouse_1\phase_1_training\trial_1"

data = load_all_files(path)

# Print loaded filenames
for k in data:
    print("Loaded:", k)

x=1