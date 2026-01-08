import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime

# --- CONFIGURATION ---
BASE_PATH = r"C:\Users\shach\OneDrive\שולחן העבודה\Miki London Lab\Mice learning analysis\mouse_2\mouse_2"
# Ensure this points to your CLEANED csv
WATER_LOG_PATH = r"C:\Users\shach\PycharmProjects\mickey_london_lab\mice_water_consumption_cleaned.csv"

# --- MOUSE MAPPING ---
MOUSE_WATER_MAP = {
    'mouse_1': 'P1R',
    'mouse_2': 'P1L',
    'mouse_3': 'PNO'
}


def get_file_date_str(filepath):
    """
    Extracts the 'Modified' timestamp from a file.
    Returns: String 'YYYY-MM-DD'
    """
    try:
        timestamp = os.path.getmtime(filepath)
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error reading date for {filepath}: {e}")
        return None


def create_water_lookup(path, target_mouse_col):
    """
    Reads the CLEANED CSV and creates a dictionary mapping.
    Since the file is now clean (DD-MM-YYYY), we parse it directly.
    """
    water_map = {}

    if not os.path.exists(path):
        print(f"Water log file not found at: {path}")
        return water_map

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return water_map

    # Clean Column Names
    df.columns = [str(c).strip() for c in df.columns]

    # Find Date Column (Looking for 'formatted_date' or similar)
    date_col = None
    if 'formatted_date' in df.columns:
        date_col = 'formatted_date'
    elif 'date' in df.columns:
        date_col = 'date'
    else:
        # Fallback search
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break

    if not date_col or target_mouse_col not in df.columns:
        print(f"WARNING: Missing 'formatted_date' or '{target_mouse_col}' column.")
        return water_map

    print(f"Building Water Lookup for '{target_mouse_col}' using date column '{date_col}'...")

    # CRITICAL FIX: Parse the dates using dayfirst=True because we know the format is DD-MM-YYYY
    # We convert to a standard YYYY-MM-DD string to match the file timestamps
    df['std_date'] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')

    # Drop rows where date parsing failed
    df = df.dropna(subset=['std_date'])

    # Create the map: {'2025-09-07': 1500, ...}
    water_map = dict(zip(df['std_date'], df[target_mouse_col]))

    return water_map


def analyze_trial_outcome(phase, t_start, t_mid, t_end, licks, water, punishment_time):
    """
    Determines outcome based on specific Phase rules.
    """
    window_dur = 3.0
    if phase == 4:
        window_dur = 5.0
    elif phase == 2:
        window_dur = 5.0

    window_end = t_mid + window_dur

    # 1. Explicit Punishment
    if punishment_time is not None:
        return "False Alarm", np.nan

    # 2. Water Received (Success)
    if len(water) > 0:
        post_sound_licks = licks[licks > t_mid]
        if len(post_sound_licks) > 0:
            reaction_time = post_sound_licks[0] - t_mid
            return "Hit", reaction_time
        return "Hit", 0.0

    # 3. Calculated Outcome
    early_licks = licks[(licks >= t_start) & (licks <= t_mid)]
    if len(early_licks) > 0:
        if phase >= 3:
            return "False Alarm", np.nan

    post_sound_licks = licks[licks > t_mid]
    if len(post_sound_licks) > 0:
        first_lick_time = post_sound_licks[0]
        rt = first_lick_time - t_mid
        if first_lick_time <= window_end:
            return "Hit (Calc)", rt

    return "Miss", np.nan


def process_all_data(base_path, water_lookup_map):
    all_trials = []
    phases = ['phase_1_training', 'phase_2_training', 'phase_3_training', 'phase_4_training']
    current_mouse_id = os.path.basename(os.path.normpath(base_path))

    for phase_name in phases:
        phase_dir = os.path.join(base_path, phase_name)
        if not os.path.exists(phase_dir):
            continue

        try:
            phase_num = int(phase_name.split('_')[1])
        except:
            phase_num = 1

        session_folders = glob.glob(os.path.join(phase_dir, "trial_*"))

        for session_path in session_folders:
            session_id = os.path.basename(session_path).split('_')[-1]

            def load_npy(name):
                pattern = os.path.join(session_path, f"*{name}*trial_{session_id}*.npy")
                files = glob.glob(pattern)
                if files:
                    return np.load(files[0]), files[0]
                return np.array([]), None

            cycles, cycle_path = load_npy("cycles")
            if len(cycles) == 0: continue

            date_key = get_file_date_str(cycle_path)

            # LOOKUP WATER
            water_val = water_lookup_map.get(date_key, np.nan)

            licks, _ = load_npy("licking")
            waters, _ = load_npy("water")
            rests, _ = load_npy("rest")
            punish_10s, _ = load_npy("ten_s_delay")

            for i in range(len(cycles)):
                t_start = cycles[i][0]
                t_mid = cycles[i][1]

                t_end = t_mid + 15
                if phase_num < 3:
                    if i < len(rests) and rests.size > 0:
                        t_end = rests[i]
                if i + 1 < len(cycles):
                    t_end = cycles[i + 1][0]

                t_licks = licks[(licks >= t_start) & (licks < t_end + 5)]
                t_water = waters[(waters >= t_start) & (waters < t_end)]

                punishment_time = None
                if len(punish_10s) > 0:
                    p_check = punish_10s[(punish_10s >= t_start) & (punish_10s < t_end)]
                    if len(p_check) > 0:
                        punishment_time = p_check[0]

                outcome, rt = analyze_trial_outcome(phase_num, t_start, t_mid, t_end, t_licks, t_water, punishment_time)

                all_trials.append({
                    'Mouse_ID': current_mouse_id,
                    'Date': date_key,
                    'Water_Received_ml': water_val,
                    'Phase': phase_num,
                    'Session_ID': int(session_id),
                    'Trial_ID': i + 1,
                    'Trial_Start': t_start,
                    'Outcome': outcome,
                    'Reaction_Time': rt,
                    'Is_Punished': 1 if punishment_time else 0,
                    'Total_Licks': len(t_licks)
                })

    return pd.DataFrame(all_trials)


# --- EXECUTION ---

current_mouse = os.path.basename(os.path.normpath(BASE_PATH)).lower()
target_col = MOUSE_WATER_MAP.get(current_mouse)

if not target_col:
    print(f"WARNING: No mapping for '{current_mouse}'. Defaulting to P1R")
    target_col = 'P1R'

# 1. Load Water Map
water_lookup = create_water_lookup(WATER_LOG_PATH, target_col)

# 2. Process Trial Data
master_df = process_all_data(BASE_PATH, water_lookup)

# 3. Add Missing Dates (Water but No Trial)
print("Checking for missing dates (Water log vs Trials)...")

# Get sets of dates
existing_dates = set(master_df['Date'].unique()) if not master_df.empty else set()
all_water_dates = set(water_lookup.keys())

# Find dates that are in water log but NOT in the trial data
dates_no_trial = all_water_dates - existing_dates

if dates_no_trial:
    print(f"Found {len(dates_no_trial)} days with water but no trials. Adding them...")

    non_trial_rows = []
    for d_str in dates_no_trial:
        non_trial_rows.append({
            'Mouse_ID': current_mouse,
            'Date': d_str,
            'Water_Received_ml': water_lookup[d_str],
            'Phase': 0,  # <--- Marked as 0 per your request
            'Session_ID': -1,  # Placeholder
            'Trial_ID': -1,  # Placeholder
            'Outcome': 'No Trial',
            'Reaction_Time': np.nan,
            'Is_Punished': 0,
            'Total_Licks': 0
        })

    # Combine
    missing_df = pd.DataFrame(non_trial_rows)
    master_df = pd.concat([master_df, missing_df], ignore_index=True)
else:
    print("All dates in water log have corresponding trials.")

# 4. Final Sort and Save
if not master_df.empty:
    print("Sorting master dataframe...")
    # Sort by Date. We fill NaN/Negative Session IDs to ensure they don't break sorting
    master_df.sort_values(by=['Date', 'Session_ID', 'Trial_ID'], ascending=True, inplace=True)

    if 'Date' in master_df.columns:
        master_df['Day_Name'] = pd.to_datetime(master_df['Date']).dt.day_name()

    output_filename = f"{current_mouse}_master_analysis.csv"
    master_df.to_csv(output_filename, index=False)
    print(f"\nSuccess! Saved sorted analysis to: {output_filename}")

    # Optional: Check for gaps
    print(f"Total rows: {len(master_df)}")
    print(f"Date range: {master_df['Date'].min()} to {master_df['Date'].max()}")
else:
    print("No data processed.")