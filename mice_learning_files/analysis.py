import pandas as pd
import numpy as np
from scipy.stats import norm
import os


# --- 1. DEFINE THE HELPER FUNCTIONS ---
def calculate_sdt(df_subset):
    """
    Takes a chunk of trials (a session or a bin) and returns ONE row of metrics.
    """
    # Filter for hits/misses/false alarms
    hits = len(df_subset[df_subset['Outcome'].str.startswith('Hit', na=False)])
    misses = len(df_subset[df_subset['Outcome'] == 'Miss'])
    fas = len(df_subset[df_subset['Outcome'] == 'False Alarm'])

    # Define N_Go and N_NoGo
    n_go = hits + misses
    # Assuming roughly equal Go/No-Go trials if explicit CRs aren't logged
    n_nogo = fas + (n_go - fas if n_go > fas else 0)
    if n_nogo == 0: n_nogo = n_go  # Fallback to prevent division by zero

    # Log-Linear Correction (prevents infinite d' values)
    hit_rate = (hits + 0.5) / (n_go + 1)
    fa_rate = (fas + 0.5) / (n_nogo + 1)

    # Calculate SDT Metrics
    z_hit = norm.ppf(hit_rate)
    z_fa = norm.ppf(fa_rate)

    d_prime = z_hit - z_fa
    criterion = -0.5 * (z_hit + z_fa)

    # Efficiency & RT
    total_licks = df_subset['Total_Licks'].sum()
    efficiency = total_licks / hits if hits > 0 else np.nan
    rt_median = df_subset.loc[df_subset['Outcome'].str.startswith('Hit', na=False), 'Reaction_Time'].median()

    return pd.Series({
        'd_prime': d_prime,
        'criterion': criterion,
        'hit_rate': hit_rate,
        'fa_rate': fa_rate,
        'median_rt': rt_median,
        'efficiency': efficiency,
        'total_trials': len(df_subset),
        'total_hits': hits
    })


def analyze_learning(master_df, bin_size=20):
    print("Starting Learning Analysis...")

    # --- LEVEL 1: MACRO (One row per Session) ---
    print("  Aggregating Daily Performance...")
    # Filter out placeholder "No Trial" rows
    valid_trials = master_df[master_df['Session_ID'] != -1].copy()

    daily_stats = valid_trials.groupby(['Date', 'Session_ID', 'Phase']).apply(calculate_sdt).reset_index()
    daily_stats.sort_values('Date', inplace=True)

    # --- LEVEL 2: MICRO (Within-Session Bins) ---
    print(f"  Aggregating Within-Session Performance (Bin Size = {bin_size})...")
    valid_trials['Bin_ID'] = valid_trials.groupby('Session_ID').cumcount() // bin_size
    binned_stats = valid_trials.groupby(['Date', 'Session_ID', 'Phase', 'Bin_ID']).apply(calculate_sdt).reset_index()

    return daily_stats, binned_stats


# --- 2. EXECUTION ---
# Configuration
base_path = r"D:\mickey_london_lab"
current_mouse = "mouse_2"  # Replace with logic if looping through mice

# Construct path safely
file_path = os.path.join(base_path, f"{current_mouse}_master_analysis.csv")

if os.path.exists(file_path):
    print(f"Loading data from: {file_path}")
    master_df = pd.read_csv(file_path)

    if not master_df.empty:
        daily_df, binned_df = analyze_learning(master_df)

        print("\n--- Daily Summary (First 5 Rows) ---")
        print(daily_df[['Date', 'Phase', 'd_prime', 'hit_rate', 'median_rt']].head())

        # Save results
        daily_output = os.path.join(base_path, f"{current_mouse}_daily_learning.csv")
        binned_output = os.path.join(base_path, f"{current_mouse}_binned_learning.csv")

        daily_df.to_csv(daily_output, index=False)
        binned_df.to_csv(binned_output, index=False)
        print(f"\nAnalysis saved to:\n  {daily_output}\n  {binned_output}")
    else:
        print("Error: The CSV file is empty.")
else:
    print(f"Error: Could not find file at {file_path}")