from loader_mice_files import load_all_files
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import os
from datetime import datetime

# ---------------------
# Example usage:
path = r"C:\Users\shach\OneDrive\שולחן העבודה\Miki London Lab\Mice learning analysis\mouse_1\mouse_1\phase_2_training\trial_1"
data = load_all_files(path)




def create_run_folder(base_name="Analysis_Run"):
    """
    Creates a new folder with a timestamp to store outputs from this specific run.
    Returns the path to the created folder.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{base_name}_{timestamp}"

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    print(f"Created output folder: {folder_name}")
    return folder_name
def get_trial_outcome(phase, t_start, t_mid, t_end, trial_licks, trial_water, trial_punishment_time=None):
    """
    Determines the outcome of a single trial based on the phase rules.
    Returns: status_string, color, window_duration
    """
    # 1. Determine Window Duration based on Phase
    if phase == 2:
        window_dur = 5.0
    else:
        window_dur = 3.0  # Default for Phase 1, 3, 4

    window_start = t_mid
    window_end = t_mid + window_dur

    # 2. Check for Water (Ultimate success indicator)
    # If water was dispensed during this trial, it is a success.
    if len(trial_water) > 0:
        return "Success", "palegreen", window_dur

    # 3. Check for Punishment (Phase 3+ explicit marker)
    # If a punishment timestamp exists in this trial boundaries
    if trial_punishment_time is not None:
        return "Punished", "salmon", window_dur

    # 4. Lick Analysis (If no explicit Water/Punishment markers found)
    # Filter licks that happened strictly after the sound (t_mid)
    post_sound_licks = trial_licks[trial_licks > t_mid]

    if len(post_sound_licks) > 0:
        first_lick = post_sound_licks[0]
        if first_lick <= window_end:
            # Lick was inside the window (and water wasn't logged? suspicious but count as hit logic)
            return "Success (Calc)", "palegreen", window_dur
        else:
            # Lick was after the window closed
            return "Miss/Late", "lightyellow", window_dur

    # Check for early licks (before sound)
    pre_sound_licks = trial_licks[(trial_licks >= t_start) & (trial_licks <= t_mid)]
    if len(pre_sound_licks) > 0:
        # In Phase 3+, this causes punishment. In Phase 1, it's just an early lick.
        if phase >= 3:
            return "Early Lick (Fail)", "salmon", window_dur
        else:
            return "Early Lick", "peachpuff", window_dur

    return "No Response", "whitesmoke", window_dur


def plot_mouse_session_phased(data_dict, session_num, phase=1, trials_per_row=10, max_rows_per_fig=4):
    """
    Plots the session timeline, splitting into multiple image files if too long.

    Args:
        trials_per_row: How many trials to show in one horizontal row (subplot).
        max_rows_per_fig: How many rows to stack in one image file before starting a new file.
    """

    # --- 1. Load Data with Fallbacks for different file naming conventions ---
    # Helper to find key efficiently
    def get_data(var_name):
        # Try specific phase 3 names first if relevant, else standard
        key_std = f'mouse1_{var_name}_time_trial_{session_num}.npy'
        if key_std in data_dict: return data_dict[key_std]
        # Fallback for scalar ending which might lack .npy in some dicts
        key_no_ext = f'mouse1_{var_name}_time_trial_{session_num}'
        if key_no_ext in data_dict: return data_dict[key_no_ext]
        return np.array([])  # Return empty array if missing

    cycles = get_data('cycles')
    if len(cycles) == 0:
        print(f"Skipping Session {session_num}: 'cycles' data not found.")
        return

    licks = get_data('licking')
    waters = get_data('water')
    rests = get_data('rest')
    ten_s_delays = get_data('ten_s_delay')  # Punishments

    num_trials = len(cycles)

    # Calculate total structure
    total_rows_needed = int(np.ceil(num_trials / trials_per_row))
    total_images_needed = int(np.ceil(total_rows_needed / max_rows_per_fig))

    current_trial_idx = 0

    # --- OUTER LOOP: Create one Image File per iteration ---
    for img_idx in range(total_images_needed):

        # How many rows go into THIS file?
        rows_remaining = total_rows_needed - (img_idx * max_rows_per_fig)
        rows_in_this_img = min(max_rows_per_fig, rows_remaining)

        # Create Figure for this chunk
        fig, axes = plt.subplots(rows_in_this_img, 1, figsize=(15, 5 * rows_in_this_img), constrained_layout=True)
        if rows_in_this_img == 1: axes = [axes]

        fig.suptitle(f'Mouse 1 - Phase {phase} - Session {session_num} (Part {img_idx + 1}/{total_images_needed})',
                     fontsize=16, weight='bold')

        # --- INNER LOOP: Fill the rows for this file ---
        for row_idx in range(rows_in_this_img):
            ax = axes[row_idx]

            # Determine which trials go in this row
            start_idx = current_trial_idx
            end_idx = min(current_trial_idx + trials_per_row, num_trials)

            # Define Time Limits for Plot
            chunk_start = cycles[start_idx][0]
            if end_idx < len(cycles):
                chunk_end = cycles[end_idx][0]
            else:
                chunk_end = cycles[end_idx - 1][1] + 20  # Fallback buffer

            ax.set_xlim(chunk_start - 2, chunk_end)
            ax.set_title(f"Trials {start_idx + 1} to {end_idx}")

            # --- PLOT TRIALS IN THIS ROW ---
            for i in range(start_idx, end_idx):
                t_start = cycles[i][0]
                t_mid = cycles[i][1]

                # Determine End of Trial
                if i < len(rests) and rests.size > 0:
                    t_end = rests[i]
                elif i + 1 < len(cycles):
                    t_end = cycles[i + 1][0]
                else:
                    t_end = t_mid + 15

                    # Filter Events for this specific trial
                t_licks = licks[(licks >= t_start) & (licks < t_end + 5)]
                t_waters = waters[(waters >= t_start) & (waters < t_end)]

                # Check for punishment
                t_punish = None
                if len(ten_s_delays) > 0:
                    p_in_trial = ten_s_delays[(ten_s_delays >= t_start) & (ten_s_delays < t_end)]
                    if len(p_in_trial) > 0:
                        t_punish = p_in_trial[0]

                # --- ANALYZE STATUS ---
                status, bg_color, win_dur = get_trial_outcome(phase, t_start, t_mid, t_end, t_licks, t_waters, t_punish)

                # --- DRAW GRAPHICS ---
                ax.axvspan(t_start, t_end, ymin=0.4, ymax=0.9, color=bg_color, alpha=0.4)

                rect = Rectangle((t_mid, 1.5), win_dur, 1.0, linewidth=1, edgecolor='green', facecolor='lime',
                                 alpha=0.2, hatch='...')
                ax.add_patch(rect)

                ax.vlines(t_mid, 1.4, 2.6, colors='black', linestyles='dashed', alpha=0.5)

                if t_punish:
                    ax.vlines(t_punish, 1.4, 2.6, colors='darkred', linewidth=3, alpha=0.8)
                    ax.text(t_punish, 2.7, "PUNISH", color='darkred', fontsize=8, ha='center', weight='bold')

                ax.text(t_start, 3.3, status, fontsize=9, color='black', style='italic', ha='left')

            # --- PLOT EVENTS (Bulk for efficiency in this row) ---
            view_licks = licks[(licks >= chunk_start - 2) & (licks <= chunk_end)]
            view_waters = waters[(waters >= chunk_start - 2) & (waters <= chunk_end)]

            ax.vlines(view_licks, 0.5, 1.0, color='red', linewidth=1.5, label='Lick')
            ax.scatter(view_waters, np.ones_like(view_waters) * 1.2, color='blue', s=80, zorder=10, label='Water')

            # Formatting
            ax.set_ylim(0, 3.6)
            ax.set_yticks([0.75, 1.2, 2.0])
            ax.set_yticklabels(['Licks', 'Water', 'Trial Status'])
            ax.grid(True, axis='x', alpha=0.2)

            # Legend on the very first subplot of the file
            if row_idx == 0:
                legend_elements = [
                    Patch(facecolor='palegreen', alpha=0.4, label='Success'),
                    Patch(facecolor='salmon', alpha=0.4, label='Fail/Punish'),
                    Patch(facecolor='lime', alpha=0.2, hatch='...', label=f'Reward Window ({win_dur}s)'),
                    Line2D([0], [0], color='black', linestyle='dashed', label='Sound Peak'),
                    Line2D([0], [0], color='red', lw=2, label='Lick'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Water')
                ]
                ax.legend(handles=legend_elements, loc='upper right', ncol=2)

            # Advance the trial counter for the next row
            current_trial_idx = end_idx

        plt.xlabel("Time (seconds)")

        # Save this specific Part
        filename = f"mouse1_phase_{phase}_session_{session_num}_part_{img_idx + 1}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Saved plot: {filename}")

        plt.close(fig)


# --- EXAMPLE USAGE ---
# This assumes 'data' is loaded.
# You can call this in your loop:
# phase = 1  (or 2, 3, 4 depending on your folder logic)
# plot_mouse_session_phased(data, session_num=1, phase=1, trials_per_plot=8)


import os

# 1. Define the Base Path (Remove 'trial_1' from the end)
base_path = r"C:\Users\shach\OneDrive\שולחן העבודה\Miki London Lab\Mice learning analysis\mouse_1\mouse_1\phase_3_training"
run_folder = create_run_folder(base_name="Phase1_Analysis")

# 2. Run your loop
for i in range(1, 9):
    # ... (load your data here) ...
    session_data = load_all_files(base_path)
    # 3. Pass the folder to the function
    plot_mouse_session_phased(
        session_data,
        session_num=i,
        phase=1,
        output_dir=run_folder  # <--- The images will go here
    )



