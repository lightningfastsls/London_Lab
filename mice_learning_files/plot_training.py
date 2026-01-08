import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import os

# --- CONFIGURATION ---
BASE_PATH = r"C:\Users\shach\PycharmProjects\mickey_london_lab"
MOUSE_ID = "mouse_2"  # Change this to plot a different mouse

# Load the Daily Data
file_path = os.path.join(BASE_PATH, f"{MOUSE_ID}_daily_learning.csv")
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Could not find {file_path}. Run the analysis script first!")

df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])  # Ensure dates are datetime objects

# Filter out Phase 0 (No Trial days) for the learning curves
plot_data = df[df['Phase'] > 0].copy()

# Set the visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def add_phase_lines(ax, data):
    """Draws vertical lines where the Phase changes."""
    # Find the indices where Phase changes
    # We use the 'Date' of the first session of the new phase
    phase_changes = data.loc[data['Phase'].diff() > 0, 'Date']

    for date in phase_changes:
        ax.axvline(date, color='black', linestyle='--', alpha=0.5)
        # Optional: Add text label
        # phase_val = data.loc[data['Date'] == date, 'Phase'].values[0]
        # ax.text(date, ax.get_ylim()[1], f"Phase {phase_val}", rotation=90, verticalalignment='top')


# --- PLOT 1: The Learning Curve (d' and Bias) ---
fig, ax1 = plt.subplots()

# Plot d' (Sensitivity) on Left Axis
sns.lineplot(data=plot_data, x='Date', y='d_prime', marker='o', color='b', label="Sensitivity (d')", ax=ax1)
ax1.set_ylabel("Sensitivity ($d'$)", color='b', fontsize=12)
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_ylim(-1, 4)  # Standard d' range
ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)  # Chance level
ax1.axhline(1.5, color='green', linestyle=':', alpha=0.5, label="Learning Threshold")  # Criterion

# Plot Criterion (Bias) on Right Axis
ax2 = ax1.twinx()
sns.lineplot(data=plot_data, x='Date', y='criterion', marker='x', linestyle='--', color='r', label="Bias (c)", ax=ax2)
ax2.set_ylabel("Bias (c)\n(>0 = Conservative, <0 = Liberal)", color='r', fontsize=12)
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_ylim(-1.5, 1.5)
ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)  # Unbiased

# Formatting
add_phase_lines(ax1, plot_data)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax1.set_title(f"{MOUSE_ID}: Learning Curve (Sensitivity vs. Bias)", fontsize=16)

# Combine legends manually
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()

# --- PLOT 2: Diagnostic (Hit Rate vs False Alarm) ---
plt.figure()
sns.lineplot(data=plot_data, x='Date', y='hit_rate', marker='o', color='g', label="Hit Rate")
sns.lineplot(data=plot_data, x='Date', y='fa_rate', marker='o', color='m', label="False Alarm Rate")

add_phase_lines(plt.gca(), plot_data)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.ylim(0, 1.05)
plt.ylabel("Probability")
plt.title(f"{MOUSE_ID}: Performance Breakdown", fontsize=16)
plt.legend()
plt.tight_layout()
plt.show()

# --- PLOT 3: Reaction Time ---
# Filter out sessions where RT is NaN (e.g. 0 hits)
rt_data = plot_data.dropna(subset=['median_rt'])

plt.figure()
sns.lineplot(data=rt_data, x='Date', y='median_rt', marker='s', color='orange')

add_phase_lines(plt.gca(), rt_data)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.ylabel("Median Reaction Time (s)")
plt.title(f"{MOUSE_ID}: Processing Speed", fontsize=16)
plt.tight_layout()
plt.show()