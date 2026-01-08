import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import stft
from scipy.io import wavfile

PATH = r"C:\presentation dump\T0000001.wav"

# ---- Load WAV (note: this reads full file; for very large files see note below) ----
fs, x = wavfile.read(PATH)
if x.ndim == 2:
    x = x.mean(axis=1)
if np.issubdtype(x.dtype, np.integer):
    x = x.astype(np.float32) / np.iinfo(x.dtype).max
else:
    x = x.astype(np.float32)

duration_s = x.shape[0] / fs
print("fs =", fs, "duration_s =", duration_s)

# ---- Spectrogram controls ----
L = 1024                 # window length (samples) ~4.1 ms @ 250 kHz
H = 250                  # hop (samples) = 1.0 ms @ 250 kHz
NFFT = 1024              # can set to 2048 for smoother freq sampling (zero-padding)
WINDOW = "hann"

FMIN, FMAX = 30_000, 120_000
GAIN_DB = 0.0
DYN_RANGE_DB = 80.0
VMAX_PERCENTILE = 99.9

VIEW_SEC = 10.0          # how much time to display at once (10s at 1ms hop -> 10,000 columns)

def compute_spec_segment(t0_s: float):
    i0 = int(t0_s * fs)
    i1 = int(min((t0_s + VIEW_SEC) * fs, x.shape[0]))
    seg = x[i0:i1]

    f, t, Z = stft(
        seg, fs=fs, window=WINDOW,
        nperseg=L, noverlap=L - H, nfft=NFFT,
        boundary=None, padded=False
    )
    S = np.abs(Z)
    S_db = 20 * np.log10(S + 1e-12) + GAIN_DB

    band = (f >= FMIN) & (f <= FMAX)
    f_band = f[band]
    S_db_band = S_db[band, :]

    # Robust display scaling (prevents rare loud noise from crushing contrast)
    vmax = np.percentile(S_db_band, VMAX_PERCENTILE)
    vmin = vmax - DYN_RANGE_DB

    return f_band, t0_s + t, S_db_band, vmin, vmax

# ---- Initial plot ----
t0 = 0.0
f_band, tt, S_db_band, vmin, vmax = compute_spec_segment(t0)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.18)

im = ax.imshow(
    S_db_band,
    origin="lower", aspect="auto",
    extent=[tt[0], tt[-1], f_band[0], f_band[-1]],
    vmin=vmin, vmax=vmax
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("dB (relative)")

ax.set_title("Spectrogram (30–120 kHz), 1 ms hop, robust scaling")

# ---- Slider to move along time ----
ax_slider = plt.axes([0.12, 0.06, 0.76, 0.04])
slider = Slider(ax_slider, "Start time (s)", 0.0, max(0.0, duration_s - VIEW_SEC), valinit=t0)

def on_change(val):
    t0_s = slider.val
    f_band, tt, S_db_band, vmin, vmax = compute_spec_segment(t0_s)

    im.set_data(S_db_band)
    im.set_extent([tt[0], tt[-1], f_band[0], f_band[-1]])
    im.set_clim(vmin=vmin, vmax=vmax)

    fig.canvas.draw_idle()

slider.on_changed(on_change)

plt.show()
