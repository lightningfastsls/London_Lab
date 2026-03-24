# Real-time Spectrogram Streaming

I want to add real-time streaming to the spectrogram viewer. The basic idea:

## Step 1: Audio Capture

Set up microphone input at 300kHz. Buffer audio chunks for processing.

## Step 2: Streaming STFT

Process audio chunks as they arrive using overlapping windows. Show a live scrolling spectrogram display.

## Step 3: Live Detection

Run the energy detector on the stream. Highlight detected USVs in real-time on the spectrogram.

## Step 4: Recording

Let the user start/stop recording interesting segments to WAV files for later analysis.
