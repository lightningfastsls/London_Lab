import numpy as np
import simpleaudio as sa
import multiprocessing
import atexit
import time

def play_tone(frequency, duration, sample_rate=44100):
    # Generate a sine wave for the given frequency and duration
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(frequency * t * 2 * np.pi)
    # Normalize to 16-bit range
    wave = (wave * 32767).astype(np.int16)
    # Start playback
    play_obj = sa.play_buffer(wave, 1, 2, sample_rate)
    # Ensure the playback finishes
    play_obj.wait_done()

def cleanup_process(process):
    # Cleanup function to terminate the process
    if process.is_alive():
        process.terminate()
    print("Process terminated and resources released.")

def sound_process(frequency, duration):
    # Create a separate process for playing the sound
    p = multiprocessing.Process(target=play_tone, args=(frequency, duration))
    p.start()
    # Register the cleanup function
    atexit.register(cleanup_process, p)
    return p

if __name__ == "__main__":
    # Example: Play a 440Hz tone for 1 second
    process = sound_process(440, 10)

    for i in range(10):
        print(i)
        time.sleep(0.2)
        process = sound_process(440+i*10, 10)

    time.sleep(3)
    # Additional logic could go here
    # Wait for the sound to finish or perform other tasks

    # Optionally, manually stop the sound process if needed
    # This is more of a just-in-case measure since the tone should stop on its own
    # after the specified duration.
    if process.is_alive():
        process.terminate()
        print("Sound process was manually terminated.")
