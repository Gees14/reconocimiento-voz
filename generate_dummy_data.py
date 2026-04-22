"""
generate_dummy_data.py - Creates synthetic WAV files for testing without real recordings.
DELETE this file once you have your actual recordings in place.
"""

import numpy as np
import soundfile as sf
from pathlib import Path

WORDS = ["start", "stop", "lift", "drop", "left", "right", "up", "down", "go", "back"]
SR = 16000
N_FILES = 15
DURATION = 0.8  # seconds


def make_word_signal(word_idx, file_idx, sr=SR, duration=DURATION):
    """
    Synthetic signal: different base frequency per word, slight variation per file.
    Real recordings will replace this.
    """
    t = np.linspace(0, duration, int(sr * duration))
    base_freq = 150 + word_idx * 40
    variation = 1 + 0.05 * (file_idx - 8)  # small pitch variation across files

    # Sum of harmonics with amplitude envelope
    signal = np.zeros_like(t)
    for harmonic in range(1, 6):
        signal += (1 / harmonic) * np.sin(2 * np.pi * base_freq * harmonic * variation * t)

    # Amplitude envelope: ramp up/down to simulate a spoken word
    env = np.ones_like(t)
    ramp = int(0.05 * sr)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    signal *= env * 0.3

    # Add noise
    signal += np.random.randn(len(t)) * 0.01
    return signal.astype(np.float32)


def main():
    data_dir = Path("data")
    for wi, word in enumerate(WORDS):
        word_dir = data_dir / word
        word_dir.mkdir(parents=True, exist_ok=True)
        for fi in range(1, N_FILES + 1):
            sig = make_word_signal(wi, fi)
            out = word_dir / f"{word}_{fi:02d}.wav"
            sf.write(str(out), sig, SR)
        print(f"  Created {N_FILES} files for '{word}'")
    print(f"\nDone. Data in '{data_dir.resolve()}'")


if __name__ == "__main__":
    main()
