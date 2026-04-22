"""
preprocessing.py - Signal preprocessing for speech recognition.
Handles pre-emphasis, framing, windowing, and voice activity detection (VAD).

VAD logic is a direct Python translation of inicio_fin.m:
  - Linear normalized energy:  E[i] = sum(frame^2) / frame_len
  - ZCR:                       Z[i] = sum(|diff(sign(frame))|) / 2 / frame_len
  - Thresholds:                zcr_thr  = 0.08 * max(Z)
                               energy_thr = 0.03 * max(E)
  - Voice frame if:            Z[i] > zcr_thr  AND  E[i] > energy_thr
  - VAD uses 20 ms frames / 10 ms hop (320 / 160 samples at 16 kHz)
  - LPC framing uses 320-point Hamming window with 128-sample hop
"""

import numpy as np
import librosa

# VAD parameters (matches inicio_fin.m at 16 kHz)
VAD_FRAME_LEN = 320   # 20 ms
VAD_HOP_LEN   = 160   # 10 ms
ZCR_THRESH_FACTOR    = 0.08   # 8 % of max ZCR
ENERGY_THRESH_FACTOR = 0.03   # 3 % of max energy

# LPC framing parameters (task specification)
LPC_FRAME_LEN = 320
LPC_HOP_LEN   = 128


def load_audio(filepath, sr=16000):
    """Load WAV file and resample to target sample rate."""
    signal, _ = librosa.load(filepath, sr=sr, mono=True)
    return signal


def preemphasis(signal, coeff=0.95):
    """Apply pre-emphasis filter: y[n] = x[n] - coeff * x[n-1]."""
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def frame_signal(signal, frame_len=LPC_FRAME_LEN, hop_len=LPC_HOP_LEN):
    """
    Split signal into overlapping frames with Hamming window.
    Returns array of shape (num_frames, frame_len).
    """
    if len(signal) < frame_len:
        signal = np.pad(signal, (0, frame_len - len(signal)))

    num_frames = (len(signal) - frame_len) // hop_len + 1
    window = np.hamming(frame_len)
    frames = np.zeros((num_frames, frame_len))
    for i in range(num_frames):
        start = i * hop_len
        frames[i] = signal[start:start + frame_len] * window
    return frames


def _compute_vad_features(signal, frame_len=VAD_FRAME_LEN, hop_len=VAD_HOP_LEN):
    """
    Compute per-frame ZCR and linear normalized energy (no windowing, matches .m).
    Equivalent to the for-loop in inicio_fin.m.
    """
    num_frames = (len(signal) - frame_len) // hop_len
    zcr    = np.zeros(num_frames)
    energy = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_len          # Python 0-indexed
        frame = signal[start:start + frame_len]

        # ZCR: matches  sum(abs(diff(sign(frame)))) / 2 / frame_length
        signs = np.sign(frame)
        zcr[i] = np.sum(np.abs(np.diff(signs))) / 2 / frame_len

        # Energy: matches  sum(frame.^2) / frame_length
        energy[i] = np.sum(frame ** 2) / frame_len

    return zcr, energy


def detect_voice_activity(signal,
                           frame_len=VAD_FRAME_LEN,
                           hop_len=VAD_HOP_LEN,
                           zcr_factor=ZCR_THRESH_FACTOR,
                           energy_factor=ENERGY_THRESH_FACTOR):
    """
    Detect start and end of voiced speech.
    Direct Python translation of inicio_fin.m.

    Parameters
    ----------
    signal : np.ndarray   Raw (or pre-emphasized) audio signal.
    frame_len : int       Frame length in samples (default 320 = 20 ms).
    hop_len : int         Hop length in samples (default 160 = 10 ms).
    zcr_factor : float    Fraction of max ZCR used as threshold (default 0.08).
    energy_factor : float Fraction of max energy used as threshold (default 0.03).

    Returns
    -------
    start_sample : int    First sample of detected speech region.
    end_sample : int      Last sample of detected speech region.
    zcr : np.ndarray      Per-frame ZCR values.
    energy : np.ndarray   Per-frame linear energy values.
    """
    zcr, energy = _compute_vad_features(signal, frame_len, hop_len)

    # Thresholds — matches inicio_fin.m lines 24-25
    zcr_threshold    = zcr_factor    * np.max(zcr)
    energy_threshold = energy_factor * np.max(energy)

    # Voice frames: both ZCR AND energy above threshold — matches line 28
    voice_flags = (zcr > zcr_threshold) & (energy > energy_threshold)

    voiced_indices = np.where(voice_flags)[0]

    if len(voiced_indices) == 0:
        # Fallback: return full signal
        return 0, len(signal) - 1, zcr, energy

    first_voice_frame = voiced_indices[0]
    last_voice_frame  = voiced_indices[-1]

    # Convert frame indices to sample indices — matches lines 40-41 of .m
    # (.m is 1-indexed; Python is 0-indexed, so no +1 offset needed)
    start_sample = first_voice_frame * hop_len
    end_sample   = last_voice_frame  * hop_len + frame_len

    # Clamp to signal length — matches line 44
    end_sample = min(end_sample, len(signal) - 1)

    return start_sample, end_sample, zcr, energy


def preprocess_file(filepath, sr=16000,
                    lpc_frame_len=LPC_FRAME_LEN, lpc_hop_len=LPC_HOP_LEN,
                    preemph_coeff=0.95):
    """
    Full preprocessing pipeline for one audio file.

    Steps
    -----
    1. Load WAV at sr Hz.
    2. Apply pre-emphasis filter.
    3. Run VAD (inicio_fin.m logic) to find speech boundaries.
    4. Trim signal to VAD region.
    5. Frame trimmed signal with Hamming window (for LPC).

    Returns
    -------
    frames : np.ndarray  (n_frames, lpc_frame_len)
    start_sample, end_sample : int   VAD boundaries in the original signal.
    raw_signal : np.ndarray          Original signal (for plotting).
    zcr, energy : np.ndarray         Per-frame VAD features (for plotting).
    """
    raw_signal = load_audio(filepath, sr=sr)
    emphasized = preemphasis(raw_signal, preemph_coeff)

    start_sample, end_sample, zcr, energy = detect_voice_activity(emphasized)

    trimmed = emphasized[start_sample:end_sample + 1]
    frames  = frame_signal(trimmed, lpc_frame_len, lpc_hop_len)

    return frames, start_sample, end_sample, raw_signal, zcr, energy
