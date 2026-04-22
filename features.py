"""
features.py - LPC and LSF feature extraction.
"""

import numpy as np
from scipy.linalg import solve_toeplitz


def lpc_coefficients(frame, order=12):
    """
    Compute LPC coefficients of given order using the Levinson-Durbin algorithm.

    Parameters
    ----------
    frame : np.ndarray  (frame_len,)
    order : int

    Returns
    -------
    a : np.ndarray  (order,)
        LPC coefficients a[1..order]  (a[0] = 1 is implicit).
    gain : float
        Prediction error power.
    """
    # Autocorrelation via FFT (efficient)
    n = len(frame)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size <<= 1

    X = np.fft.rfft(frame, n=fft_size)
    R_full = np.fft.irfft(X * np.conj(X))[:order + 1]
    R = R_full.real

    if R[0] < 1e-10:
        return np.zeros(order), 1e-10

    # Levinson-Durbin via Toeplitz solver
    try:
        a = solve_toeplitz(R[:order], -R[1:order + 1])
    except np.linalg.LinAlgError:
        return np.zeros(order), R[0]

    gain = R[0] + np.dot(a, R[1:order + 1])
    gain = max(gain, 1e-10)
    return a, gain


def lpc_to_lsf(a, order=12):
    """
    Convert LPC coefficients to Line Spectral Frequencies (LSF / LSP).

    The LPC polynomial A(z) = 1 + a[0]z^-1 + ... + a[p-1]z^-p is split into
    two symmetric polynomials P and Q, whose roots alternate on the unit circle.

    Returns
    -------
    lsf : np.ndarray  (order,)
        LSF values in [0, π] radians, sorted ascending.
    """
    p = len(a)
    # Form P(z) and Q(z)
    a_full = np.concatenate(([1.0], a))
    a_flip = a_full[::-1]

    P = a_full + a_flip   # symmetric
    Q = a_full - a_flip   # anti-symmetric

    # Remove trivial roots: P has root at z=-1, Q has root at z=+1
    P = np.polymul(P, [1, 1])     # divide out (z+1) factor after root finding
    Q = np.polymul(Q, [1, -1])

    # Find roots of each polynomial
    roots_P = np.roots(P)
    roots_Q = np.roots(Q)

    # Keep roots on (or near) the unit circle with positive imaginary part
    def unit_circle_angles(roots):
        angles = []
        for r in roots:
            if np.abs(np.abs(r) - 1.0) < 0.3 and np.imag(r) > 1e-6:
                angles.append(np.angle(r))
        return np.sort(angles)

    angles_P = unit_circle_angles(roots_P)
    angles_Q = unit_circle_angles(roots_Q)

    lsf = np.sort(np.concatenate([angles_P, angles_Q]))

    # If root finding fails to give exactly p/2 from each, fall back to
    # uniformly spaced LSFs to keep feature vector size fixed.
    if len(lsf) != p:
        lsf = np.linspace(0.1, np.pi - 0.1, p)

    return lsf[:p]


def extract_features(frames, lpc_order=12):
    """
    Extract LSF feature matrix from a set of frames.

    Parameters
    ----------
    frames : np.ndarray  (n_frames, frame_len)
    lpc_order : int

    Returns
    -------
    features : np.ndarray  (n_frames, lpc_order)
        Each row is the LSF vector for one frame.
    lpc_matrix : np.ndarray  (n_frames, lpc_order)
        Raw LPC coefficients (needed for Itakura-Saito distance).
    gains : np.ndarray  (n_frames,)
        Per-frame prediction gain.
    """
    n_frames = len(frames)
    features = np.zeros((n_frames, lpc_order))
    lpc_matrix = np.zeros((n_frames, lpc_order))
    gains = np.zeros(n_frames)

    for i, frame in enumerate(frames):
        a, g = lpc_coefficients(frame, lpc_order)
        lsf = lpc_to_lsf(a, lpc_order)
        features[i] = lsf
        lpc_matrix[i] = a
        gains[i] = g

    return features, lpc_matrix, gains
