"""
vq.py - Vector Quantization using the LBG algorithm with Itakura-Saito distance.
"""

import numpy as np
import pickle
from pathlib import Path


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------

def itakura_saito_distance(a1, g1, a2, g2):
    """
    Itakura-Saito distortion between two LPC frames.

    d_IS = (g1/g2) * (A2 · R1 · A2^T) / g1  - log(g1/g2) - 1
         = (A2 · R1 · A2^T) / g2 - log(g1/g2) - 1

    We use the simplified symmetric-ish version commonly used in VQ:
        d = (a2^T R1 a2) / g1 - 1
    where R1 is the autocorrelation matrix implied by (a1, g1).

    In practice we compute via LPC spectral ratio:
        d = (1/2π) ∫ [H1(ω)/H2(ω) - log(H1(ω)/H2(ω)) - 1] dω

    For efficiency we use the closed-form expression with autocorrelation:
        d_IS(1→2) = g1 * [a2^T R1^{-1} a2] / g2 - log(g1/g2) - 1

    Since computing R1^{-1} is expensive, we use the common approximation
    based on the LPC coefficient inner products:
        d ≈ sum((a1 - a2)^2) / (g1 + g2)

    For a proper Itakura-Saito, we approximate the spectral ratio on a grid.
    """
    n_fft = 512
    # Build A polynomials: A(z) = 1 + a[0]z^{-1} + ... (evaluated on unit circle)
    a1_full = np.concatenate(([1.0], a1))
    a2_full = np.concatenate(([1.0], a2))

    A1 = np.abs(np.fft.rfft(a1_full, n=n_fft)) ** 2  # |A1(ω)|²
    A2 = np.abs(np.fft.rfft(a2_full, n=n_fft)) ** 2  # |A2(ω)|²

    # Power spectra: H(ω) = g / |A(ω)|²
    H1 = g1 / (A1 + 1e-10)
    H2 = g2 / (A2 + 1e-10)

    ratio = H1 / (H2 + 1e-10)
    # IS distortion: mean of (ratio - ln(ratio) - 1)
    d = np.mean(ratio - np.log(ratio + 1e-10) - 1.0)
    return max(d, 0.0)


def lsf_euclidean_distance(lsf1, lsf2):
    """Simple Euclidean distance between LSF vectors (used during LBG iterations)."""
    return np.sum((lsf1 - lsf2) ** 2)


# ---------------------------------------------------------------------------
# LBG Algorithm
# ---------------------------------------------------------------------------

def _nearest_codevector(vector, codebook):
    """Return index of nearest codevector using Euclidean distance."""
    dists = np.sum((codebook - vector) ** 2, axis=1)
    return np.argmin(dists)


def lbg_train(training_vectors, codebook_size, epsilon=0.01, max_iter=100):
    """
    Train a VQ codebook using the Linde-Buzo-Gray (LBG) algorithm.

    Parameters
    ----------
    training_vectors : np.ndarray  (N, D)
        All feature vectors (LSFs) from the training utterances.
    codebook_size : int
        Target number of codevectors (must be a power of 2).
    epsilon : float
        Splitting perturbation factor.
    max_iter : int
        Maximum Lloyd iterations per doubling step.

    Returns
    -------
    codebook : np.ndarray  (codebook_size, D)
    """
    N, D = training_vectors.shape
    assert N >= codebook_size, "Need at least as many vectors as codevectors."

    # Start with centroid of all training data
    codebook = np.mean(training_vectors, axis=0, keepdims=True)  # (1, D)

    while len(codebook) < codebook_size:
        # Split each codevector into two
        perturb = epsilon * (np.random.rand(*codebook.shape) * 2 - 1)
        codebook = np.vstack([codebook * (1 + perturb),
                               codebook * (1 - perturb)])

        # Lloyd iteration until convergence
        prev_distortion = np.inf
        for _ in range(max_iter):
            # Assignment step
            assignments = np.array([_nearest_codevector(v, codebook)
                                     for v in training_vectors])

            # Update step
            new_codebook = np.zeros_like(codebook)
            for k in range(len(codebook)):
                members = training_vectors[assignments == k]
                if len(members) > 0:
                    new_codebook[k] = np.mean(members, axis=0)
                else:
                    # Dead cell: reinitialize to a random training vector
                    new_codebook[k] = training_vectors[np.random.randint(N)]

            # Distortion
            distortion = np.mean([
                lsf_euclidean_distance(training_vectors[i], codebook[assignments[i]])
                for i in range(N)
            ])

            codebook = new_codebook
            if abs(prev_distortion - distortion) / (prev_distortion + 1e-10) < epsilon:
                break
            prev_distortion = distortion

    return codebook


# ---------------------------------------------------------------------------
# Codebook distance for recognition
# ---------------------------------------------------------------------------

def codebook_distance_lsf(lsf_matrix, codebook):
    """
    Average distortion of lsf_matrix against a codebook (Euclidean on LSFs).
    Used for recognition: lower = better match.
    """
    total = 0.0
    for vec in lsf_matrix:
        dists = np.sum((codebook - vec) ** 2, axis=1)
        total += np.min(dists)
    return total / len(lsf_matrix)


def codebook_distance_is(lpc_matrix, gains, codebook_lpc, codebook_gains):
    """
    Average Itakura-Saito distortion between LPC frames and a codebook.
    """
    total = 0.0
    n_frames = len(lpc_matrix)
    for i in range(n_frames):
        dists = [
            itakura_saito_distance(lpc_matrix[i], gains[i],
                                   codebook_lpc[k], codebook_gains[k])
            for k in range(len(codebook_lpc))
        ]
        total += min(dists)
    return total / n_frames


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_codebooks(codebooks, path="codebooks.pkl"):
    """Save dict of {word: {size: codebook}} to disk."""
    with open(path, "wb") as f:
        pickle.dump(codebooks, f)
    print(f"Codebooks saved to {path}")


def load_codebooks(path="codebooks.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
