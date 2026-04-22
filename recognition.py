"""
recognition.py - Word recognition and confusion matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from preprocessing import preprocess_file
from features import extract_features
from vq import codebook_distance_lsf


def classify(lsf_matrix, codebooks, words):
    """
    Classify a feature matrix by finding the word whose codebook gives
    the minimum average distortion.

    Parameters
    ----------
    lsf_matrix : np.ndarray  (n_frames, lpc_order)
    codebooks : dict  {word: codebook_array}
    words : list[str]

    Returns
    -------
    predicted_word : str
    distances : dict {word: float}
    """
    distances = {}
    for word in words:
        cb = codebooks[word]
        distances[word] = codebook_distance_lsf(lsf_matrix, cb)
    predicted = min(distances, key=distances.get)
    return predicted, distances


def evaluate(data_dir, words, codebooks_by_size, codebook_sizes,
             test_indices=None, lpc_order=12, output_dir="output"):
    """
    Evaluate recognition accuracy on test files and plot confusion matrices.

    Parameters
    ----------
    data_dir : str | Path
    words : list[str]
    codebooks_by_size : dict  {size: {word: codebook}}
    codebook_sizes : list[int]
    test_indices : list[int]  1-based file indices used for testing
    lpc_order : int
    output_dir : str | Path
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    if test_indices is None:
        test_indices = list(range(11, 16))  # files 11-15

    results = {size: {"true": [], "pred": []} for size in codebook_sizes}

    print("\n--- Recognition phase ---")
    for word in words:
        word_dir = data_dir / word
        for idx in test_indices:
            # Try zero-padded filenames: word_01.wav, word_1.wav, etc.
            candidates = [
                word_dir / f"{word}_{idx:02d}.wav",
                word_dir / f"{word}_{idx}.wav",
            ]
            filepath = next((p for p in candidates if p.exists()), None)
            if filepath is None:
                print(f"  [WARN] File not found for {word} index {idx}")
                continue

            frames, *_ = preprocess_file(str(filepath))
            lsf_matrix, _, _ = extract_features(frames, lpc_order)

            for size in codebook_sizes:
                predicted, _ = classify(lsf_matrix, codebooks_by_size[size], words)
                results[size]["true"].append(word)
                results[size]["pred"].append(predicted)
                print(f"  [{size:2d}cv] {filepath.name}: true={word:10s} pred={predicted}")

    for size in codebook_sizes:
        true_labels = results[size]["true"]
        pred_labels = results[size]["pred"]
        plot_confusion_matrix(true_labels, pred_labels, words,
                               title=f"Confusion Matrix — Codebook size {size}",
                               save_path=output_dir / f"confusion_{size}.png")
        acc = accuracy(true_labels, pred_labels)
        per_word_acc = per_word_accuracy(true_labels, pred_labels, words)
        print(f"\n=== Codebook size {size} ===")
        print(f"  Global accuracy: {acc:.1%}")
        for w, a in per_word_acc.items():
            print(f"  {w:12s}: {a:.1%}")

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def accuracy(true_labels, pred_labels):
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    return correct / len(true_labels) if true_labels else 0.0


def per_word_accuracy(true_labels, pred_labels, words):
    acc = {}
    for word in words:
        indices = [i for i, t in enumerate(true_labels) if t == word]
        if not indices:
            acc[word] = 0.0
            continue
        correct = sum(pred_labels[i] == word for i in indices)
        acc[word] = correct / len(indices)
    return acc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(true_labels, pred_labels, words,
                           title="Confusion Matrix", save_path=None):
    """Build and plot a confusion matrix."""
    n = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}

    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        if t in word_to_idx and p in word_to_idx:
            matrix[word_to_idx[t]][word_to_idx[p]] += 1

    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n)))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(n), yticks=np.arange(n),
        xticklabels=words, yticklabels=words,
        ylabel="True label", xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = matrix.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(matrix[i, j]),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black",
                    fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()


def plot_vad(raw_signal, start_sample, end_sample, zcr, energy,
             sr=16000, vad_hop=160, title="VAD Detection", save_path=None):
    """
    Plot signal with VAD markers, energy, and ZCR per frame.
    Matches the visualization from inicio_fin.m.
    """
    t = np.arange(len(raw_signal)) / sr
    t_start = start_sample / sr
    t_end   = end_sample   / sr

    # Frame time axis for VAD features (hop = 160 samples = 10 ms)
    frame_times = np.arange(len(zcr)) * vad_hop / sr

    fig, axes = plt.subplots(3, 1, figsize=(12, 7))
    fig.suptitle(title)

    # Waveform — equivalent to subplot(2,1,1) in .m with xline markers
    axes[0].plot(t, raw_signal, linewidth=0.5, color="steelblue")
    axes[0].axvline(t_start, color="green", linestyle="--", label="Inicio detectado")
    axes[0].axvline(t_end,   color="red",   linestyle="--", label="Fin detectado")
    axes[0].set_ylabel("Amplitud")
    axes[0].set_title("Señal original con detección de inicio y fin de palabra")
    axes[0].legend(fontsize=8)

    # Normalized linear energy (sum(frame^2)/frame_len) — matches .m
    axes[1].plot(frame_times, energy, color="darkorange")
    axes[1].axhline(0.03 * np.max(energy), color="black", linestyle=":", linewidth=0.8,
                    label="Umbral energía (3%)")
    axes[1].axvline(t_start, color="green", linestyle="--")
    axes[1].axvline(t_end,   color="red",   linestyle="--")
    axes[1].set_ylabel("Energía normalizada")
    axes[1].legend(fontsize=8)

    # ZCR
    axes[2].plot(frame_times, zcr, color="purple")
    axes[2].axhline(0.08 * np.max(zcr), color="black", linestyle=":", linewidth=0.8,
                    label="Umbral ZCR (8%)")
    axes[2].axvline(t_start, color="green", linestyle="--")
    axes[2].axvline(t_end,   color="red",   linestyle="--")
    axes[2].set_ylabel("ZCR")
    axes[2].set_xlabel("Tiempo [s]")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()
