"""
recognition.py - Word recognition and confusion matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from preprocessing import preprocess_file
from features import extract_features
from vq import codebook_distance_is


def classify(lpc_matrix, gains, codebooks, words):
    """
    Classify a feature matrix using Itakura-Saito distance to LPC codebooks.

    Parameters
    ----------
    lpc_matrix : np.ndarray  (n_frames, lpc_order)
    gains : np.ndarray  (n_frames,)
    codebooks : dict  {word: {'lsf': ..., 'lpc': ..., 'gains': ...}}
    words : list[str]

    Returns
    -------
    predicted_word : str
    distances : dict {word: float}
    """
    distances = {}
    for word in words:
        cb = codebooks[word]
        lpc_cb = cb['lpc']
        gains_cb = cb['gains']
        distances[word] = codebook_distance_is(lpc_matrix, gains, lpc_cb, gains_cb)
    predicted = min(distances, key=distances.get)
    return predicted, distances


def evaluate(data_dir, words, codebooks_by_size, codebook_sizes,
             test_indices=None, lpc_order=12, output_dir="output"):
    """
    Evaluate recognition accuracy on test files and plot confusion matrices.
    Uses Itakura-Saito distance on LPC codebooks.

    Parameters
    ----------
    data_dir : str | Path
    words : list[str]
    codebooks_by_size : dict  {size: {word: {'lsf': ..., 'lpc': ..., 'gains': ...}}}
    codebook_sizes : list[int]
    test_indices : list[int]  1-based file indices used for testing
    lpc_order : int
    output_dir : str | Path

    Returns
    -------
    results : dict  {size: {'true': [...], 'pred': [...]}}
    accuracies : dict  {size: float}
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    if test_indices is None:
        test_indices = list(range(11, 16))  # files 11-15

    results = {size: {"true": [], "pred": []} for size in codebook_sizes}

    print("\n--- Recognition phase (using Itakura-Saito distance) ---")
    for word in words:
        word_dir = data_dir / word
        for idx in test_indices:
            candidates = [
                word_dir / f"{word}_{idx:02d}.wav",
                word_dir / f"{word}_{idx}.wav",
            ]
            filepath = next((p for p in candidates if p.exists()), None)
            if filepath is None:
                print(f"  [WARN] File not found for {word} index {idx}")
                continue

            frames, _, _, _, _, _ = preprocess_file(str(filepath))
            lsf_matrix, lpc_matrix, gains = extract_features(frames, lpc_order)

            for size in codebook_sizes:
                predicted, _ = classify(lpc_matrix, gains, codebooks_by_size[size], words)
                results[size]["true"].append(word)
                results[size]["pred"].append(predicted)
                print(f"  [{size:2d}cv] {filepath.name}: true={word:10s} pred={predicted}")

    accuracies = {}
    confusions_by_size = {}
    n = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}

    for size in codebook_sizes:
        true_labels = results[size]["true"]
        pred_labels = results[size]["pred"]

        # Build confusion matrix array for analysis
        matrix = np.zeros((n, n), dtype=int)
        for t, p in zip(true_labels, pred_labels):
            if t in word_to_idx and p in word_to_idx:
                matrix[word_to_idx[t]][word_to_idx[p]] += 1

        plot_confusion_matrix(true_labels, pred_labels, words,
                               title=f"Matriz de Confusión — Codebook tamaño {size}",
                               save_path=output_dir / f"confusion_{size}.png")

        acc = accuracy(true_labels, pred_labels)
        accuracies[size] = acc
        per_word_acc = per_word_accuracy(true_labels, pred_labels, words)
        pairs = top_confusions(matrix, words, n=5)
        confusions_by_size[size] = pairs

        print(f"\n=== Codebook tamaño {size} ===")
        print(f"  Precisión global: {acc:.1%}")
        for w, a in per_word_acc.items():
            print(f"  {w:12s}: {a:.1%}")
        if pairs:
            print(f"  Top confusiones:")
            for count, true_w, pred_w in pairs:
                print(f"    {true_w:10s} → {pred_w:10s}: {count} veces")

    # Summary plots
    plot_accuracy_by_size(
        codebook_sizes,
        [accuracies[s] for s in codebook_sizes],
        save_path=output_dir / "accuracy_by_size.png",
    )
    plot_top_confusions(confusions_by_size, save_path=output_dir / "top_confusions.png")

    return results, accuracies


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


def top_confusions(matrix, words, n=5):
    """Return top N off-diagonal (true, predicted) pairs with most errors."""
    pairs = []
    for i in range(len(words)):
        for j in range(len(words)):
            if i != j and matrix[i, j] > 0:
                pairs.append((matrix[i, j], words[i], words[j]))
    pairs.sort(reverse=True)
    return pairs[:n]


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


def plot_accuracy_by_size(sizes, accuracies, save_path=None):
    """Bar chart comparing global accuracy across codebook sizes."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["steelblue", "darkorange", "seagreen", "tomato", "mediumpurple"]
    bars = ax.bar(
        [str(s) for s in sizes],
        [a * 100 for a in accuracies],
        color=colors[:len(sizes)],
        edgecolor="black",
        width=0.5,
    )
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1%}",
            ha="center", va="bottom", fontweight="bold", fontsize=11,
        )
    ax.set_ylim(0, 115)
    ax.set_xlabel("Tamaño del codebook (número de codevectores)", fontsize=11)
    ax.set_ylabel("Precisión global (%)", fontsize=11)
    ax.set_title("Comparación de precisión por tamaño de codebook", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()


def plot_top_confusions(confusions_by_size, save_path=None):
    """Horizontal bar chart of top confused word pairs aggregated across all codebook sizes."""
    agg = {}
    for size, pairs in confusions_by_size.items():
        for count, true_w, pred_w in pairs:
            key = f"{true_w} → {pred_w}"
            agg[key] = agg.get(key, 0) + count

    if not agg:
        print("  No confusions recorded — all predictions correct.")
        return

    top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:10]
    labels = [t[0] for t in top]
    counts = [t[1] for t in top]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.55)))
    bars = ax.barh(labels[::-1], counts[::-1], color="salmon", edgecolor="black")
    for bar, cnt in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", fontweight="bold")
    ax.set_xlabel("Total de confusiones (suma de todos los codebooks)", fontsize=11)
    ax.set_title("Top palabras que más se confunden entre sí", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
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
