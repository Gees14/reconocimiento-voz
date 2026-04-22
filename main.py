"""
main.py - Speech word recognition system entry point.

Usage:
    python main.py --data_dir data --codebook_sizes 16 32 64
"""

import argparse
import numpy as np
from pathlib import Path

from preprocessing import preprocess_file
from features import extract_features
from vq import lbg_train, save_codebooks, load_codebooks
from recognition import evaluate, plot_vad


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_words(data_dir):
    """Return sorted list of word subdirectories found under data_dir."""
    data_dir = Path(data_dir)
    return sorted([d.name for d in data_dir.iterdir() if d.is_dir()])


def collect_files(data_dir, word, indices):
    """Return list of existing WAV paths for a word and a list of 1-based indices."""
    word_dir = Path(data_dir) / word
    paths = []
    for idx in indices:
        candidates = [
            word_dir / f"{word}_{idx:02d}.wav",
            word_dir / f"{word}_{idx}.wav",
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found:
            paths.append(found)
        else:
            print(f"  [WARN] Missing file for {word} index {idx}")
    return paths


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_codebooks(data_dir, words, codebook_sizes,
                    train_indices=None, lpc_order=12):
    """
    Build one LSF codebook per word per requested size.

    Returns
    -------
    codebooks_by_size : dict  {size: {word: np.ndarray (size, lpc_order)}}
    """
    if train_indices is None:
        train_indices = list(range(1, 11))  # files 1-10

    codebooks_by_size = {size: {} for size in codebook_sizes}

    print("--- Training phase ---")
    for word in words:
        print(f"\n  Word: {word}")
        all_lsf = []

        for filepath in collect_files(data_dir, word, train_indices):
            frames, _, _, raw, energies, zcrs = preprocess_file(str(filepath))
            lsf_matrix, _, _ = extract_features(frames, lpc_order)
            all_lsf.append(lsf_matrix)

        if not all_lsf:
            print(f"  [ERROR] No training files found for '{word}'. Skipping.")
            continue

        all_lsf = np.vstack(all_lsf)
        print(f"  Training vectors: {all_lsf.shape}")

        for size in codebook_sizes:
            print(f"  Training codebook size={size} ...", end=" ", flush=True)
            codebook = lbg_train(all_lsf, codebook_size=size)
            codebooks_by_size[size][word] = codebook
            print("done")

    return codebooks_by_size


# ---------------------------------------------------------------------------
# Visualization demo
# ---------------------------------------------------------------------------

def visualize_vad_examples(data_dir, words, output_dir, n_examples=2):
    """Plot VAD detection for a few files to verify preprocessing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for word in words[:n_examples]:
        filepath = collect_files(data_dir, word, [1])
        if not filepath:
            continue
        filepath = filepath[0]
        _, start, end, raw, energies, zcrs = preprocess_file(str(filepath))
        plot_vad(
            raw, start, end, energies, zcrs,
            title=f"VAD — {filepath.name}",
            save_path=output_dir / f"vad_{word}.png",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Speech word recognizer (LPC + VQ)")
    p.add_argument("--data_dir", default="data",
                   help="Root folder containing one subfolder per word")
    p.add_argument("--codebook_sizes", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--lpc_order", type=int, default=12)
    p.add_argument("--train_files", nargs="+", type=int,
                   default=list(range(1, 11)),
                   help="1-based file indices used for training (default: 1-10)")
    p.add_argument("--test_files", nargs="+", type=int,
                   default=list(range(11, 16)),
                   help="1-based file indices used for testing (default: 11-15)")
    p.add_argument("--output_dir", default="output",
                   help="Directory for figures and codebook pickle")
    p.add_argument("--codebook_file", default="codebooks.pkl")
    p.add_argument("--skip_training", action="store_true",
                   help="Load existing codebooks from --codebook_file")
    p.add_argument("--vad_demo", action="store_true",
                   help="Plot VAD detection for the first 2 words")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    words = discover_words(data_dir)
    if not words:
        print(f"[ERROR] No word subdirectories found in '{data_dir}'. "
              "Create data/<word>/ folders with WAV files.")
        return

    print(f"Words found: {words}")
    print(f"Codebook sizes: {args.codebook_sizes}")
    print(f"Train indices: {args.train_files}  |  Test indices: {args.test_files}\n")

    # Optional VAD visualization
    if args.vad_demo:
        print("Generating VAD demo plots...")
        visualize_vad_examples(data_dir, words, output_dir)

    # Training
    codebook_path = output_dir / args.codebook_file
    if args.skip_training and codebook_path.exists():
        print(f"Loading codebooks from {codebook_path}")
        codebooks_by_size = load_codebooks(str(codebook_path))
    else:
        codebooks_by_size = train_codebooks(
            data_dir, words, args.codebook_sizes,
            train_indices=args.train_files,
            lpc_order=args.lpc_order,
        )
        save_codebooks(codebooks_by_size, str(codebook_path))

    # Recognition + confusion matrices
    evaluate(
        data_dir=data_dir,
        words=words,
        codebooks_by_size=codebooks_by_size,
        codebook_sizes=args.codebook_sizes,
        test_indices=args.test_files,
        lpc_order=args.lpc_order,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
