#!/usr/bin/env python3
"""Extract SentencePiece vocabulary from tokenizer.model → vocab.json.

Run this once after downloading the model:
    python tools/extract_vocab.py --model_dir models/cohere-transcribe-03-2026

This produces vocab.json in the model directory, which the Rust binary uses
to decode output token IDs to text without requiring the C++ sentencepiece library.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Extract vocab.json from tokenizer.model")
    parser.add_argument("--model_dir", required=True, type=Path,
                        help="Path to the model directory")
    args = parser.parse_args()

    model_path = args.model_dir / "tokenizer.model"
    output_path = args.model_dir / "vocab.json"

    if not model_path.exists():
        raise FileNotFoundError(f"tokenizer.model not found at {model_path}")

    try:
        import sentencepiece as spm
    except ImportError:
        raise ImportError(
            "sentencepiece not installed. Install it with: pip install sentencepiece"
        )

    sp = spm.SentencePieceProcessor()
    sp.Load(str(model_path))

    vocab = {str(i): sp.IdToPiece(i) for i in range(sp.GetPieceSize())}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=None)

    print(f"Wrote {len(vocab)} tokens to {output_path}")


if __name__ == "__main__":
    main()
