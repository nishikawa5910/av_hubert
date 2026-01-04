#!/usr/bin/env python3
"""Extract contentvec decoder weights from a full checkpoint.

Usage:
  python avhubert/misc/extract_contentvec_decoder.py \
    --input /path/to/checkpoint.pt \
    --output /path/to/contentvec_decoder_only.pt
"""

import argparse
import torch

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract contentvec decoder weights")
    parser.add_argument("--input", required=True, help="Path to full checkpoint")
    parser.add_argument("--output", required=True, help="Output path for decoder-only checkpoint")
    args = parser.parse_args()

    state = torch.load(args.input, map_location="cpu")
    model_state = state.get("model", {})
    decoder_state = {
        key: value for key, value in model_state.items() if key.startswith("decoder.")
    }
    if not decoder_state:
        raise ValueError("No decoder.* weights found in checkpoint.")

    torch.save({"model": decoder_state}, args.output)


if __name__ == "__main__":
    main()
