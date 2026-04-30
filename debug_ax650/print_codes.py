#!/usr/bin/env python3
"""
Print AX650 output_codes.bin for inspection.

Usage:
    python print_codes.py [output_codes.bin] [output_meta.json]

Defaults to files in the same directory.
"""

import os
import sys
import json
import struct
import numpy as np


def main():
    work_dir = os.path.dirname(os.path.abspath(__file__))

    bin_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(work_dir, "output_codes.bin")
    meta_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(work_dir, "output_meta.json")

    if not os.path.exists(bin_path):
        print(f"[ERROR] File not found: {bin_path}")
        sys.exit(1)

    # Read metadata if available
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        num_frames = meta.get("num_frames")
        num_codebooks = meta.get("num_codebooks", 16)
        print(f"[META] {meta}")
    else:
        # Infer from file size: each frame = 16 * 4 bytes (int32)
        file_size = os.path.getsize(bin_path)
        num_codebooks = 16
        num_frames = file_size // (num_codebooks * 4)
        print(f"[INFER] No meta.json, inferred num_frames={num_frames} from file_size={file_size}")

    # Read and reshape
    codes = np.fromfile(bin_path, dtype=np.int32)
    if num_frames is None:
        num_frames = len(codes) // num_codebooks
    codes = codes.reshape(num_frames, num_codebooks)

    print(f"\n[SHAPE] {codes.shape}  dtype={codes.dtype}")
    print(f"[RANGE] primary (codebook 0): [{codes[:, 0].min()}, {codes[:, 0].max()}]")
    print(f"[RANGE] residual (codebook 1..15): [{codes[:, 1:].min()}, {codes[:, 1:].max()}]")

    # Print first N frames
    n_show = min(800, num_frames)
    print(f"\n[FIRST {n_show} FRAMES]")
    for i in range(n_show):
        row = codes[i]
        print(f"  frame={i:3d}  primary={row[0]:4d}  residuals=[{', '.join(f'{v:4d}' for v in row[1:])}]")

    # Print last few frames if enough data
    if num_frames > n_show:
        n_tail = min(50, num_frames - n_show)
        print(f"\n[LAST {n_tail} FRAMES]")
        for i in range(num_frames - n_tail, num_frames):
            row = codes[i]
            print(f"  frame={i:3d}  primary={row[0]:4d}  residuals=[{', '.join(f'{v:4d}' for v in row[1:])}]")

    # Check for EOS token (2150)
    eos_positions = np.where(codes[:, 0] == 2150)[0]
    if len(eos_positions) > 0:
        print(f"\n[EOS] codec_eos_token_id=2150 found at frame(s): {eos_positions.tolist()}")
    else:
        print(f"\n[EOS] codec_eos_token_id=2150 not found in primary codes")

    print()


if __name__ == "__main__":
    main()
