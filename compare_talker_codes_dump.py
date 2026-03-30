import argparse
from pathlib import Path

import numpy as np


def load_codes(path: Path) -> np.ndarray:
    data = np.load(path)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D codes array in {path}, got shape {data.shape}")
    return data


def format_row(row: np.ndarray, limit: int = 16) -> str:
    clipped = row[:limit].tolist()
    text = ", ".join(str(int(value)) for value in clipped)
    if row.shape[0] > limit:
        text += ", ..."
    return f"[{text}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Python and C++ talker codec frame dumps")
    parser.add_argument(
        "--python-codes",
        default="debug_output/python_sample_0_codes.npy",
        help="Path to the Python talker codes .npy file",
    )
    parser.add_argument(
        "--cpp-codes",
        default="debug_output/cpp_sample_0_codes.npy",
        help="Path to the C++ talker codes .npy file",
    )
    parser.add_argument(
        "--show-limit",
        type=int,
        default=16,
        help="How many mismatches to print",
    )
    args = parser.parse_args()

    python_codes = load_codes(Path(args.python_codes))
    cpp_codes = load_codes(Path(args.cpp_codes))

    print(f"python_shape={python_codes.shape}")
    print(f"cpp_shape={cpp_codes.shape}")

    if python_codes.shape[0] > 0:
        print(f"python_frame0={format_row(python_codes[0])}")
    if cpp_codes.shape[0] > 0:
        print(f"cpp_frame0={format_row(cpp_codes[0])}")

    min_frames = min(python_codes.shape[0], cpp_codes.shape[0])
    min_codebooks = min(python_codes.shape[1], cpp_codes.shape[1])

    common_python = python_codes[:min_frames, :min_codebooks]
    common_cpp = cpp_codes[:min_frames, :min_codebooks]
    equal = np.array_equal(common_python, common_cpp)
    print(f"common_equal={equal}")
    print(f"python_extra_frames={python_codes.shape[0] - min_frames}")
    print(f"cpp_extra_frames={cpp_codes.shape[0] - min_frames}")
    if python_codes.shape[0] != cpp_codes.shape[0]:
        print("note=frame count differs, compare frame0 first before tuning sampling params")

    mismatch_indices = np.argwhere(common_python != common_cpp)
    print(f"mismatch_count={len(mismatch_indices)}")
    for frame_idx, codebook_idx in mismatch_indices[: args.show_limit]:
        print(
            f"mismatch frame={int(frame_idx)} codebook={int(codebook_idx)} "
            f"python={int(common_python[frame_idx, codebook_idx])} "
            f"cpp={int(common_cpp[frame_idx, codebook_idx])}"
        )

    if min_frames > 0:
        python_code0 = python_codes[:min_frames, 0]
        cpp_code0 = cpp_codes[:min_frames, 0]
        code0_mismatch = np.flatnonzero(python_code0 != cpp_code0)
        print(f"code0_mismatch_count={len(code0_mismatch)}")
        if len(code0_mismatch) > 0:
            first = int(code0_mismatch[0])
            print(
                f"first_code0_mismatch_frame={first} "
                f"python={int(python_code0[first])} cpp={int(cpp_code0[first])}"
            )


if __name__ == "__main__":
    main()