import argparse
import struct
from pathlib import Path

import numpy as np


def load_bf16_tensor3d(path: Path):
    with path.open("rb") as f:
        header = f.read(12)
        if len(header) != 12:
            raise ValueError(f"Invalid header in {path}")
        batch, seq, hidden = struct.unpack("III", header)
        count = batch * seq * hidden
        raw = f.read()
    data = np.frombuffer(raw, dtype=np.uint16)
    if data.size != count:
        raise ValueError(f"Payload size mismatch in {path}: expected {count}, got {data.size}")
    return (batch, seq, hidden), data.reshape(batch, seq, hidden)


def bf16_to_fp32(arr: np.ndarray) -> np.ndarray:
    return (arr.astype(np.uint32) << 16).view(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Compare Python and C++ talker prefill bf16 dumps")
    parser.add_argument(
        "--python-dump",
        default="debug_output/python_talker_prefill_input.bin",
        help="Path to Python bf16 dump",
    )
    parser.add_argument(
        "--cpp-dump",
        default="debug_output/cpp_talker_prefill_input.bin",
        help="Path to C++ bf16 dump",
    )
    parser.add_argument(
        "--show-limit",
        type=int,
        default=8,
        help="How many mismatches to print",
    )
    args = parser.parse_args()

    python_shape, python_bf16 = load_bf16_tensor3d(Path(args.python_dump))
    cpp_shape, cpp_bf16 = load_bf16_tensor3d(Path(args.cpp_dump))

    print(f"python_shape={python_shape}")
    print(f"cpp_shape={cpp_shape}")

    if python_shape != cpp_shape:
        print("shape_match=False")
        return

    print("shape_match=True")

    bit_equal = np.array_equal(python_bf16, cpp_bf16)
    print(f"bit_equal={bit_equal}")

    python_fp32 = bf16_to_fp32(python_bf16)
    cpp_fp32 = bf16_to_fp32(cpp_bf16)
    abs_diff = np.abs(python_fp32 - cpp_fp32)
    max_abs_diff = float(abs_diff.max(initial=0.0))
    print(f"max_abs_diff={max_abs_diff}")

    mismatch_indices = np.argwhere(python_bf16 != cpp_bf16)
    print(f"mismatch_count={len(mismatch_indices)}")
    for idx in mismatch_indices[: args.show_limit]:
        b, s, h = idx.tolist()
        py_bf16 = int(python_bf16[b, s, h])
        cpp_bf16_val = int(cpp_bf16[b, s, h])
        py_fp32 = float(python_fp32[b, s, h])
        cpp_fp32_val = float(cpp_fp32[b, s, h])
        print(
            f"mismatch batch={b} seq={s} hidden={h} "
            f"python_bf16=0x{py_bf16:04x} cpp_bf16=0x{cpp_bf16_val:04x} "
            f"python_fp32={py_fp32} cpp_fp32={cpp_fp32_val}"
        )


if __name__ == "__main__":
    main()