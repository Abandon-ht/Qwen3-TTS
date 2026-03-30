import argparse
import struct
from pathlib import Path

import numpy as np


def load_bf16_tensor3d(path: Path):
    with path.open("rb") as file_obj:
        header = file_obj.read(12)
        if len(header) != 12:
            raise ValueError(f"Invalid header in {path}")
        batch, seq, hidden = struct.unpack("III", header)
        count = batch * seq * hidden
        raw = file_obj.read()
    data = np.frombuffer(raw, dtype=np.uint16)
    if data.size != count:
        raise ValueError(f"Payload size mismatch in {path}: expected {count}, got {data.size}")
    return (batch, seq, hidden), data.reshape(batch, seq, hidden)


def bf16_to_fp32(arr: np.ndarray) -> np.ndarray:
    return (arr.astype(np.uint32) << 16).view(np.float32)


def summarize_topk(values: np.ndarray, topk: int) -> str:
    top_indices = np.argsort(values)[-topk:][::-1]
    return ", ".join(f"{int(index)}:{float(values[index]):.6f}" for index in top_indices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Python and C++ talker prefill last hidden and step0 logits")
    parser.add_argument(
        "--python-hidden",
        default="debug_output/python_talker_prefill_last_hidden.bin",
        help="Path to Python prefill last hidden bf16 dump",
    )
    parser.add_argument(
        "--cpp-hidden",
        default="debug_output/cpp_talker_prefill_last_hidden.bin",
        help="Path to C++ prefill last hidden bf16 dump",
    )
    parser.add_argument(
        "--python-logits",
        default="debug_output/python_talker_step0_logits.npy",
        help="Path to Python step0 raw logits .npy",
    )
    parser.add_argument(
        "--cpp-logits",
        default="debug_output/cpp_talker_step0_logits.npy",
        help="Path to C++ step0 raw logits .npy",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=8,
        help="How many top logits to print",
    )
    args = parser.parse_args()

    python_hidden_shape, python_hidden_bf16 = load_bf16_tensor3d(Path(args.python_hidden))
    cpp_hidden_shape, cpp_hidden_bf16 = load_bf16_tensor3d(Path(args.cpp_hidden))

    print(f"python_hidden_shape={python_hidden_shape}")
    print(f"cpp_hidden_shape={cpp_hidden_shape}")
    if python_hidden_shape != cpp_hidden_shape:
        print("hidden_shape_match=False")
    else:
        print("hidden_shape_match=True")
        python_hidden = bf16_to_fp32(python_hidden_bf16)
        cpp_hidden = bf16_to_fp32(cpp_hidden_bf16)
        hidden_abs_diff = np.abs(python_hidden - cpp_hidden)
        print(f"hidden_max_abs_diff={float(hidden_abs_diff.max(initial=0.0))}")
        print(f"hidden_mean_abs_diff={float(hidden_abs_diff.mean())}")

    python_logits = np.load(Path(args.python_logits))
    cpp_logits = np.load(Path(args.cpp_logits))

    print(f"python_logits_shape={python_logits.shape}")
    print(f"cpp_logits_shape={cpp_logits.shape}")
    if python_logits.shape != cpp_logits.shape:
        print("logits_shape_match=False")
        return

    print("logits_shape_match=True")
    python_step0 = python_logits.reshape(-1)
    cpp_step0 = cpp_logits.reshape(-1)
    logits_abs_diff = np.abs(python_step0 - cpp_step0)
    print(f"logits_max_abs_diff={float(logits_abs_diff.max(initial=0.0))}")
    print(f"logits_mean_abs_diff={float(logits_abs_diff.mean())}")

    python_argmax = int(np.argmax(python_step0))
    cpp_argmax = int(np.argmax(cpp_step0))
    print(f"python_argmax={python_argmax}")
    print(f"cpp_argmax={cpp_argmax}")
    print(f"argmax_match={python_argmax == cpp_argmax}")

    print(f"python_top{args.topk}={summarize_topk(python_step0, args.topk)}")
    print(f"cpp_top{args.topk}={summarize_topk(cpp_step0, args.topk)}")


if __name__ == "__main__":
    main()