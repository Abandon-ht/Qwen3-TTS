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


def compare_fp32_tensor(name: str, python_values: np.ndarray, cpp_values: np.ndarray) -> None:
    print(f"{name}_python_shape={python_values.shape}")
    print(f"{name}_cpp_shape={cpp_values.shape}")
    if python_values.shape != cpp_values.shape:
        print(f"{name}_shape_match=False")
        return
    print(f"{name}_shape_match=True")
    abs_diff = np.abs(python_values - cpp_values)
    print(f"{name}_max_abs_diff={float(abs_diff.max(initial=0.0))}")
    print(f"{name}_mean_abs_diff={float(abs_diff.mean())}")


def normalize_logits_shape(values: np.ndarray) -> np.ndarray:
    if values.ndim == 3 and values.shape[0] == 1 and values.shape[1] == 1:
        return values.reshape(1, values.shape[-1])
    return values


def compare_int_tensor(name: str, python_values: np.ndarray, cpp_values: np.ndarray) -> None:
    print(f"{name}_python_shape={python_values.shape}")
    print(f"{name}_cpp_shape={cpp_values.shape}")
    if python_values.shape != cpp_values.shape:
        print(f"{name}_shape_match=False")
        return
    print(f"{name}_shape_match=True")
    is_equal = np.array_equal(python_values, cpp_values)
    print(f"{name}_equal={is_equal}")
    if not is_equal:
        mismatch_indices = np.argwhere(python_values != cpp_values)
        print(f"{name}_mismatch_count={len(mismatch_indices)}")
        if len(mismatch_indices) > 0:
            first_index = tuple(int(index) for index in mismatch_indices[0])
            print(
                f"{name}_first_mismatch_index={first_index} "
                f"python={int(python_values[first_index])} cpp={int(cpp_values[first_index])}"
            )


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Python and C++ talker decode step dumps")
    parser.add_argument("--step", type=int, default=0, help="Decode step index to compare")
    parser.add_argument(
        "--python-dir",
        default="debug_output/python_talker_decode",
        help="Directory containing Python decode step dumps",
    )
    parser.add_argument(
        "--cpp-dir",
        default="debug_output/cpp_talker_decode",
        help="Directory containing C++ decode step dumps",
    )
    parser.add_argument("--topk", type=int, default=8, help="How many top logits to print")
    args = parser.parse_args()

    step_prefix = f"step_{args.step:04d}"
    python_dir = Path(args.python_dir)
    cpp_dir = Path(args.cpp_dir)

    python_input_shape, python_input_bf16 = load_bf16_tensor3d(
        require_path(python_dir / f"{step_prefix}_input_embed.bin")
    )
    cpp_input_shape, cpp_input_bf16 = load_bf16_tensor3d(require_path(cpp_dir / f"{step_prefix}_input_embed.bin"))
    print(f"python_input_embed_shape={python_input_shape}")
    print(f"cpp_input_embed_shape={cpp_input_shape}")
    if python_input_shape != cpp_input_shape:
        print("input_embed_shape_match=False")
    else:
        print("input_embed_shape_match=True")
        compare_fp32_tensor("input_embed", bf16_to_fp32(python_input_bf16), bf16_to_fp32(cpp_input_bf16))

    python_hidden_shape, python_hidden_bf16 = load_bf16_tensor3d(
        require_path(python_dir / f"{step_prefix}_last_hidden.bin")
    )
    cpp_hidden_shape, cpp_hidden_bf16 = load_bf16_tensor3d(require_path(cpp_dir / f"{step_prefix}_last_hidden.bin"))
    print(f"python_last_hidden_shape={python_hidden_shape}")
    print(f"cpp_last_hidden_shape={cpp_hidden_shape}")
    if python_hidden_shape != cpp_hidden_shape:
        print("last_hidden_shape_match=False")
    else:
        print("last_hidden_shape_match=True")
        compare_fp32_tensor("last_hidden", bf16_to_fp32(python_hidden_bf16), bf16_to_fp32(cpp_hidden_bf16))

    python_logits = normalize_logits_shape(np.load(require_path(python_dir / f"{step_prefix}_logits.npy")))
    cpp_logits = normalize_logits_shape(np.load(require_path(cpp_dir / f"{step_prefix}_logits.npy")))
    compare_fp32_tensor("logits", python_logits, cpp_logits)
    if python_logits.shape == cpp_logits.shape:
        python_step = python_logits.reshape(-1)
        cpp_step = cpp_logits.reshape(-1)
        python_argmax = int(np.argmax(python_step))
        cpp_argmax = int(np.argmax(cpp_step))
        print(f"python_argmax={python_argmax}")
        print(f"cpp_argmax={cpp_argmax}")
        print(f"argmax_match={python_argmax == cpp_argmax}")
        print(f"python_top{args.topk}={summarize_topk(python_step, args.topk)}")
        print(f"cpp_top{args.topk}={summarize_topk(cpp_step, args.topk)}")

    python_codec_ids_path = python_dir / f"{step_prefix}_codec_ids.npy"
    cpp_codec_ids_path = cpp_dir / f"{step_prefix}_codec_ids.npy"
    if python_codec_ids_path.exists() and cpp_codec_ids_path.exists():
        compare_int_tensor("codec_ids", np.load(python_codec_ids_path), np.load(cpp_codec_ids_path))

    python_input_ids_path = python_dir / f"{step_prefix}_input_ids.npy"
    cpp_input_ids_path = cpp_dir / f"{step_prefix}_input_ids.npy"
    if python_input_ids_path.exists() and cpp_input_ids_path.exists():
        compare_int_tensor("input_ids", np.load(python_input_ids_path), np.load(cpp_input_ids_path))

    python_attention_mask_path = python_dir / f"{step_prefix}_attention_mask.npy"
    cpp_attention_mask_path = cpp_dir / f"{step_prefix}_attention_mask.npy"
    if python_attention_mask_path.exists() and cpp_attention_mask_path.exists():
        compare_int_tensor("attention_mask", np.load(python_attention_mask_path), np.load(cpp_attention_mask_path))

    python_cache_position_path = python_dir / f"{step_prefix}_cache_position.npy"
    cpp_cache_position_path = cpp_dir / f"{step_prefix}_cache_position.npy"
    if python_cache_position_path.exists() and cpp_cache_position_path.exists():
        compare_int_tensor("cache_position", np.load(python_cache_position_path), np.load(cpp_cache_position_path))


if __name__ == "__main__":
    main()
