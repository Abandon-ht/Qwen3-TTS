import argparse
import re
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


def collect_layer_indices(directory: Path, prefix: str, suffix: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(prefix)}_layer_(\d+)_" + re.escape(suffix) + r"$")
    layer_indices = []
    for path in directory.glob(f"{prefix}_layer_*_{suffix}"):
        match = pattern.match(path.name)
        if match:
            layer_indices.append(int(match.group(1)))
    return sorted(layer_indices)


def compare_cache_tensor(name: str, python_path: Path, cpp_path: Path) -> None:
    python_shape, python_bf16 = load_bf16_tensor3d(python_path)
    cpp_shape, cpp_bf16 = load_bf16_tensor3d(cpp_path)
    print(f"{name}_python_shape={python_shape}")
    print(f"{name}_cpp_shape={cpp_shape}")
    if python_shape != cpp_shape:
        print(f"{name}_shape_match=False")
        return

    print(f"{name}_shape_match=True")
    python_values = bf16_to_fp32(python_bf16)
    cpp_values = bf16_to_fp32(cpp_bf16)
    abs_diff = np.abs(python_values - cpp_values)
    print(f"{name}_max_abs_diff={float(abs_diff.max(initial=0.0))}")
    print(f"{name}_mean_abs_diff={float(abs_diff.mean())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Python and C++ talker KV cache dumps")
    parser.add_argument("--python-dir", default="debug_output/python_talker_decode")
    parser.add_argument("--cpp-dir", default="debug_output/cpp_talker_decode")
    parser.add_argument("--step", type=int, default=None, help="Decode step index to compare; omit to compare prefill cache")
    args = parser.parse_args()

    prefix = "prefill" if args.step is None else f"step_{args.step:04d}"
    python_dir = Path(args.python_dir)
    cpp_dir = Path(args.cpp_dir)

    python_layers = set(collect_layer_indices(python_dir, prefix, "key_cache.bin"))
    cpp_layers = set(collect_layer_indices(cpp_dir, prefix, "key_cache.bin"))
    common_layers = sorted(python_layers & cpp_layers)
    if not common_layers:
        raise FileNotFoundError(f"No common cache layers found for prefix {prefix}")

    print(f"prefix={prefix}")
    print(f"common_layers={common_layers}")
    for layer_idx in common_layers:
        print(f"===== layer {layer_idx} =====")
        compare_cache_tensor(
            f"layer_{layer_idx:02d}_key_cache",
            python_dir / f"{prefix}_layer_{layer_idx:02d}_key_cache.bin",
            cpp_dir / f"{prefix}_layer_{layer_idx:02d}_key_cache.bin",
        )
        compare_cache_tensor(
            f"layer_{layer_idx:02d}_value_cache",
            python_dir / f"{prefix}_layer_{layer_idx:02d}_value_cache.bin",
            cpp_dir / f"{prefix}_layer_{layer_idx:02d}_value_cache.bin",
        )


if __name__ == "__main__":
    main()
