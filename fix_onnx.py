import onnx
from onnxsim import simplify

# 1. 加载模型
model_path = "qwen3_tts_talker.onnx"
output_path = "qwen3_tts_talker_sim.onnx"

print("Loading model...")
model = onnx.load(model_path)

print("Simplifying model (with no-large-tensor logic if needed)...")
# 注意：onnxsim.simplify 返回的是内存中的模型对象
model_simp, check = simplify(model)

# 2. 保存为外部数据格式 (关键步骤)
# save_as_external_data=True 会将权重拆分到单独的文件
print("Saving with external data...")
onnx.save_model(
    model_simp, 
    output_path, 
    save_as_external_data=True, 
    all_tensors_to_one_file=True, 
    location="qwen3_tts_talker_sim.data"
)
print("Done!")
