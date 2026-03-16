import torch
import os
from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

def export_12hz_decoder_to_onnx(model_dir: str, output_onnx_path: str):
    print(f"Loading tokenizer from: {model_dir}")
    
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        model_dir,
        attn_implementation="eager" 
    )
    

    decoder_model = tokenizer.model.decoder
    decoder_model.eval()

    batch_size = 1
    num_quantizers = decoder_model.config.num_quantizers
    codes_length = 50
    
    print(f"Dummy input shape: (batch_size={batch_size}, num_quantizers={num_quantizers}, codes_length={codes_length})")
    
    dummy_codes = torch.randint(
        low=0,
        high=decoder_model.config.codebook_size,
        size=(batch_size, num_quantizers, codes_length),
        dtype=torch.long,
        device=tokenizer.device
    )

    dynamic_axes = {
        "codes": {
            0: "batch_size", 
            2: "codes_length"
        },
        "wav": {
            0: "batch_size", 
            2: "audio_length"
        }
    }

    print(f"Exporting model to {output_onnx_path}...")
    
    # 5. 执行 ONNX 导出
    with torch.no_grad():
        torch.onnx.export(
            model=decoder_model,                   # 导出的模型实例
            args=(dummy_codes,),                   # 模型的输入元组
            f=output_onnx_path,                    # 导出的路径
            export_params=True,                    # 权重一并打包到 onnx 文件中
            opset_version=19,                      # 使用 19+ 版本，能很好支持 RoPE 和复杂激活函数
            do_constant_folding=True,              # 对常量进行折叠优化
            input_names=["codes"],                 # 定义输入的变量名
            output_names=["wav"],                  # 定义输出的变量名
            dynamic_axes=dynamic_axes              # 指定哪些维度是动态可变的
        )

    print("ONNX Export completed successfully!")

if __name__ == "__main__":

    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/speech_tokenizer"
    OUTPUT_ONNX = "qwen3_tts_12hz_decoder.onnx"
    
    export_12hz_decoder_to_onnx(MODEL_PATH, OUTPUT_ONNX)