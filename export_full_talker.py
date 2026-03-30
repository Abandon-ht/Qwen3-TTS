"""
导出完整的 Talker 模型 (28 层) + 可选 Code Predictor (5 层)
"""
import torch
import os
from qwen_tts import Qwen3TTSModel


class SimpleTalkerWrapper(torch.nn.Module):
    """
    简化的 Talker 包装器，支持文本输入
    流程：text_ids → text_embedding → text_projection → 28 层 Transformer → norm → 输出
    """
    def __init__(self, talker_model):
        super().__init__()
        self.text_embedding = talker_model.model.text_embedding  # (151936, 2048)
        self.text_projection = talker_model.text_projection  # MLP(2048→2048→2048)
        self.layers = talker_model.model.layers  # 28 层
        self.norm = talker_model.model.norm  # RMSNorm
    
    def forward(self, input_ids):
        # 1. 文本嵌入
        hidden = self.text_embedding(input_ids)  # [B, seq, 2048]
        
        # 2. 文本投影
        hidden = self.text_projection(hidden)  # [B, seq, 2048]
        
        # 3. 28 层 Transformer
        for layer in self.layers:
            layer_outputs = layer(hidden)
            hidden = layer_outputs[0]
        
        # 4. 归一化
        hidden = self.norm(hidden)
        
        return hidden  # [B, seq, 2048]


class FullTalkerWithPredictor(torch.nn.Module):
    """
    完整 Talker + Code Predictor (28+5 层)
    流程：text_ids → text_embedding → 28 层 → norm → codec_head → 输出 logits
    """
    def __init__(self, talker_wrapper):
        super().__init__()
        self.talker_wrapper = talker_wrapper  # SimpleTalkerWrapper
        self.codec_head = talker_wrapper.talker_wrapper.codec_head  # Linear(2048→3072)
    
    def forward(self, input_ids):
        hidden = self.talker_wrapper(input_ids)  # [B, seq, 2048]
        logits = self.codec_head(hidden)  # [B, seq, 3072]
        return logits


def export_simple_talker(model_path: str, output_onnx_path: str):
    """导出简化的 Talker (28 层，文本→hidden)"""
    print("="*60)
    print("加载模型并导出 Talker (28 层)")
    print("="*60)
    
    model_wrapper = Qwen3TTSModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="eager"
    )
    
    # 获取完整的 talker
    full_talker = model_wrapper.model.talker
    
    # 创建简化包装器
    simple_model = SimpleTalkerWrapper(full_talker)
    simple_model.eval()
    
    # 创建输入（使用文本词表）
    batch_size = 1
    text_seq_len = 50
    dummy_input_ids = torch.randint(0, 151936, (batch_size, text_seq_len), dtype=torch.long)
    
    print(f"输入：input_ids {dummy_input_ids.shape} (text_vocab_size=151936)")
    print(f"导出 Talker (28 层)...")
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                model=simple_model,
                args=(dummy_input_ids,),
                f=output_onnx_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input_ids"],
                output_names=["hidden_states"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "seq_len"},
                    "hidden_states": {0: "batch_size", 1: "seq_len"}
                },
                dynamo=False
            )
        
        print(f"\n✅ Talker ONNX 导出成功!")
        print(f"   文件：{os.path.abspath(output_onnx_path)}")
        print(f"   大小：{os.path.getsize(output_onnx_path) / 1024 / 1024:.2f} MB")
        print(f"   层数：28 层 Transformer")
        print(f"   输入：[batch_size, seq_len] int64")
        print(f"   输出：[batch_size, seq_len, 2048] float32")
        return True
    except Exception as e:
        print(f"\n❌ 导出失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def export_full_talker_with_predictor(model_path: str, output_onnx_path: str):
    """导出完整 Talker + Code Predictor (28+5 层)"""
    print("="*60)
    print("加载模型并导出完整 Talker (28+5 层)")
    print("="*60)
    
    model_wrapper = Qwen3TTSModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="eager"
    )
    
    full_talker = model_wrapper.model.talker
    
    # 先创建简化 Talker
    simple_talker = SimpleTalkerWrapper(full_talker)
    simple_talker.eval()
    
    # 再包装 Code Predictor
    class PredictorWrapper(torch.nn.Module):
        def __init__(self, simple_talker, code_predictor):
            super().__init__()
            self.simple_talker = simple_talker
            self.code_predictor = code_predictor
        
        def forward(self, input_ids):
            # Talker 输出
            hidden = self.simple_talker(input_ids)  # [B, seq, 2048]
            
            # Code Predictor 处理（简化版，只返回 logits）
            # 注意：Code Predictor 实际用于生成多个 codec tokens
            # 这里只导出简单的线性映射
            logits = self.code_predictor.model.norm(hidden)  # 简化处理
            return logits
    
    # 直接导出 Talker + codec_head
    class TalkerWithHead(torch.nn.Module):
        def __init__(self, simple_talker, codec_head):
            super().__init__()
            self.simple_talker = simple_talker
            self.codec_head = codec_head
        
        def forward(self, input_ids):
            hidden = self.simple_talker(input_ids)
            logits = self.codec_head(hidden)
            return logits
    
    model_with_head = TalkerWithHead(simple_talker, full_talker.codec_head)
    model_with_head.eval()
    
    dummy_input_ids = torch.randint(0, 151936, (1, 50), dtype=torch.long)
    
    print(f"输入：input_ids {dummy_input_ids.shape}")
    print(f"导出 Talker + codec_head (28 层 + 输出层)...")
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                model=model_with_head,
                args=(dummy_input_ids,),
                f=output_onnx_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input_ids"],
                output_names=["codec_logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "seq_len"},
                    "codec_logits": {0: "batch_size", 1: "seq_len"}
                },
                dynamo=False
            )
        
        print(f"\n✅ 完整 Talker ONNX 导出成功!")
        print(f"   文件：{os.path.abspath(output_onnx_path)}")
        print(f"   大小：{os.path.getsize(output_onnx_path) / 1024 / 1024:.2f} MB")
        print(f"   层数：28 层 Transformer + codec_head")
        print(f"   输入：[batch_size, seq_len] int64")
        print(f"   输出：[batch_size, seq_len, 3072] float32")
        return True
    except Exception as e:
        print(f"\n❌ 导出失败：{e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="导出 Qwen3-TTS Talker 为 ONNX 格式")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        help="模型路径或 HuggingFace 模型 ID"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="qwen3_tts_talker_28layers.onnx",
        help="输出 ONNX 文件路径"
    )
    parser.add_argument(
        "--with_head",
        action="store_true",
        help="添加 codec_head 输出层"
    )
    
    args = parser.parse_args()
    
    if args.with_head:
        success = export_full_talker_with_predictor(args.model_path, args.output)
    else:
        success = export_simple_talker(args.model_path, args.output)
    
    if success:
        print("\n" + "="*60)
        print("下一步：使用 Netron 查看模型")
        print("="*60)
        print("""
方法 1: 使用 Netron 桌面应用
  1. 下载 Netron: https://github.com/lutzroeder/netron
  2. 打开 Netron，选择导出的 .onnx 文件

方法 2: 使用 Netron Web
  1. 访问：https://netron.app/
  2. 上传导出的 .onnx 文件

方法 3: 使用命令行启动 Netron
  pip install netron
  netron qwen3_tts_talker_28layers.onnx
  
  然后在浏览器中打开 http://localhost:8080
""")