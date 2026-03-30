"""
导出 Qwen3-TTS 模型为 ONNX 格式，然后用 Netron 可视化
"""
import torch
import onnx
import os
from qwen_tts import Qwen3TTSModel

class TalkerWrapper(torch.nn.Module):
    """包装器，修复 embed_tokens 属性问题"""
    def __init__(self, talker_model):
        super().__init__()
        self.talker_model = talker_model
        # 添加 embed_tokens 别名，指向 codec_embedding
        self.embed_tokens = talker_model.codec_embedding
    
    def forward(self, input_ids):
        # 使用 codec_embedding 处理输入
        inputs_embeds = self.embed_tokens(input_ids)
        outputs = self.talker_model.forward(
            input_ids=None,
            inputs_embeds=inputs_embeds
        )
        return outputs.last_hidden_state


def export_talker_only(model_path: str, output_onnx_path: str):
    """只导出 Talker 部分（较小）"""
    print("="*60)
    print("加载模型并导出 Talker 部分...")
    print("="*60)
    
    model_wrapper = Qwen3TTSModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="eager"
    )
    
    # 包装 talker 模型
    talker_model = model_wrapper.model.talker.model
    talker_model.eval()
    
    wrapped_model = TalkerWrapper(talker_model)
    wrapped_model.eval()
    
    dummy_input_ids = torch.randint(0, 3072, (1, 50), dtype=torch.long)
    
    print(f"导出 Talker Model...")
    print(f"  - 输入形状：{dummy_input_ids.shape}")
    print(f"  - codec_embedding: {talker_model.codec_embedding}")
    
    # 使用临时文件导出，然后合并为单个文件
    temp_onnx_path = output_onnx_path.replace(".onnx", "_temp.onnx")
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                model=wrapped_model,
                args=(dummy_input_ids,),
                f=temp_onnx_path,
                export_params=True,
                opset_version=19,
                do_constant_folding=True,
                input_names=["input_ids"],
                output_names=["hidden_states"],
                dynamo=False,
                # 关键参数：确保权重内联到单个文件中
                keep_initializers_as_inputs=False,
            )
        
        # 加载并重新保存，确保所有权重内联到单个文件中
        onnx_model = onnx.load(temp_onnx_path)
        
        # 检查是否有外部数据引用
        has_external_data = any(
            node.HasField("data_location") and 
            node.data_location == onnx.TensorProto.EXTERNAL
            for tensor in onnx_model.graph.initializer
        )
        
        if has_external_data:
            print("  - 检测到外部数据引用，正在合并到单个文件...")
            # 使用 onnx.save_model 确保所有数据内联
            onnx.save_model(
                onnx_model, 
                output_onnx_path, 
                save_as_external_data=False,
                all_tensors_to_one_file=True
            )
            # 删除临时文件
            if os.path.exists(temp_onnx_path):
                os.remove(temp_onnx_path)
            # 清理可能生成的外部数据文件
            base_name = temp_onnx_path.replace(".onnx", "")
            for ext in [".data", "_data", ""]:
                data_file = base_name + ext + ".data"
                if os.path.exists(data_file):
                    os.remove(data_file)
        else:
            # 没有外部数据，直接重命名
            os.rename(temp_onnx_path, output_onnx_path)
        
        print(f"\n✅ Talker ONNX 导出成功!")
        print(f"   文件：{os.path.abspath(output_onnx_path)}")
        print(f"   大小：{os.path.getsize(output_onnx_path) / 1024 / 1024:.2f} MB")
        
        # 验证文件完整性
        try:
            onnx.checker.check_model(onnx.load(output_onnx_path))
            print(f"   验证：✅ ONNX 模型格式有效")
        except Exception as e:
            print(f"   验证：⚠️  模型验证警告：{e}")
        
        return True
    except Exception as e:
        print(f"\n❌ 导出失败：{e}")
        import traceback
        traceback.print_exc()
        # 清理临时文件
        if os.path.exists(temp_onnx_path):
            os.remove(temp_onnx_path)
        return False


def export_simple_graph(model_path: str, output_onnx_path: str):
    """导出简化的模型图（只包含主要结构）"""
    print("="*60)
    print("加载模型并导出简化结构...")
    print("="*60)
    
    model_wrapper = Qwen3TTSModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="eager"
    )
    
    # 只导出 codec_embedding + 第一层 + norm（用于展示结构）
    talker_model = model_wrapper.model.talker.model
    talker_model.eval()
    
    class SimpleTalker(torch.nn.Module):
        def __init__(self, talker_model):
            super().__init__()
            self.codec_embedding = talker_model.codec_embedding
            self.text_embedding = talker_model.text_embedding
            self.first_layer = talker_model.layers[0]
            self.norm = talker_model.norm
        
        def forward(self, input_ids):
            hidden = self.codec_embedding(input_ids)
            hidden = self.first_layer(hidden)[0]
            hidden = self.norm(hidden)
            return hidden
    
    simple_model = SimpleTalker(talker_model)
    simple_model.eval()
    
    dummy_input_ids = torch.randint(0, 3072, (1, 20), dtype=torch.long)
    
    print(f"导出简化 Talker 结构...")
    
    # 使用临时文件导出，然后合并为单个文件
    temp_onnx_path = output_onnx_path.replace(".onnx", "_temp.onnx")
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                model=simple_model,
                args=(dummy_input_ids,),
                f=temp_onnx_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input_ids"],
                output_names=["hidden_states"],
                dynamo=False,
                keep_initializers_as_inputs=False,
            )
        
        # 加载并重新保存，确保所有权重内联到单个文件中
        onnx_model = onnx.load(temp_onnx_path)
        
        # 使用 onnx.save_model 确保所有数据内联
        onnx.save_model(
            onnx_model, 
            output_onnx_path, 
            save_as_external_data=False,
            all_tensors_to_one_file=True
        )
        
        # 删除临时文件
        if os.path.exists(temp_onnx_path):
            os.remove(temp_onnx_path)
        
        # 清理可能生成的外部数据文件
        base_name = temp_onnx_path.replace(".onnx", "")
        for ext in [".data", "_data", ""]:
            data_file = base_name + ext + ".data"
            if os.path.exists(data_file):
                os.remove(data_file)
        
        print(f"\n✅ 简化结构导出成功!")
        print(f"   文件：{os.path.abspath(output_onnx_path)}")
        print(f"   大小：{os.path.getsize(output_onnx_path) / 1024 / 1024:.2f} MB")
        
        # 验证文件完整性
        try:
            onnx.checker.check_model(onnx.load(output_onnx_path))
            print(f"   验证：✅ ONNX 模型格式有效")
        except Exception as e:
            print(f"   验证：⚠️  模型验证警告：{e}")
        
        return True
    except Exception as e:
        print(f"\n❌ 导出失败：{e}")
        import traceback
        traceback.print_exc()
        # 清理临时文件
        if os.path.exists(temp_onnx_path):
            os.remove(temp_onnx_path)
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="导出 Qwen3-TTS 为 ONNX 格式")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        help="模型路径或 HuggingFace 模型 ID"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="qwen3_tts_talker.onnx",
        help="输出 ONNX 文件路径"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="导出简化结构（只包含部分层，更快）"
    )
    
    args = parser.parse_args()
    
    if args.simple:
        success = export_simple_graph(args.model_path, args.output)
    else:
        success = export_talker_only(args.model_path, args.output)
    
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
  netron qwen3_tts_talker.onnx
  
  然后在浏览器中打开 http://localhost:8080
""")