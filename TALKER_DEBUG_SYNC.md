# Talker Decode 调试同步文档

## 一、目标

这一轮工作的核心目标，是把 Qwen3-TTS 的 talker 路径单独拆出来，专门做 Python 与 C++ 的一致性验证。

原始需求是：

1. Python 侧已经能 dump 出 talker prefill 输入 embedding，用于 C++ 做单次 prefill 验证。
2. 还需要把 Python 侧的 talker decode 结果也 dump 出来，因为 talker decode 与 code predictor 是耦合的。
3. C++ 侧需要能够验证 talker decode，但不能被 C++ 自己的 code predictor 实现差异干扰。

一句话概括：

这轮工作的目标是验证 talker 本身，而不是验证整条 code predictor + talker 联合链路。


## 二、核心推理流程

Python 侧的 talker / code 推理流程如下：

1. talker 根据当前 hidden state 输出 logits。
2. 从这些 logits 中采样出 `code_0`。
3. code predictor 以 `past_hidden + code_0 embedding` 为条件，继续生成 `code_1 .. code_15`。
4. `code_0 .. code_15` 对应的 embedding 会被融合成一个单步输入 embedding。
5. 这个 fused embedding，再加上 `trailing_text_hidden` 或 `tts_pad_embed`，会被回灌给 talker，作为下一步 decode 输入。

重要结论：

talker 下一步 decode 吃的不是一个离散 token，而是由整帧 codec 生成并融合出来的 embedding。


## 三、本轮增加了什么能力

### 1. Python 侧 dump 能力

Python 模型现在已经可以从真实生成路径中 dump 出 talker debug 工件。

当前 Python 可以导出的内容包括：

1. prefill 输入 embedding
2. prefill 最后一个 hidden
3. step-0 logits
4. 每一步 decode 的输入 embedding
5. 每一步 decode 的最后 hidden
6. 每一步 decode 的 logits
7. 每一步的 `input_ids`、`codec_ids`、`attention_mask`、`position_ids`、`cache_position`
8. prefill 阶段所有层的 KV cache snapshot
9. 每个 decode step 所有层的 KV cache snapshot

Python 侧涉及改动的文件：

1. `qwen_tts/core/models/modeling_qwen3_tts.py`
2. `examples/test_model_12hz_voice_design.py`
3. `compare_talker_decode_debug.py`
4. `compare_talker_kv_cache_debug.py`


### 2. C++ 侧 replay 能力

C++ 的 debug 入口已经被扩展，可以直接回放 Python dump 出来的 talker decode 输入。

这意味着：

在验证 talker decode 时，C++ 不需要自己重新生成 codec frame，从而避免被 C++ code predictor 干扰。

当前 C++ replay 路径已经支持：

1. 使用 Python 的 prefill 输入 embedding
2. 使用 Python 每一步的 talker 输入 embedding
3. 使用 Python 的 `attention_mask`
4. 使用 Python 的 `cache_position`
5. dump 出 C++ 每一步 talker 输出结果用于比较
6. dump 出 C++ prefill 和每一步的 KV cache snapshot

C++ 侧涉及改动的文件：

1. `src/runner/LLM.cpp`
2. `src/runner/LLM.hpp`
3. `src/debug_codec.cpp`
4. `debug_talker_decode_commands.txt`


## 四、调试产物与工具

### 1. Python 输出目录

Python 侧主要输出目录是：

`debug_output/python_talker_decode/`

其中包括：

1. `prefill_layer_XX_key_cache.bin`
2. `prefill_layer_XX_value_cache.bin`
3. `step_000N_input_embed.bin`
4. `step_000N_last_hidden.bin`
5. `step_000N_logits.npy`
6. `step_000N_attention_mask.npy`
7. `step_000N_cache_position.npy`
8. `step_000N_input_ids.npy`
9. `step_000N_codec_ids.npy`
10. `step_000N_layer_XX_key_cache.bin`
11. `step_000N_layer_XX_value_cache.bin`


### 2. C++ 输出目录

C++ 侧主要输出目录是：

`debug_output/cpp_talker_decode/`

命名规则与 Python 保持一致，便于直接比较。


### 3. 对比脚本

当前主要使用以下对比脚本：

1. `compare_talker_step0_debug.py`
   用于比较 prefill 最后 hidden 和 step-0 logits。
2. `compare_talker_decode_debug.py`
   用于比较每一步 decode 的输入、hidden、logits 和控制张量。
3. `compare_talker_kv_cache_debug.py`
   用于比较 prefill 或任意 decode step 的 layer-wise KV cache。


## 五、实际使用过的命令

### 1. Python 侧生成调试工件

在专用 `qwen3-tts` Conda 环境中运行：

```bash
cd /home/m5stack/Workspace/Qwen3-TTS
conda run -n qwen3-tts python examples/test_model_12hz_voice_design.py
```


### 2. AX650 板端运行 C++ replay

```bash
./debug_codec Qwen3-TTS-12Hz-1.7B-VoiceDesign-AX650 dummy 65 debug_output/python_talker_prefill_input.bin "" debug_output/cpp_sample_0_codes.npy debug_output/python_talker_decode debug_output/cpp_talker_decode
```


### 3. 比较 decode step

```bash
cd /home/m5stack/Workspace/Qwen3-TTS
for i in 0 1 2 3 4 5 6 7; do
  echo "===== step $i ====="
  conda run -n qwen3-tts python compare_talker_decode_debug.py --step "$i" --python-dir debug_output/python_talker_decode --cpp-dir debug_output/cpp_talker_decode
done
```


### 4. 比较 prefill KV cache

```bash
cd /home/m5stack/Workspace/Qwen3-TTS
conda run -n qwen3-tts python compare_talker_kv_cache_debug.py --python-dir debug_output/python_talker_decode --cpp-dir debug_output/cpp_talker_decode
```


### 5. 比较 step-0 KV cache

```bash
cd /home/m5stack/Workspace/Qwen3-TTS
conda run -n qwen3-tts python compare_talker_kv_cache_debug.py --step 0 --python-dir debug_output/python_talker_decode --cpp-dir debug_output/cpp_talker_decode
```


## 六、环境说明

Python 侧验证必须使用专用 Conda 环境 `qwen3-tts`。

这个环境已经验证：

1. 使用的是当前工作区源码
2. 依赖版本与项目预期一致
3. 包括 `transformers 4.57.3` 和相关音频依赖

`base` 环境不适合做当前这套可重复验证。


## 七、当前已经测试了什么

### 1. Prefill 外部输入一致性

Python 的 prefill 输入 embedding 已经被 dump，并在 C++ 中回放。

这一步证明了：

Python 和 C++ 可以从完全相同的 talker prefill 输入 embedding 起步。


### 2. Step-0 hidden / logits 对比

通过 `compare_talker_step0_debug.py` 比较了：

1. prefill 最后 hidden
2. step-0 logits

观察结果：

1. prefill 最后 hidden 并不完全一致
2. step-0 logits 仍然高度相似
3. step-0 argmax 一致

解释：

hidden 的中等偏差，不一定会立刻在最终 vocab 投影上完全爆炸。这不能说明 prefill 正确，只能说明 step-0 logits 还足够接近。


### 3. 去掉 predictor 干扰后的 decode replay

C++ replay 模式已改成直接使用 Python 每一步生成出来的 talker 输入 embedding，而不是让 C++ predictor 自己重新生成 codec frame。

观察结果：

1. 每一步 `input_embed` 都完全一致
2. 但 decode hidden 仍然存在明显差异

解释：

这一步证明问题不是由 C++ code predictor 引起的。


### 4. Decode 控制状态对齐

之后，replay 路径又进一步改为直接读取 Python 的：

1. `attention_mask`
2. `cache_position`

观察结果：

1. `attention_mask_equal=True`
2. `cache_position_equal=True`
3. `input_embed_max_abs_diff=0.0`
4. decode 的 `last_hidden` 仍然从 step 0 开始就不一致

解释：

这一步排除了“外部 decode 控制状态不一致”这个主要怀疑方向。


### 5. Prefill 阶段 KV cache 对比

最关键的一组证据来自 prefill KV cache 的逐层比较。

观察到的模式是：

1. 所有层 shape 都一致
2. 浅层 `key_cache` 比较接近
3. 浅层 `value_cache` 也比较接近
4. 越往深层，差异越明显
5. `value_cache` 的发散明显比 `key_cache` 更严重
6. 在最后几层，`value_cache` 的 mean diff 已经明显偏大


## 八、目前已经证明了什么

截至当前阶段，下面这些结论已经比较可靠。

### 1. 已经证明为真

1. Python 的 decode dump 已经被 C++ replay 正确使用。
2. C++ talker decode replay 已经不再受 C++ code predictor 干扰。
3. Python 和 C++ 每一步 fused talker 输入 embedding 完全一致。
4. Python 和 C++ 每一步 `attention_mask` 完全一致。
5. Python 和 C++ 每一步 `cache_position` 完全一致。
6. Python 和 C++ 的 prefill KV cache 并不一致。
7. 这种发散在 decode 开始之前就已经存在。


### 2. 已经排除或明显降级的方向

1. 问题核心不是“decode token 不知道是什么”。
2. 问题核心不是“code embedding 没有回灌到 talker”。
3. 问题核心不是 C++ code predictor 实现。
4. 问题核心不是简单的 replay 输入不一致。
5. 问题核心不是 decode mask 或 cache_position 不一致。


## 九、为什么当前最怀疑的是 K/V cache

`K/V cache` 在这里的作用，是保存 talker 对历史上下文的内部记忆状态，供后续 autoregressive decode 使用。

更具体地说：

1. prefill 阶段输入整段上下文 embedding
2. 每一层 self-attention 会为这些 token 生成 key 和 value
3. 这些 key/value 会被保存成 cache
4. decode 时不会重算全部历史，而是只输入当前步 embedding，再结合已有 cache 做 attention

所以，decode 的结果不只依赖当前步 `input_embed`，还强依赖 prefill 阶段留下来的 `K/V cache`。

这也解释了一个常见误区：

为什么在还没有对齐所有 replay 输入之前，某些结果看起来也比较接近，但最后还是怀疑 `K/V cache`？

原因是：

1. `step-0 logits` 接近，并不等于 cache 正确
2. hidden 的中小偏差可能不会立刻把 logits 完全打散
3. 但同样的偏差如果进入了 cache，会成为后续所有 decode step 共享的内部状态误差
4. 这种误差会随着层数和步数持续累积

当前已经观测到的现象正好符合这个模式：

1. prefill 最后 hidden 有差异
2. step-0 logits 仍然相似
3. 但 prefill 的 layer-wise KV cache，尤其是深层 `value_cache`，已经逐层发散
4. decode 从 step 0 开始就不再稳定一致

因此，当前怀疑 `K/V cache` 不是因为“名字看起来可疑”，而是因为已经通过实验直接测到：

在相同外部输入和相同 decode 控制状态下，Python 与 C++ 独立生成出来的 prefill cache 已经不一致。

还需要特别说明一点：

当前的 KV cache 对比结果没有被“把 Python cache 注入给 C++”污染。

因为当前 C++ replay 只读取了 Python 的：

1. `input_embed`
2. `attention_mask`
3. `cache_position`

而 Python dump 出来的 `K/V cache` 文件本身并没有被灌回 C++ 作为运行输入。

所以现在对比的是：

1. Python 独立生成的 KV cache
2. C++ 在相同外部条件下独立生成的 KV cache

这意味着当前关于 cache 发散的结论是可信的。


## 十、当前阶段的解释

当前最合理的解释是：

C++ talker prefill 路径已经在数值上偏离了 Python，而且这种偏差会逐层累积。decode 阶段只是继承了一个已经偏掉的 KV cache 状态，所以即使 decode 输入 embedding、attention mask、cache position 都完全一致，decode 输出仍然会从 step 0 开始出错。

最强证据来自 prefill KV cache 的逐层对比模式：

1. 浅层误差较小
2. 越往深层误差越积越大
3. 深层 `value_cache` 发散最明显

这不像单一的地址偏移或 buffer 拷贝错误，更像 prefill 计算路径本身和 Python 不等价。


## 十一、最可能的根本原因候选

### 1. 高优先级怀疑点

1. C++ 的 `PrefillTalkerEmbeddings` 与 Python talker prefill 语义不完全等价。
2. prefill mask 构造或 grouped prefill 执行方式与 Python / Hugging Face 行为存在细微差异。
3. AX / C++ prefill 路径中的某些层计算存在小偏差，并在 20 多层传播后逐层放大。


### 2. 低优先级怀疑点

1. decode cache 写回 offset 错误
   这一项优先级已经下降，因为 prefill cache 在 decode 开始前就已经发散。
2. replay 阶段 decode 控制信息不一致
   这一项已经通过 `attention_mask` 和 `cache_position` 比较基本排除。
3. predictor 生成的 codec 不一致
   这一项已经通过 replay 路径排除。


## 十二、为什么 decode hidden 会错，但 logits 往往还比较像

这与当前观察结果是吻合的。

原因是：

1. hidden state 出现明显差异，不代表 logits 立刻完全崩溃
2. 中小程度 hidden 偏移，往往仍能保留相同 top-1 候选若干步
3. 当 top-1 和 top-2 分数差距变小时，错误 hidden 才会把 argmax 翻转

这已经在后续 step 中看到过：

大多数 step 的 argmax 仍一致，但在某些 top 候选接近的 step 上，C++ 会开始翻转。


## 十三、下一步建议

从当前证据看，下一步最有价值的动作已经不是继续调 replay 输入，而是：

比较 prefill 的每层输出 hidden，而不只是比较 KV cache。

具体建议：

1. dump Python 每层 prefill output hidden
2. dump C++ 每层 prefill output hidden
3. 找出第一层开始出现明显 hidden 发散的位置

这会回答下一个关键问题：

发散是在 layer output 阶段就已经产生，还是只在写入 K/V cache 时引入。


## 十四、当前阶段的最终结论

截至这一阶段，当前工作假设是：

根本问题最可能位于 C++ talker 的 prefill 计算路径，而不在 replay 机制本身，不在 decode 控制张量，也不在 C++ code predictor。

当前 decode mismatch 更像是一个已经在 prefill 阶段形成的 cache 状态偏差的下游后果。