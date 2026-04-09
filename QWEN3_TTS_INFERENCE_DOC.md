# Qwen3-TTS 模型推理架构与 C++ 移植指南

本文档总结了 Qwen3-TTS（特别是 Voice Design 模式及 12Hz V2 Tokenizer）的推理架构。详细记录了核心生成网络 Talker 与 Code Predictor 的协作机制、KV Cache 的管理方式，并附带了关键的 Python 参考代码，旨在为后续在其他平台（如 C++ / ONNX / TensorRT 等）的推理移植提供核心参考。

## 一、 整体推理管线 (Voice Design 模式)

在 Voice Design 模式下（无参考音频输入），系统的推理管线由以下核心模块组成：

1. **Text Processor (Tokenizer)**：将自然语言（文本与声音设计指令）转换为离散的 Token IDs (`input_ids`, `instruct_ids`)。
2. **Talker (Main LLM)**：处理文本和指令上下文，并在时间轴上自回归生成第一层音频码（Code 0，决定主语义与基音）。
3. **Code Predictor (Sub LLM)**：在每一个时间步内，根据 Talker 的状态，在深度轴上自回归生成剩余的音频残差码（Code 1 ~ N-1，决定音色细节与高频特征）。
4. **Speech Decoder (Codec / Qwen3TTSTokenizerV2Decoder)**：将生成的离散多层音频码（Audio Codes）上采样并解码为 24kHz 的 PCM 连续音频波形。

---

## 二、 核心协作机制：Talker 与 Code Predictor

整个生成过程采用**分层多码本预测 (Hierarchical Multi-Codebook Prediction)**，也称作 Delayed-Pattern 或 MTP 机制。

### 1. 结构嵌套与分工
* **Talker (`Qwen3TTSTalkerForConditionalGeneration`)**：维护**时序全局 KV Cache**。每生成一帧（时间步推移），序列长度 +1。
* **Code Predictor (`Qwen3TTSTalkerCodePredictorModelForConditionalGeneration`)**：维护**帧内局部 KV Cache**。在每个时间步内，从长度 2 开始增长至 `num_code_groups`。进入下一帧时，该 Cache 丢弃或重置。

### 2. 协作流程图

```text
[时间步 t 走向 t+1 的完整生命周期]

============== (1) 全局自回归 (Talker) =================
        [ 上一帧的综合特征向量 inputs_embeds ]
                        |
            +-----------------------+
            |      Talker (LLM)     | ---> 读取/更新时序全局 KV Cache
            +-----------------------+
                        |
            [ 隐状态 H_t (past_hidden) ]
                        |
                 (线性层分类头)
                        |
                 预测出 Code_t_c0
                        |
        +---------------+---------------+
        |                               |
    (作为下一帧的特征之一)        (去查表得到 last_id_hidden)

============== (2) 帧内细节补齐 (Code Predictor) ========
        |                               |
    H_t (降维/投射)               last_id_hidden
        |                               |
        +---------------+---------------+
                        |
                 (拼接: shape=[B, 2, D])
                        |
            +-----------------------+
            |  Code Predictor (LLM) | ---> 创建/更新帧内局部 KV Cache
            +-----------------------+
                        |
         (多层独立的分类头, 循环预测 c1 到 c15)
                        |
              预测出 Code_t_c1 ~ c15

============== (3) 特征融合走向下一帧 ===================
        |                               |
   Code_t_c0 的 Embed            Code_t_c1~15 的 Embeds
        |                               |
        +---------------+---------------+
                        |
                      (Sum 求和)
                        |
        [ 这一帧合并后的综合特征向量 inputs_embeds ]
                        |
             (进入下一个时间步 t+1 ...)
```

---

## 三、 C++ 移植核心参考代码

以下是 `Qwen3TTSTalkerForConditionalGeneration` 的 `forward` 函数中的核心代码段。C++ 实现时必须严格对齐这一段的逻辑（包含 Prefill 和 Decode 的分支，以及特征的 Sum 操作）：

```python
# 代码位置: qwen_tts/core/models/modeling_qwen3_tts.py
# 类: Qwen3TTSTalkerForConditionalGeneration

def forward(self, input_ids=None, inputs_embeds=None, past_hidden=None, ...):
    
    # 【分支 1】: Prefill (预填充阶段)
    # 输入的是一段很长的 prompt (文本+指令)，此时主要是填满 Talker 的 KV Cache
    if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
        generation_step = -1
        codec_ids = None

    # 【分支 2】: Decode (自回归解码阶段)
    else:
        # 1. 拿到刚才 Talker 预测的 Code 0 的 Embedding
        last_id_hidden = self.get_input_embeddings()(input_ids)
        
        # 2. 启动 Code Predictor，预测剩余的残差码
        # 注意输入是拼接了 past_hidden (Talker在t时刻的输出) 和 last_id_hidden (Code 0的Embedding)
        predictor_result = self.code_predictor.generate(
            inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
            max_new_tokens=self.config.num_code_groups - 1, # 预测剩余的码本
            do_sample=subtalker_dosample,
            # ... 各种采样参数
        )
        
        # 3. 收集这一帧完整的音频码: [Code_0, Code_1, ..., Code_15]
        codec_ids = torch.cat((input_ids, predictor_result.sequences), dim=-1)
        
        # 4. 把所有的码各自做 Embedding (注意使用 Predictor 中各自独立的 Embedding 表)
        codec_hiddens = torch.cat(
            [last_id_hidden]
            + [self.code_predictor.get_input_embeddings()[i](predictor_result.sequences[..., i:i+1]) 
               for i in range(self.config.num_code_groups - 1)],
            dim=1,
        )
        
        # 5. 特征融合：将多层码的 Embedding 沿维度 1 相加 (Sum)
        inputs_embeds = codec_hiddens.sum(1, keepdim=True)

        # 处理 padding/trailing 逻辑 (对齐时间轴)
        if generation_step < trailing_text_hidden.shape[1]:
            inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1)
        else:
            inputs_embeds = inputs_embeds + tts_pad_embed

    # 【Talker 自身的 Transformer 前向传播】
    # 将更新好的 inputs_embeds 送入 Talker，计算出用于预测下一帧 Code 0 的 Hidden States
    outputs = self.model(
        inputs_embeds=inputs_embeds,
        past_key_values=past_key_values, # 全局时序 KV Cache
        # ...
    )

    hidden_states = outputs.last_hidden_state
    
    # 通过 Linear 头预测下一帧的 Code 0
    logits = self.codec_head(hidden_states)

    return Qwen3TTSTalkerOutputWithPast(
        logits=logits,
        past_key_values=outputs.past_key_values,
        past_hidden=hidden_states[:, -1:, :], # 传递给下一步的 past_hidden
        # ...
    )
```

---

## 四、 C++ 移植注意事项 (Checklist)

1. **双重 KV Cache 机制：**
   * **Talker Cache**：在 C++ 中需要维护一个长期存在的 KV Cache (Context Length 随生成的音频帧数线性增加)。
   * **Predictor Cache**：C++ 中需要为子模型分配一个小型的 KV Cache（Context Length 固定为 `num_code_groups`，如 16）。**必须在 Talker 每步解码后清空/重置该 Cache**。
2. **多套 Embedding 权重：**
   * Talker 有一套 `codec_embedding`。
   * Predictor 有一个 `nn.ModuleList` 的 `codec_embedding`（共 15 个独立查找表，分别对应 Code 1 到 Code 15）。查表时切勿混淆索引。
3. **多套 LM Head 权重：**
   * Talker 有一个 `codec_head`。
   * Predictor 同样有一个 `nn.ModuleList` 的 `lm_head`（共 15 个线性层）。在帧内循环预测时，第 `i` 步要使用 `lm_head[i]`。
4. **小模型维度投射：**
   * 注意 `small_to_mtp_projection` (Linear)。在将特征拼接送入 Predictor 之前，如果有维度差异，可能会经过一层投射运算。
5. **采样逻辑：**
   * C++ 的 Sampler 需要支持 `Talker` (主模型) 和 `Predictor` (子模型) 采用不同的采样参数 (`top_k`, `top_p`, `temperature`)。通常子模型生成细节时可以容忍更高的 randomness。

---

## 五、 “Code_t_c” 是什么？（离散 Token 解析）

在理解模型协作流程之前，必须明确 **`Code_t_c`** 到底是什么：
* 它是一个 **离散的单个整数 Token ID**（比如数字 `452`，范围通常在 `[0, vocab_size-1]` 之间）。
* **`t`** 代表**时间步 (Time step)**，也就是音频的第几帧。在 12Hz 架构下，每一个时间步代表 1/12 秒的音频片段。
* **`c`** 代表**码本层级 (Codebook Index)**。由于高保真音频单靠一个 Token 无法表达丰富的细节（基音、音色、齿音、环境音），神经音频编解码器（如 EnCodec）引入了“残差向量量化 (RVQ)”。每一帧音频不是 1 个 Token，而是 `N` 个 Token（比如 `N=16`）。
  * **`Code_t_c0`**（第 0 层）：决定主要语义和基音（由主模型 Talker 预测）。
  * **`Code_t_c1 ~ c15`**（第 1~15 层）：决定音色和高频细节残差（由子模型 Code Predictor 逐个预测）。

所以，`Code_t0_c0` 就是：**第 0 帧、第 0 层码本的一个离散整数 Token。**

---

## 六、 详细流程图解析

### 1. 宏观协作流程图：Talker 与 Code Predictor 如何交替工作

下面展示的是生成连续两帧（第 `t` 帧和第 `t+1` 帧）时，两个模型是如何接力、交替运行的。

```text
======================= 时间步 t (第 t 帧) =======================

[Talker 的输入上下文 H_{t-1} 等]
        |
+-------------------------------------------------------------+
| 1. Talker (主模型) 前向传播                                 |
|    - 输入: 上一帧融合的 Embeddings                          |
|    - 行为: 延长全局 KV Cache (长度 +1)                      |
|    - 输出: 隐状态 H_t                                       |
+-------------------------------------------------------------+
        |
        v
 (线性分类头 codec_head)
        |
        v
[ 预测出单一整数 Token: Code_t_c0 ] (如: 843)
        |
        +-----------------------------------+
                                            |
                                            v
                +-------------------------------------------------------------+
                | 2. Code Predictor (子模型) 帧内循环预测                     |
                |    - 初始化: 拼接 [H_t, Embed(Code_t_c0)] 作为初始输入      |
                |    - 行为: 开始帧内局部 KV Cache (长度从 2 涨到 N)          |
                |-------------------------------------------------------------|
                |  循环 i 从 1 到 15:                                         |
                |    (a) 子模型前向传播                                       |
                |    (b) 使用独立的线性分类头 lm_head[i]                      |
                |    (c) 预测出 [Code_t_ci]                                   |
                |    (d) 将 [Code_t_ci] 的 Embed 塞回，更新局部 KV Cache      |
                +-------------------------------------------------------------+
                                            |
                                            v
        <--- 收集到本帧完整的残差码 [Code_t_c1, c2, ..., c15] ---+
        |
+-------------------------------------------------------------+
| 3. 帧结束特征融合 (准备迈向下一帧)                          |
|    - 动作: 取 Code_t_c0 和所有的 Code_t_c1~c15，分别查各自  |
|            独立的 Embedding 表。                            |
|    - 融合: 将这 16 个特征向量沿层级维度求和 (Sum)           |
|            Inputs_Embeds_t = Sum( Embed(Code_t_c0..c15) )   |
+-------------------------------------------------------------+
        |
        v
======================= 时间步 t+1 (第 t+1 帧) =====================
        |
+-------------------------------------------------------------+
| 1. Talker (主模型) 前向传播                                 |
|    - 输入: Inputs_Embeds_t (刚才融合的特征)                 |
|    - 行为: 继续延长全局 KV Cache (长度 +1)                  |
|    - 输出: 隐状态 H_{t+1}                                   |
+-------------------------------------------------------------+
        |
        v
(继续预测 Code_{t+1}_c0 ...)
```

---

### 2. Talker (主模型) 内部数据流图

在这个截面中，我们只看 Talker 在一步（一步 = 生成一个时间步的主码）中发生了什么。

```text
[ Talker 模型内部流程 (Qwen3TTSTalkerModel) ]

输入: Inputs_Embeds (Shape: [Batch, 1, Hidden_Size])
      (来自上一帧 16 个 Code Embeddings 的求和结果)

  +--> [ 加上绝对/相对位置编码 (Rotary Position Embedding) ]
  |
  v
+---------------------------------------------------+
|                  Transformer Blocks               |
|  (循环 N 层, 例如 24 层)                          |
|                                                   |
|  1. Attention 层                                  |
|     - Query = W_q * X                             |
|     - Key   = W_k * X  -> [存入时序全局 KV Cache] |
|     - Value = W_v * X  -> [存入时序全局 KV Cache] |
|     - 出发: Attention(Q, K_cache, V_cache)        |
|                                                   |
|  2. FFN 层 (MLP / SwiGLU)                         |
+---------------------------------------------------+
  |
  v
[ 隐状态 Hidden_State (Shape: [Batch, 1, Hidden_Size]) ]
  |
  |---(分支 A: 被保存为 past_hidden, 等下喂给子模型)
  |
  v
[ 线性层 codec_head (Shape: [Hidden_Size, Vocab_Size]) ]
  |
  v
[ Logits (Shape: [Batch, Vocab_Size]) ]
  |
  v
[ 采样策略 (Top-K / Top-P / Temperature) ]
  |
  v
[ 输出: 离散 Token ID (Code_0) ]
```

---

### 3. Code Predictor (子模型) 内部数据流图

在这个截面中，我们看子模型如何在一个时间步（比如一帧）之内，快速循环 15 次吐出所有残差码。

```text
[ Code Predictor 内部流程 (Qwen3TTSTalkerCodePredictorModel) ]

[ 初始状态: 第 0 步 (Prefill) ]
输入特征: Concat [ 
           Talker的 past_hidden (作为条件), 
           Code_0 的 Embedding (作为初始起点) 
         ]
         (Shape: [Batch, 2, Sub_Hidden_Size]) 
         (注: 若主子模型维度不同，此处会有 Linear 降维投射)

  +--> [ 存入帧内局部 KV Cache (长度为 2) ]
  |
  v
+---------------------------------------------------+
|                  Transformer Blocks               |
|  (通常层数较少，例如 6 层)                        |
|  - 前向传播，输出序列最后的 Hidden_State          |
+---------------------------------------------------+
  |
  v
[ 隐状态 Hidden_State_step_1 ]
  |
  v
[ 使用专属分类头: lm_head[0] (对应第 1 层码) ]  <-- 注意: 每次循环用的 head 不同
  |
  v
[ 采样输出: 离散 Token ID (Code_1) ]

================ 循环 14 次 (Decode 阶段) ================
循环第 i 步 (i = 2...15):
  
  将刚才生成的 Code_{i-1} 查对应的专属 Embedding 表
  输入特征: Embed[i-1] (Code_{i-1})  (Shape: [Batch, 1, Sub_Hidden_Size])

  +--> [ 仅用当前 Token 算 K, V, 追加到局部 KV Cache 中 (长度 +1) ]
  |
  v
+---------------------------------------------------+
|                  Transformer Blocks               |
|  - 仅利用 KV Cache 增量计算                       |
+---------------------------------------------------+
  |
  v
[ 隐状态 Hidden_State_step_i ]
  |
  v
[ 使用专属分类头: lm_head[i-1] ]
  |
  v
[ 采样输出: 离散 Token ID (Code_i) ]

================ 循环结束 ==============================

[ 输出收集完毕: Code_1 到 Code_15 ]
(注: 此时，它的局部 KV Cache 使命结束，等待下一个时间步重置)
```
````
This is the description of what the code block changes:
<changeDescription>
Clarify the input to Code Predictor: both past_hidden and the embedding of the predicted Code_0 token are concatenated
</changeDescription>

This is the code block that represents the suggested code change:
````markdown
## 七、 关键细节澄清：Code Predictor 的输入是什么？

针对你的疑问，我在这里详细澄清 **Code Predictor (子模型)** 的输入组成：

### 1. 输入的两个部分
喂给 Code Predictor 的不是单一的 `past_hidden`，而是 **两个向量拼接在一起**：
* **第一部分：`past_hidden`**（Talker 的隐状态输出）
  * 来源：Talker 模型在当前时间步（第 `t` 帧）经过 Transformer 前向传播后的最后隐状态。
  * 形状：`[Batch, 1, Hidden_Size]`（通常 `Hidden_Size = 2048`）。
  * 作用：作为“上下文条件”，告诉子模型当前帧的整体语义和音色倾向（比如“这一帧应该是个高音 A”）。
* **第二部分：`last_id_hidden`**（Talker 预测的 Code_0 Token 的 Embedding）
  * 来源：Talker 刚刚通过 `codec_head` 预测出的离散 Token ID（Code_t_c0），然后经过 Talker 的 `get_input_embeddings()` 查表得到的向量。
  * 形状：`[Batch, 1, Hidden_Size]`。
  * 作用：作为“起点种子”，告诉子模型从这个具体的音频码开始，预测后续的残差细节。

### 2. 拼接操作
是的，这两个向量 **必须拼接在一起**（Concatenation），形成一个长度为 2 的序列：
```python
inputs_embeds = torch.cat((past_hidden, last_id_hidden), dim=1)  # Shape: [Batch, 2, Hidden_Size]
```
* **拼接维度**：`dim=1`（序列长度维度），所以从 `[Batch, 1, Hidden_Size]` 变成 `[Batch, 2, Hidden_Size]`。
* **为什么拼接？** 因为 Code Predictor 是一个标准的 Transformer 自回归模型，它需要一个“序列”作为输入（至少长度 2）。`past_hidden` 作为条件上下文，`last_id_hidden` 作为第一个“Token”嵌入。

### 3. 完整流程回顾
```text
[Talker 输出 Code_t_c0]
        |
        v
[ 查 Talker 的 Embedding 表 -> last_id_hidden ]
        |
        +-------------------+
                            |
[ past_hidden ] ----------> |
                            |
        +-------------------+
                |
        [ Concat: [past_hidden, last_id_hidden] ]
                |
        [ 喂给 Code Predictor 作为初始输入 ]
                |
        [ Code Predictor 开始帧内自回归预测 Code_t_c1 ~ c15 ]
```

### 4. C++ 移植提醒
在实现时，确保：
* `past_hidden` 是 Talker Transformer 输出的 `last_hidden_state[:, -1:, :]`。
* `last_id_hidden` 是 `Talker.get_input_embeddings()(predicted_code_0_token)`。
* 拼接后，Code Predictor 的输入序列长度为 2（Prefill），然后在循环中逐个追加新预测的 Token Embedding。
````