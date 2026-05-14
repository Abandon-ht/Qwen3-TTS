# Qwen3-TTS 贪心解码静音固定点分析

## 现象回顾

在 `do_sample=False, subtalker_dosample=False`（都禁用）的配置下，失控到 163.76s 的音频呈现两种模式：

1. **大部分音频**：前半段有正常语音（如 "Good one..."），后半段衰减为**静音**（非全零，而是数值极小的振荡，如 `00 00 FF FF`）。
2. **两个特例**（`case3_promptBatch_synBatch_direct_icl_1` 与 `case3_promptBatch_synBatch_promptThenGen_icl_1`）：**从第 1 帧开始就完全静音**。

本报告从代码层面解释这两种现象的根本原因。

---

## 1. 后半段静音：固定点吸引子（Fixed-Point Attractor）

### 1.1 静音不是全零

用户观测到的数据如：
```
00001920   00 00 FF FF  00 00 00 00  00 00 00 00  FF FF FF FF
```

这是 16-bit 有符号小端整数（PCM）表示的**接近零的微小波动**（`0x0000 = 0`, `0xFFFF = -1`）。其物理振幅约为 -60dB 以下，人耳完全无法感知，因此听起来是静音。

### 1.2 为什么会进入静音区？

根据 `modeling_qwen3_tts.py:1684–1702`，Talker 的每一步输入是所有 codebook embedding 的**求和**：

```python
codec_hiddens = torch.cat(
    [last_id_hidden]
    + [self.code_predictor.get_input_embeddings()[i](
           predictor_result.sequences[..., i:i+1]
       ) for i in range(self.config.num_code_groups - 1)],
    dim=1,
)
inputs_embeds = codec_hiddens.sum(1, keepdim=True)   # [B, 1, H]
```

当两个采样都禁用时：
- `input_ids`（codebook-0）每步都是 `argmax(logits)` → 完全确定
- `predictor_result.sequences`（codebook 1..N-1）每步也都是 `argmax` → 完全确定
- 因此 `inputs_embeds` 在 step *t* 和 step *t+1* **逐元素相等**
- 进而 `hidden_states`、`logits`、`input_ids` 全部逐帧重复

模型进入了一个**长度为 1 的确定性极限环**（deterministic limit cycle of length 1）：每帧都吐出完全相同的 codec token 组合。

### 1.3 语音 tokenizer 如何将重复 token 渲染为静音

语音 tokenizer（12Hz）使用 `SplitResidualVectorQuantizer` 解码：

```python
# modeling_qwen3_tts_tokenizer_v2.py ~784
def decode(self, codes: torch.Tensor) -> torch.Tensor:
    quantized = self.rvq_first.decode(codes[:, :self.n_q_semantic])
    if codes.shape[1] > self.n_q_semantic:
        quantized += self.rvq_rest.decode(codes[:, self.n_q_semantic:])
    return quantized
```

如果每帧的 `codes` 完全相同，解码器得到的 latent 向量也是常量。该常量经过因果 ConvNeXt、转置卷积和 `SnakeBeta` 激活后，输出一段**周期性接近零的波形**——这正是 `00 00 FF FF` 小幅振荡的来源。

### 1.4 为什么不是从头到尾都静音？

在生成初期，模型有**强烈的文本条件**支撑。具体而言：

```python
# modeling_qwen3_tts.py:1704-1707
if generation_step < trailing_text_hidden.shape[1]:
    inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1)
else:
    inputs_embeds = inputs_embeds + tts_pad_embed
```

每一步的 `inputs_embeds` 还会叠加上对应位置的文本嵌入 `trailing_text_hidden[:, step]`。只要文本还没用完，这个额外的条件向量就像一个**外部驱动力**，把系统推离静音固定点。

一旦 `generation_step` 超过了文本长度，`trailing_text_hidden` 被替换为无信息的 `tts_pad_embed`。此时：
- 外部驱动力消失
- 系统完全由自身的确定性反馈控制
- 如果当前状态离 EOS token（2150）较远，且离某个低能量（静音）token 较近，argmax 就会锁定在静音 token 上
- 一旦锁定，由于求和嵌入的逐帧一致性，**永远不可能自行跳出**

因此，音频呈现出：**前半段受文本牵引生成语音 → 文本耗尽后跌入静音固定点 → 后半段无限重复静音帧**。

---

## 2. 整段静音：case3-icl-1 的双重错配

### 2.1 是哪两个文件？

| 文件 | 时长 | 内容 |
|------|------|------|
| `case3_promptBatch_synBatch_direct_icl_1` | 163.76s | 全程静音 |
| `case3_promptBatch_synBatch_promptThenGen_icl_1` | 163.76s | 全程静音 |

而同一目录下的 `case3_..._xvec_only_1` 仅 6.32s，完全正常。

这说明问题出在 **ICL 模式** 与 **case3 的 batch 配置** 的交互上。

### 2.2 示例代码中的语言标签错配

在 `examples/test_model_12hz_base.py` 中：

```python
syn_text_batch = [
    "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.",  # ← 英文
    "其实我真的有发现，我是一个特别善于观察别人情绪的人。",                       # ← 中文
]
syn_lang_batch = ["Chinese", "English"]   # ← 标签写反了！
```

这意味着：
- **Index 0**：英文文本 + `language_id = Chinese`
- **Index 1**：中文文本 + `language_id = English`

### 2.3 Streaming ICL 模式下的文本耗尽

模型默认运行在 **streaming 模式**（`non_streaming_mode=False`）。在 `generate_icl_prompt` 中：

```python
# modeling_qwen3_tts.py:2029-2034
if text_lens > codec_lens:
    return text_embed[:, :codec_lens] + codec_embed, text_embed[:, codec_lens:]
else:
    text_embed = torch.cat([text_embed] + [tts_pad_embed] * (codec_lens - text_lens), dim=1)
    return text_embed + codec_embed, tts_pad_embed
```

对于 **index 1**（中文）：
- 参考音频 `clone_1.wav` 是一段完整语音，其 codec 帧数（`codec_len`）较大
- 参考文本 "甚至出现交易几乎停滞的情况。" token 化后的长度（`text_len`）较短
- 因此很可能满足 `text_len <= codec_len`
- 结果：`trailing_text_hidden` 直接变成无信息的 **`tts_pad_embed`**

也就是说，**index 1 从生成第一步开始就没有任何文本条件引导**，唯一的信息只有：
1. 错误的 English `language_id`
2. 参考音频的 `ref_code`（通过 ICL prompt 提供）
3. 一个与语言完全不匹配的合成文本

### 2.4 为什么第一步就跌入静音？

在 `do_sample=False` 的贪心模式下，模型的 `codec_head` 输出 logits。由于：
- 语言标签（English）与输入文本/音频（中文）严重错配 → 模型处于高度不确定的分布外状态
- 无 trailing text 提供逐帧指引 → 没有外部驱动力将 logits 推向语音相关区域
- 参考音频的 ICL 条件在 greedy 模式下无法提供足够的"惯性"来启动语音生成

此时，logits 的最大值很可能落在**码本中最频繁的低能量 token**（通常是 codebook-0 的 token 0，代表空白/静音）。argmax 选中它后，Sub-talker（也是 greedy）也选中对应的辅助 codebook 组合，产生一组代表静音的 code。

下一步：`inputs_embeds` 与上一步完全相同 → `hidden_states` 相同 → logits 相同 → 再次选中同一个静音 token。**从第 1 帧就锁定了静音固定点**。

### 2.5 为什么英文 index 0 不会整段静音？

Index 0 虽然也有语言标签错配（英文文本 + Chinese tag），但：
- 英文参考文本较长，在 streaming ICL 中可能满足 `text_len > codec_len`
- 因此它保留了有效的 `trailing_text_hidden`，在生成初期有强文本驱动
- 这足以让模型先输出一段有意义的语音，直到文本耗尽后才跌入静音区

### 2.6 为什么 xvec_only_1 完全正常？

`xvec_only` 模式不使用 ICL，不依赖 `ref_text` 与 `ref_code` 的交错对齐，也不受 `text_len <=> codec_len` 的 streaming 逻辑影响。它仅使用 speaker embedding，语言标签虽然仍是 English，但模型在这种模式下的分布更鲁棒。加上 `max_new_tokens` 上限内成功命中了 EOS token，因此 6.32s 就正常停止了。

---

## 3. 为什么静音固定点的数值是小幅振荡而不是恒定 0？

即使 latent 向量是完全恒定的，语音解码器内部的 **SnakeBeta 激活** 和 **因果卷积** 会引入微小的周期性非线性振荡。

```python
# modeling_qwen3_tts_tokenizer_v2.py ~873-888
hidden = self.quantizer.decode(codes)
hidden = self.pre_conv(hidden).transpose(1, 2)
hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state
...
wav = hidden
for block in self.decoder:
    wav = block(wav)
return wav.clamp(min=-1, max=1)
```

`SnakeBeta` 的公式为 `x + sin²(αx)/α`，即使输入 `x` 接近零，其周期性项仍会产生微小的非零输出。此外，chunked decode 的帧重叠和上下文累积也会引入 1 LSB 级别的数值噪声。这正是 `00 00 FF FF`（即 0 和 -1）交替出现的来源。

---

## 4. 总结

| 现象 | 根因 |
|------|------|
| **大部分失控音频：前半段语音 + 后半段静音** | 文本条件 `trailing_text_hidden` 在初期提供外部驱动力，生成正常语音；文本耗尽后，确定性反馈系统跌入**静音固定点**，无限重复同一静音帧。 |
| **case3-icl-1：全程静音** | 双重错配：(1) 语言标签 `English` 与中文内容错配；(2) streaming ICL 下 `text_len <= codec_len` 导致 `trailing_text_hidden = tts_pad_embed`，从第 1 步起就**没有任何文本引导**。贪心 argmax 直接选中静音 token 并永久锁定。 |
| **静音不是全零** | 语音解码器的 SnakeBeta 激活和因果卷积将恒定 latent 渲染为接近零的周期性微幅振荡（`0x0000` / `0xFFFF`）。 |
| **xvec_only_1 正常** | xvec_only 不依赖 ICL 的文本-音频对齐，避开了 `text_len <=> codec_len` 陷阱；speaker embedding 提供的条件更鲁棒，在 greedy 模式下仍能命中 EOS。 |

---

*分析基于：*
- `qwen_tts/core/models/modeling_qwen3_tts.py` (generate_icl_prompt, forward/generation branch)
- `qwen_tts/core/models/modeling_qwen3_tts_tokenizer_v2.py` (decode, SnakeBeta)
- `examples/test_model_12hz_base.py` (language tag 错配: `["Chinese", "English"]`)
