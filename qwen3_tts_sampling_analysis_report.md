# Qwen3-TTS 采样策略对比测试报告

## 1. 测试背景

本报告基于对 `examples/test_model_12hz_base.py` 在四种不同采样配置下生成的语音克隆音频进行时长的系统性分析。四个输出目录分别对应以下配置：

| 目录名 | 配置说明 | `do_sample` | `subtalker_dosample` |
|--------|---------|-------------|----------------------|
| `qwen3_tts_test_voice_clone_output_wav` | 1. 不禁用（默认） | `True` | `True` |
| `qwen3_tts_test_voice_clone_output_wav_talker_no_sample` | 2. 只禁用 `do_sample` | `False` | `True` |
| `qwen3_tts_test_voice_clone_output_wav_cp_no_sample` | 3. 只禁用 `subtalker_dosample` | `True` | `False` |
| `qwen3_tts_test_voice_clone_output_wav_talker_cp_no_sample` | 4. 都禁用 | `False` | `False` |

测试模型：`Qwen/Qwen3-TTS-12Hz-0.6B-Base`

关键生成参数（代码中固定）：
- `max_new_tokens = 2048`
- 采样率：12.5 Hz（每帧 80ms）
- 理论最大时长：`2048 / 12.5 = 163.84 s`（约 2 分 44 秒）

---

## 2. 代码机制分析

### 2.1 `do_sample` 与 `subtalker_dosample` 的分工

根据 `qwen_tts/core/models/modeling_qwen3_tts.py` 的实现：

- **`do_sample`**：控制 **主 Talker**（第一层码本）的生成策略。
  - `True`：使用 `top_k` + `top_p` + `temperature` 进行随机采样。
  - `False`：贪心解码（greedy argmax），每步都取概率最高的 token。
  - 该参数直接透传给 Transformers `GenerationMixin.generate()`。

- **`subtalker_dosample`**：控制 **Code Predictor（Sub-talker）** 的生成策略。
  - 在主 Talker 生成每一帧的第一个码本 token 后，Code Predictor 负责并行生成剩余的 `num_code_groups - 1` 个码本 token。
  - `True`：随机采样；`False`：贪心解码。

### 2.2 停止机制与 EOS Token

- **EOS Token ID**：`codec_eos_token_id = 2150`（由模型配置决定）。
- 该 token 被作为 `eos_token_id` 传入 `GenerationMixin.generate()`。
- 同时设置了 `min_new_tokens=2`，防止过早停止。
- 模型在训练时默认使用采样（`do_sample=True, temperature=0.9`），因此其输出分布对贪心解码并不鲁棒。

### 2.3 为什么贪心解码会导致无法停止？

当 `do_sample=False` 且 `subtalker_dosample=False` 时：

1. **主 Talker 无随机性**：每步都选择概率最高的 token，而 EOS token（2150）在训练分布中通常不是概率最高的选择，因此很难被选中。
2. **Sub-talker 无随机性**：固定返回相同的码本向量，导致主 Talker 的输入嵌入陷入循环或固定轨迹。
3. **正反馈死锁**：主 Talker 始终处于非 EOS 状态，Sub-talker 始终提供相同的确定性嵌入，模型无法“跳出”当前状态去生成 EOS，最终只能消耗完 `max_new_tokens`（2048 个 token）。

> 实测 163.76s 与理论值 163.84s 的微小差异（0.08s = 1 帧），属于测量精度或首帧偏移的正常误差范围。

---

## 3. 音频时长数据汇总

### 3.1 总体统计

| 配置 | 文件总数 | 最大时长 | 平均时长 | 异常 (>20s) |
|------|---------|---------|---------|------------|
| 不禁用 | 20 | 6.40 s | 5.22 s | 0 |
| 只禁用 `do_sample` | 20 | 8.48 s | 5.70 s | 0 |
| 只禁用 `subtalker_dosample` | 20 | 13.76 s | 6.14 s | 0 |
| 都禁用 | 20 | **163.76 s** | **116.29 s** | **14** |

### 3.2 详细时长表（单位：秒）

#### 3.2.1 不禁用（`do_sample=True, subtalker_dosample=True`）

| 文件名 | 时长 |
|--------|------|
| case1_promptSingle_synSingle_direct_icl_0 | 4.32 |
| case1_promptSingle_synSingle_direct_xvec_only_0 | 4.40 |
| case1_promptSingle_synSingle_promptThenGen_icl_0 | 5.04 |
| case1_promptSingle_synSingle_promptThenGen_xvec_only_0 | 5.04 |
| case2_promptSingle_synBatch_direct_icl_0 | 5.44 |
| case2_promptSingle_synBatch_direct_icl_1 | 4.80 |
| case2_promptSingle_synBatch_direct_xvec_only_0 | 4.56 |
| case2_promptSingle_synBatch_direct_xvec_only_1 | 4.00 |
| case2_promptSingle_synBatch_promptThenGen_icl_0 | 5.44 |
| case2_promptSingle_synBatch_promptThenGen_icl_1 | 5.76 |
| case2_promptSingle_synBatch_promptThenGen_xvec_only_0 | 5.84 |
| case2_promptSingle_synBatch_promptThenGen_xvec_only_1 | 5.36 |
| case3_promptBatch_synBatch_direct_icl_0 | 5.12 |
| case3_promptBatch_synBatch_direct_icl_1 | 6.32 |
| case3_promptBatch_synBatch_direct_xvec_only_0 | 4.24 |
| case3_promptBatch_synBatch_direct_xvec_only_1 | 5.84 |
| case3_promptBatch_synBatch_promptThenGen_icl_0 | 5.60 |
| case3_promptBatch_synBatch_promptThenGen_icl_1 | 6.40 |
| case3_promptBatch_synBatch_promptThenGen_xvec_only_0 | 5.28 |
| case3_promptBatch_synBatch_promptThenGen_xvec_only_1 | 5.68 |

**结论**：全部正常停止，时长分布在 4~6.5 秒之间。

---

#### 3.2.2 只禁用 `do_sample`（`do_sample=False, subtalker_dosample=True`）

| 文件名 | 时长 |
|--------|------|
| case1_promptSingle_synSingle_direct_icl_0 | 5.52 |
| case1_promptSingle_synSingle_direct_xvec_only_0 | 4.64 |
| case1_promptSingle_synSingle_promptThenGen_icl_0 | 5.36 |
| case1_promptSingle_synSingle_promptThenGen_xvec_only_0 | 5.12 |
| case2_promptSingle_synBatch_direct_icl_0 | 6.32 |
| case2_promptSingle_synBatch_direct_icl_1 | 4.48 |
| case2_promptSingle_synBatch_direct_xvec_only_0 | 6.08 |
| case2_promptSingle_synBatch_direct_xvec_only_1 | 4.72 |
| case2_promptSingle_synBatch_promptThenGen_icl_0 | 5.84 |
| case2_promptSingle_synBatch_promptThenGen_icl_1 | 4.80 |
| case2_promptSingle_synBatch_promptThenGen_xvec_only_0 | 5.12 |
| case2_promptSingle_synBatch_promptThenGen_xvec_only_1 | 4.40 |
| case3_promptBatch_synBatch_direct_icl_0 | **8.48** |
| case3_promptBatch_synBatch_direct_icl_1 | 6.72 |
| case3_promptBatch_synBatch_direct_xvec_only_0 | 5.60 |
| case3_promptBatch_synBatch_direct_xvec_only_1 | 5.84 |
| case3_promptBatch_synBatch_promptThenGen_icl_0 | 6.08 |
| case3_promptBatch_synBatch_promptThenGen_icl_1 | 7.12 |
| case3_promptBatch_synBatch_promptThenGen_xvec_only_0 | 5.76 |
| case3_promptBatch_synBatch_promptThenGen_xvec_only_1 | 5.92 |

**结论**：整体仍可正常停止，但 `case3`（prompt batch + synth batch）下时长有轻微膨胀，最长达到 **8.48s**。说明 Sub-talker 的采样单独保留时，能为系统提供足够的随机性来打破潜在循环，但主 Talker 的贪心解码仍会导致某些复杂 batch 场景生成偏长。

---

#### 3.2.3 只禁用 `subtalker_dosample`（`do_sample=True, subtalker_dosample=False`）

| 文件名 | 时长 |
|--------|------|
| case1_promptSingle_synSingle_direct_icl_0 | 6.16 |
| case1_promptSingle_synSingle_direct_xvec_only_0 | 5.44 |
| case1_promptSingle_synSingle_promptThenGen_icl_0 | 7.36 |
| case1_promptSingle_synSingle_promptThenGen_xvec_only_0 | 6.64 |
| case2_promptSingle_synBatch_direct_icl_0 | 5.04 |
| case2_promptSingle_synBatch_direct_icl_1 | 5.44 |
| case2_promptSingle_synBatch_direct_xvec_only_0 | 6.32 |
| case2_promptSingle_synBatch_direct_xvec_only_1 | 3.92 |
| case2_promptSingle_synBatch_promptThenGen_icl_0 | 4.96 |
| case2_promptSingle_synBatch_promptThenGen_icl_1 | 4.24 |
| case2_promptSingle_synBatch_promptThenGen_xvec_only_0 | 5.52 |
| case2_promptSingle_synBatch_promptThenGen_xvec_only_1 | 5.84 |
| case3_promptBatch_synBatch_direct_icl_0 | 5.92 |
| case3_promptBatch_synBatch_direct_icl_1 | 6.48 |
| case3_promptBatch_synBatch_direct_xvec_only_0 | 7.44 |
| case3_promptBatch_synBatch_direct_xvec_only_1 | 5.68 |
| case3_promptBatch_synBatch_promptThenGen_icl_0 | 4.80 |
| case3_promptBatch_synBatch_promptThenGen_icl_1 | **13.76** |
| case3_promptBatch_synBatch_promptThenGen_xvec_only_0 | 5.52 |
| case3_promptBatch_synBatch_promptThenGen_xvec_only_1 | 6.40 |

**结论**：绝大部分音频正常停止。仅 `case3_promptBatch_synBatch_promptThenGen_icl_1.wav` 出现异常延长（**13.76s**），但未耗尽 `max_new_tokens`。这说明主 Talker 的采样在大多数情况下足以引导模型生成 EOS，但在 `case3 + promptThenGen + icl` 的特定组合下，Sub-talker 的贪心解码会显著拖慢收敛。

---

#### 3.2.4 都禁用（`do_sample=False, subtalker_dosample=False`）

| 文件名 | 时长 | 状态 |
|--------|------|------|
| case1_promptSingle_synSingle_direct_icl_0 | **163.76** | ❌ 耗尽 max_new_tokens |
| case1_promptSingle_synSingle_direct_xvec_only_0 | **163.76** | ❌ 耗尽 max_new_tokens |
| case1_promptSingle_synSingle_promptThenGen_icl_0 | **163.76** | ❌ 耗尽 max_new_tokens |
| case1_promptSingle_synSingle_promptThenGen_xvec_only_0 | **163.76** | ❌ 耗尽 max_new_tokens |
| case2_promptSingle_synBatch_direct_icl_0 | **163.76** | ❌ 耗尽 max_new_tokens |
| case2_promptSingle_synBatch_direct_icl_1 | 5.20 | ✅ 正常 |
| case2_promptSingle_synBatch_direct_xvec_only_0 | **163.76** | ❌ 耗尽 max_new_tokens |
| case2_promptSingle_synBatch_direct_xvec_only_1 | 5.04 | ✅ 正常 |
| case2_promptSingle_synBatch_promptThenGen_icl_0 | **163.76** | ❌ 耗尽 max_new_tokens |
| case2_promptSingle_synBatch_promptThenGen_icl_1 | 5.20 | ✅ 正常 |
| case2_promptSingle_synBatch_promptThenGen_xvec_only_0 | **163.76** | ❌ 耗尽 max_new_tokens |
| case2_promptSingle_synBatch_promptThenGen_xvec_only_1 | 5.04 | ✅ 正常 |
| case3_promptBatch_synBatch_direct_icl_0 | **163.76** | ❌ 耗尽 max_new_tokens |
| case3_promptBatch_synBatch_direct_icl_1 | **163.76** | ❌ 耗尽 max_new_tokens |
| case3_promptBatch_synBatch_direct_xvec_only_0 | **163.76** | ❌ 耗尽 max_new_tokens |
| case3_promptBatch_synBatch_direct_xvec_only_1 | 6.32 | ✅ 正常 |
| case3_promptBatch_synBatch_promptThenGen_icl_0 | **163.76** | ❌ 耗尽 max_new_tokens |
| case3_promptBatch_synBatch_promptThenGen_icl_1 | **163.76** | ❌ 耗尽 max_new_tokens |
| case3_promptBatch_synBatch_promptThenGen_xvec_only_0 | **163.76** | ❌ 耗尽 max_new_tokens |
| case3_promptBatch_synBatch_promptThenGen_xvec_only_1 | 6.32 | ✅ 正常 |

**结论**：**14/20** 个音频耗尽 `max_new_tokens` 达到 **163.76s**。这是典型的贪心解码导致的无法生成 EOS token 的现象。

---

## 4. 文件名命名规则解析

文件名的结构为：

```
case{N}_{prompt模式}_{syn模式}_{调用方式}_{特征模式}_{batch索引}.wav
```

| 字段 | 含义 |
|------|------|
| `case1` | prompt single + synth single（单条参考，单条合成） |
| `case2` | prompt single + synth batch（单条参考，batch 合成两条文本） |
| `case3` | prompt batch + synBatch（两条参考，batch 合成两条文本） |
| `promptSingle` / `promptBatch` | 参考音频是单条还是 batch |
| `synSingle` / `synBatch` | 合成目标是单条文本还是 batch |
| `direct` | 直接调用 `generate_voice_clone()`，内部自动构建 prompt |
| `promptThenGen` | 先调用 `create_voice_clone_prompt()`，再传入 `generate_voice_clone()` |
| `icl` | `x_vector_only_mode=False`，使用参考文本 + 参考码本进行上下文学习 |
| `xvec_only` | `x_vector_only_mode=True`，仅使用说话人嵌入 |
| `_0` / `_1` | batch 中的索引。`_0` 对应英文文本，`_1` 对应中文文本 |

---

## 5. 关键发现与深层分析

### 5.1 都禁用配置下的语言/索引差异

在都禁用（`talker_cp_no_sample`）的 batch 合成中（case2 与 case3），存在一个明显的模式：

- **`_0`（英文）**：几乎全部耗尽 `max_new_tokens`（163.76s）
- **`_1`（中文）**：几乎都能正常停止（5~6s）

具体观察：
- `case2_synBatch_direct_icl_0` (英文): 163.76s ❌
- `case2_synBatch_direct_icl_1` (中文): 5.20s ✅
- `case2_synBatch_direct_xvec_only_0` (英文): 163.76s ❌
- `case2_synBatch_direct_xvec_only_1` (中文): 5.04s ✅
- `case3_synBatch_direct_xvec_only_0` (英文): 163.76s ❌
- `case3_synBatch_direct_xvec_only_1` (中文): 6.32s ✅

**推断**：在完全贪心解码下，模型对英文文本更容易陷入确定性循环，而中文文本的码本分布可能使得 EOS token 在贪心路径上更容易被选中。这与不同语言的训练数据分布或 token 化后的序列结构有关。

### 5.2 Batch 独立性验证

在 case2/case3 的 batch 生成中，同一 batch 内一个样本失控（到 163.76s）而另一个样本正常停止（5s）。这验证了代码中 `GenerationMixin.generate()` 的 **per-sample `is_done` 机制**——每个样本的停止条件是独立计算的，一个样本耗尽 max length 不会提前终止 batch 中其他样本的生成。

### 5.3 单样本合成（case1）全部失控

在都禁用配置下，所有 `case1`（单条参考 + 单条合成）无一例外都达到了 163.76s。这说明：
- 单样本场景下没有 batch 中其他样本的“干扰”或多样性影响。
- 一旦进入完全贪心模式，单样本合成对英文内容极易进入死锁状态。

### 5.4 `subtalker_dosample` 的“保险丝”作用

对比数据：
- 只禁用 `do_sample`：最长 8.48s，无失控。
- 只禁用 `subtalker_dosample`：最长 13.76s，无失控。
- 都禁用：14/20 失控到 163.76s。

这说明 **`subtalker_dosample` 和 `do_sample` 互为冗余的安全机制**。只要至少一个保持采样，就能在很大程度上避免无法生成 EOS 的问题。但从时长分布来看，保留 `do_sample`（主 Talker 采样）的效果略好于保留 `subtalker_dosample`，因为只禁用 `subtalker_dosample` 时的最大时长（13.76s）大于只禁用 `do_sample` 时的最大时长（8.48s）。

---

## 6. 结论与建议

### 6.1 核心结论

1. **必须保留至少一个采样开关**。`do_sample` 和 `subtalker_dosample` 同时禁用时，模型几乎必然在单样本场景下耗尽 `max_new_tokens`，在 batch 场景的英文合成中也极大概率失控。
2. **163.76s 的本质**：不是音频内容本身需要这么长，而是模型**无法生成 EOS token（2150）**，被 `max_new_tokens=2048` 强制截断。12.5Hz × 163.84s ≈ 2048 frames，与代码完全吻合。
3. **语言敏感性**：完全贪心解码下，英文合成比中文更容易陷入无法停止的循环。
4. **Batch 独立性**：同一 batch 内不同样本的失控与否是独立的，不影响其他样本的正常停止。

### 6.2 工程建议

1. **生产环境严禁同时关闭两个采样开关**。官方默认配置（`do_sample=True, subtalker_dosample=True`）是最安全的选择。
2. **如需确定性输出**（例如需要可复现的音频），建议：
   - 优先保留 `do_sample=True`，通过降低 `temperature`（如 0.3~0.5）来减少随机性，而不是关闭采样。
   - 若必须关闭一个采样开关，**优先保留 `do_sample`**，因为数据表明主 Talker 的采样对避免失控更为关键。
3. **增加运行时保护**：在调用 `generate_voice_clone()` 时，若检测到 `do_sample=False` 且 `subtalker_dosample=False`，可打印警告或自动限制 `max_new_tokens` 为更保守的值（如 512），避免生成过长的无意义音频。
4. **后处理过滤**：若生成的音频时长接近 `max_new_tokens / 12.5`（如 > 150s），应视为生成失败，可尝试重试或回退到采样模式。

---

*报告生成时间：2026-05-14*
*基于代码版本：`examples/test_model_12hz_base.py` 与 `qwen_tts/core/models/modeling_qwen3_tts.py`*
