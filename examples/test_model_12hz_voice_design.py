# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


def main():
    device = "cuda:0"
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    # -------- Single --------
    torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = tts.generate_voice_design(
        text="为了守护蒙德城周边的安定，我曾经发动过不少次「远征」，但比起这一次，都算不上什么…比如清剿达达乌帕谷、联合千岩军扫荡石门、从鹰翔海滩出发迎击外海魔物…嗯？你说难怪在这些地方都遇不到什么强敌…我应该还是留了些下来给人练手的吧？",
        language="Chinese",
        instruct="",
        dump_talker_prefill_path="debug_output/python_talker_prefill_input.bin",
        dump_talker_codes_npy_path="debug_output/python_sample_0_codes.npy",
        print_output_tokens=True,
    )

    torch.cuda.synchronize()
    t1 = time.time()
    elapsed = t1 - t0
    audio_duration = len(wavs[0]) / sr
    rtf = elapsed / audio_duration
    print(f"[VoiceDesign Single] time: {elapsed:.3f}s, audio: {audio_duration:.2f}s, RTF: {rtf:.3f}")

    sf.write("qwen3_tts_test_voice_design_single.wav", wavs[0], sr)

    # -------- Batch --------
    # texts = [
    #     "哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
    #     "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
    # ]
    # languages = ["Chinese", "English"]
    # instructs = [
    #     "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
    #     "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
    # ]

    # torch.cuda.synchronize()
    # t0 = time.time()

    # wavs, sr = tts.generate_voice_design(
    #     text=texts,
    #     language=languages,
    #     instruct=instructs,
    #     max_new_tokens=2048,
    # )

    # torch.cuda.synchronize()
    # t1 = time.time()
    # elapsed = t1 - t0
    # total_audio_duration = sum(len(w) for w in wavs) / sr
    # avg_rtf = elapsed / total_audio_duration
    # print(f"[VoiceDesign Batch] time: {elapsed:.3f}s, audio: {total_audio_duration:.2f}s, RTF: {avg_rtf:.3f}")

    # for i, w in enumerate(wavs):
    #     sf.write(f"qwen3_tts_test_voice_design_batch_{i}.wav", w, sr)


if __name__ == "__main__":
    main()