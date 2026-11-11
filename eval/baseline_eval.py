# -*- coding: utf-8 -*-
"""00 基线评测（Joint Exact Match）：intent 与 slots 完全一致才算对。

evaluate_model() 是全套评测的公共函数，其余 eval 脚本都从这里 import，不要重复实现。

环境变量：
    SFT_EVAL_LIMIT       >0 时只评测前 N 条，默认全量 2974 条
    SFT_MAX_NEW_TOKENS   单条最多生成 token 数，默认 1024；基座模型常不输出结束符，
                         会一路生成到上限（1024 时约 8 秒/条），main.py 会压到 128
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 保证 src 包可导入
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 只读本地缓存，避免联网查 chat template 超时

import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.messages import build_messages

BASE_PATH = Path(__file__).parents[1]
DATA_PATH = BASE_PATH / "data"


def evaluate_model(model, tokenizer):
    """返回 Joint Exact Match 准确率（0~1）。"""
    test_dataset = load_dataset(
        "json",
        data_files=str(DATA_PATH / "test.jsonl"),
        split="train",  # 单文件时用 "train" 作为 split 名，与是否训练无关
    )
    test_dataset = test_dataset.map(build_messages)  # 默认 include_answer=False，不含标准答案

    eval_limit = int(os.environ.get("SFT_EVAL_LIMIT", "0") or 0)
    if eval_limit:
        test_dataset = test_dataset.select(range(min(eval_limit, len(test_dataset))))
        print(f"[快速演示] 测试集只评测前 {len(test_dataset)} 条")

    correct_count = 0
    batch_size = 4
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    max_new_tokens = int(os.environ.get("SFT_MAX_NEW_TOKENS", "1024") or 1024)

    for start in tqdm(range(0, len(test_dataset), batch_size), desc="Evaluating"):
        batch = test_dataset[start : start + batch_size]

        prompts = tokenizer.apply_chat_template(
            batch["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # 关闭 Qwen3 thinking，否则输出思考过程导致 JSON 解析失败
        )

        inputs = tokenizer(
            prompts,
            padding=True,
            padding_side="left",  # 左侧补齐，让各样本的生成起点对齐在末尾
            return_tensors="pt",
        ).to(model.device)
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 贪心解码，结果确定可复现
                pad_token_id=pad_id,
            )
        outputs = tokenizer.batch_decode(
            output_ids[:, input_length:],  # 只取新生成的部分
            skip_special_tokens=True,
        )

        for index, output in enumerate(outputs):
            expected = {
                "intent": batch["intent"][index],
                "slots": batch["slots"][index],
            }

            try:
                prediction = json.loads(output)
            except json.JSONDecodeError:
                prediction = None

            if prediction == expected:
                correct_count += 1

    return correct_count / len(test_dataset)


if __name__ == "__main__":
    from src.gpu_guard import auto_lock  # 放在这里而非模块级：本文件会被其他脚本 import

    auto_lock()
    model_path = "Qwen/Qwen3-0.6B"  # 未微调的原始模型，作为对照基线
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 设 CUDA_VISIBLE_DEVICES="" 可强制 CPU
    # 不用 device_map="auto"：accelerate 每步做设备校验与参数搬运，在 Windows 上是 GPU hang 的高风险路径
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto" if device == "cuda" else torch.float32,  # bf16 在 CPU 上很慢
    ).to(device)
    print(f"[设备] {device}", flush=True)

    baseline_score = evaluate_model(model, tokenizer)
    print(f"Baseline score: {baseline_score:.2%}")
