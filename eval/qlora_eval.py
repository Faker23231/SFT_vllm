# -*- coding: utf-8 -*-
"""03 QLoRA 模型评测。

必须重建与训练阶段完全相同的 BitsAndBytesConfig，否则基座会以 bf16 加载，
与训练时的 4-bit 计算状态不一致，分数不可比。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gpu_guard import auto_lock

auto_lock()

from baseline_eval import BASE_PATH, evaluate_model

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig

qlora_model_path = (
    BASE_PATH
    / "outputs"
    / "qwen3-0.6b-qlora"
    / "best"  # 只含 Adapter，不含 4-bit 基座权重
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # 以下三项必须与训练一致
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(qlora_model_path)
model = AutoPeftModelForCausalLM.from_pretrained(
    qlora_model_path,
    quantization_config=bnb_config,  # 基座按 4-bit 加载后再挂 Adapter
    device_map="auto",
)

qlora_score = evaluate_model(model, tokenizer)
print(f"QLoRA score: {qlora_score:.2%}")
