# -*- coding: utf-8 -*-
"""02 LoRA 模型评测。

LoRA 只能用 AutoPeftModelForCausalLM 加载 Adapter 目录：PEFT 会读 adapter_config.json
里的 base_model_name_or_path，自动先加载基座再挂上 LoRA 参数。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gpu_guard import auto_lock

auto_lock()

from baseline_eval import BASE_PATH, evaluate_model

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

lora_model_path = (
    BASE_PATH
    / "outputs"
    / "qwen3-0.6b-lora"
    / "best"  # adapter_model.safetensors + adapter_config.json + 分词器
)

tokenizer = AutoTokenizer.from_pretrained(lora_model_path)
model = AutoPeftModelForCausalLM.from_pretrained(
    lora_model_path,  # Adapter 目录，不是基座模型路径
    torch_dtype="auto",
    device_map="auto",
)

lora_score = evaluate_model(model, tokenizer)
print(f"LoRA fine-tuned score: {lora_score:.2%}")
