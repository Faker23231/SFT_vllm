# -*- coding: utf-8 -*-
"""05 加载 QLoRA Adapter 并合并成完整模型，输出到 outputs/qwen3-0.6b-qlora-merged。

4-bit 权重不能直接与浮点 LoRA 增量相加，所以合并前必须换成 bf16 浮点基座重新加载 Adapter。
"""

from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

BASE_PATH = Path(__file__).resolve().parents[1]
adapter_path = BASE_PATH / "outputs" / "qwen3-0.6b-qlora" / "best"  # 训练脚本把 Adapter 存在 best/ 下

tokenizer = AutoTokenizer.from_pretrained(adapter_path)

model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_path,
    torch_dtype=torch.bfloat16,  # 浮点精度加载，才能与 LoRA 增量相加
    device_map="auto",
)

merged_model = model.merge_and_unload()
merged_model.save_pretrained(BASE_PATH / "outputs" / "qwen3-0.6b-qlora-merged")
tokenizer.save_pretrained(BASE_PATH / "outputs" / "qwen3-0.6b-qlora-merged")
