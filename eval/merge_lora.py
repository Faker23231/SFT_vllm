# -*- coding: utf-8 -*-
"""04 加载 LoRA Adapter 并合并成完整模型，输出到 outputs/qwen3-0.6b-lora-merged。

merge_and_unload() 不原地修改模型，必须接收返回值；合并后不再依赖 Adapter，但体积远大于 Adapter。
"""

from pathlib import Path

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

BASE_PATH = Path(__file__).resolve().parents[1]
adapter_path = BASE_PATH / "outputs" / "qwen3-0.6b-lora" / "best"  # 训练脚本把 Adapter 存在 best/ 下

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_path,
    torch_dtype="auto",
    device_map="auto",
)
merged_model = model.merge_and_unload()  # LoRA 增量按 B@A × alpha/r 叠加进基座权重

merged_model.save_pretrained(BASE_PATH / "outputs" / "qwen3-0.6b-lora-merged")
tokenizer.save_pretrained(BASE_PATH / "outputs" / "qwen3-0.6b-lora-merged")
