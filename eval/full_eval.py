# -*- coding: utf-8 -*-
"""01 全量微调模型评测：普通方式加载 outputs/qwen3-0.6b-full/best 后复用 evaluate_model。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gpu_guard import auto_lock

auto_lock()  # 批量推理会长时间占满 GPU，先锁频规避 TDR 死机

from transformers import AutoModelForCausalLM, AutoTokenizer
from baseline_eval import BASE_PATH, evaluate_model

fine_tuned_model_path = (
    BASE_PATH
    / "outputs"
    / "qwen3-0.6b-full"
    / "best"  # trainer.save_model() 保存的完整权重
)

tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
model = AutoModelForCausalLM.from_pretrained(
    fine_tuned_model_path,
    torch_dtype="auto",
    device_map="auto",
)

fine_tuned_score = evaluate_model(model, tokenizer)  # 与基线同一测试集、同一生成参数
print(f"Fine-tuned score: {fine_tuned_score:.2%}")
