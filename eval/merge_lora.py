# -*- coding: utf-8 -*-
"""04 加载 LoRA/QLoRA Adapter 并合并成完整模型（参数化版）。

merge_and_unload() 不原地修改模型，必须接收返回值；合并后不再依赖 Adapter，但体积远大于 Adapter。

两个注意点：
    1. QLoRA 训练出的 adapter 必须用 bf16 浮点基座重新加载再合并
       （4-bit 权重不能与浮点 LoRA 增量直接相加，AutoPeftModel 会按 adapter_config
       里的 bnb 配置加载，这里用 --dtype 强制 bf16 覆盖）；
    2. tokenizer 一并落盘，产物目录可直接 vllm serve / GPTQ 量化。

用法（实例上）：
    python eval/merge_lora.py \
        --adapter outputs/ablation-1to1-r16/best \
        --output outputs/merged-1to1-r16
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 LoRA/QLoRA Adapter")
    parser.add_argument("--adapter", required=True, help="训练产物目录（含 adapter_config.json）")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"],
                        help="基座加载精度，QLoRA 必须用浮点才能合并")
    args = parser.parse_args()

    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    adapter_path = Path(args.adapter)
    if not (adapter_path / "adapter_config.json").exists():
        raise SystemExit(f"缺少 {adapter_path}/adapter_config.json，确认训练已完成")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        torch_dtype=getattr(torch, args.dtype),
        device_map="auto",
    )
    merged = model.merge_and_unload()  # LoRA 增量按 B@A × alpha/r 叠加进基座权重
    Path(args.output).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"[done] merged → {args.output}")


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    main()
