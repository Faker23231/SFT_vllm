# -*- coding: utf-8 -*-
"""04 领域 QLoRA 微调（4-bit 基座 + LoRA），产物写入 outputs/<实验名>/best。

与 03_qlora_finetune.py 的差异：
    1. 基座换成 Qwen3-14B，显存按单卡 40G 规划（batch 4 / grad_accum 4 / max_length 4096）
    2. 数据集直接吃 data_pipeline 产出的 {"messages": [...]} 格式，不再依赖 MASSIVE 字段
    3. 开 packing：短样本拼批，token 利用率提升 30~50%，直接缩短训练时长（省卡时）
    4. 路径全部走数据盘，避免系统盘被 28G 权重写满

用法（实例上）：
    python finetune/04_domain_qlora.py \
        --dataset data/domain_sft_1to1_train.jsonl \
        --validation data/domain_sft_1to1_validation.jsonl \
        --output outputs/domain-r16 \
        --r 16 --max-samples 0

跑完务必：
    1. merge：python eval/merge_lora.py（或按 adapter_config.json 手动 merge）
    2. 复评：python eval/run_evalscope.py --tag ft --model qwen3-14b-ft
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA_ROOT = os.environ.get("SFT_DATA_ROOT", "/root/autodl-tmp")
os.environ.setdefault("HF_HOME", f"{DATA_ROOT}/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # 必须在 import torch 之前

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import LoraConfig, TaskType  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ATTN_ONLY = ["q_proj", "k_proj", "v_proj", "o_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="领域 QLoRA 微调")
    parser.add_argument("--model", default=os.environ.get("SFT_MODEL", "Qwen/Qwen3-14B"))
    parser.add_argument("--dataset", default=str(BASE / "data" / "domain_sft_1to1_train.jsonl"))
    parser.add_argument("--validation", default=str(BASE / "data" / "domain_sft_1to1_validation.jsonl"))
    parser.add_argument("--output", default=str(BASE / "outputs" / "domain-qlora"))
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=0, help="只训前 N 条，0 为全量（先小样本试跑）")
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--attn-only", action="store_true", help="只调 attention 层（消融用）")
    parser.add_argument("--no-packing", action="store_true", help="关闭 packing（对比训练时长用）")
    parser.add_argument("--gpu-mem-fraction", type=float, default=0.95)
    return parser.parse_args()


def load_messages(path: str, limit: int) -> "Dataset":
    dataset = load_dataset("json", data_files=str(path), split="train")
    dataset = dataset.select_columns(["messages"])  # 丢掉 node/category 等辅助列
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    print(f"[info] model={args.model} dataset={args.dataset} r={args.r} output={args.output}")

    train_dataset = load_messages(args.dataset, args.max_samples)
    eval_dataset = load_messages(args.validation, max(args.max_samples // 10, 16) if args.max_samples else 0)
    print(f"[info] train={len(train_dataset)} validation={len(eval_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # 统一用 deploy/chat_template_qwen3_nothink.jinja：
    #   1. 钉死 enable_thinking=false，与 bench/prompts.py、vLLM 评测口径一致；
    #   2. 模板里有 {% generation %} 标记，assistant_only_loss=True 才能算出 assistant mask
    #      （Qwen3 原生模板没有该标记，直接开 assistant_only_loss 会 RuntimeError）。
    nothink_tpl = BASE / "deploy" / "chat_template_qwen3_nothink.jinja"
    if nothink_tpl.exists():
        tokenizer.chat_template = nothink_tpl.read_text(encoding="utf-8")
        print(f"[info] chat template → {nothink_tpl.name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        attn_implementation="sdpa",  # 40G 卡上用 sdpa 足够，flash-attn 装不上时不必折腾
    )
    # 注意：这里**不能**手动 prepare_model_for_kbit_training + get_peft_model。
    # TRL 0.23 的 SFTTrainer 一旦发现传入 PeftModel，会在 __init__ 里再调
    # prepare_peft_model() → prepare_model_for_kbit_training()，该函数会把**所有**参数
    # requires_grad 置 False（包括已注入的 LoRA），结果训练全程 grad_norm=0、权重不变、
    # 复评分数与基线逐位相同（本仓库踩过这个坑）。
    # 正确做法：只传 4bit 基座 + peft_config，让 TRL 自己按 kbit 准备 → 注入 LoRA 的顺序执行。
    if torch.cuda.is_available():  # 限一下上限，显存问题立刻报错而不是拖垮机器
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, 0)
        print(f"[info] gpu={torch.cuda.get_device_name(0)} "
              f"mem={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}G")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=ATTN_ONLY if args.attn_only else TARGET_MODULES,
    )

    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        max_length=args.max_length,
        packing=not args.no_packing,
        assistant_only_loss=True,  # 只在 assistant 回复上算 loss，prompt 部分不参与
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=max(args.eval_steps // 10, 1),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to="tensorboard",
        run_name=Path(args.output).name,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,  # 交给 TRL 注入，顺序才是 kbit 准备 → LoRA
    )
    trainer.model.print_trainable_parameters()
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    if trainable == 0:
        raise SystemExit("[fatal] 可训参数为 0：LoRA 被冻结，训练会空转，停止执行")

    started = time.time()
    trainer.train()
    trainer.save_model(str(Path(args.output) / "best"))
    print(f"[done] 训练耗时 {(time.time() - started) / 60:.1f} 分钟 → {Path(args.output) / 'best'}")
    print("[next] 合并 adapter 后用 eval/run_evalscope.py 复评（tag=ft），再做 GPTQ 量化")


if __name__ == "__main__":
    main()
