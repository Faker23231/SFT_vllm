# -*- coding: utf-8 -*-
"""02 LoRA 监督微调（PEFT + SFTTrainer），产物写入 outputs/qwen3-0.6b-lora。

与 01 全量微调的区别：基座参数冻结，只训练注入的 LoRA 低秩矩阵，学习率更大（1e-4），
并开启 assistant_only_loss。

环境变量：
    SFT_TRAIN_LIMIT   >0 时只用前 N 条训练样本（快速演示用），默认全量
    SFT_EVAL_STEPS    每隔多少步验证/保存一次，默认 500

注意：/root/... 是原文中 Linux 服务器的路径，本机运行需换成自己的目录。
"""

import os
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 保证从任意目录执行时 src 包可导入

os.environ["TENSORBOARD_LOGGING_DIR"] = str(Path("/root/tf-logs") / time.strftime("%Y%m%d-%H%M%S"))  # 每次实验一个独立目录
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/hf"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 必须在 import torch 之前设置

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.gpu_guard import GpuClockLock
from src.message import to_message  # 原文如此导入

BASE_PATH = Path(__file__).parents[1]
MODEL_PATH = "Qwen/Qwen3-0.6B"
DATA_PATH = BASE_PATH / "data"
OUTPUT_PATH = BASE_PATH / "outputs" / "qwen3-0.6b-lora"  # 只保存 Adapter，不是完整模型

TRAIN_LIMIT = int(os.environ.get("SFT_TRAIN_LIMIT", "0") or 0)
EVAL_STEPS = int(os.environ.get("SFT_EVAL_STEPS", "500") or 500)


def train():
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(DATA_PATH / "train.jsonl"),
            "validation": str(DATA_PATH / "validation.jsonl"),
        },
    )
    dataset = dataset.map(
        build_messages,
        fn_kwargs={"include_answer": True},
        remove_columns=["text", "intent", "slots"],
    )

    if TRAIN_LIMIT:
        dataset["train"] = dataset["train"].select(range(min(TRAIN_LIMIT, len(dataset["train"]))))
        dataset["validation"] = dataset["validation"].select(range(min(max(TRAIN_LIMIT // 10, 16), len(dataset["validation"]))))
        print(f"[快速演示] 训练集截取为 {len(dataset['train'])} 条，验证集截取为 {len(dataset['validation'])} 条")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        attn_implementation="sdpa",
    )

    # Windows(WDDM) 显存超卖不报 OOM 而是换页到内存，会把整机拖死；卡上限后能直接看到错误
    torch.cuda.set_per_process_memory_fraction(0.8, 0)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # 低秩维度，越小参数越少、表达能力越弱
        lora_alpha=16,  # 实际缩放为 alpha/r = 2
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # 注意力 + FFN 全覆盖，通常比只调 q/v 效果好
        lora_dropout=0.05,
        bias="none",  # 不训练 bias，进一步减少可训练参数
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()  # 确认注入成功，一般 1% 以内

    training_args = SFTConfig(
        output_dir=str(OUTPUT_PATH),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # 等效 batch = 2 x 8 = 16
        gradient_checkpointing=True,  # 激活显存降到 1/5~1/8
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=1e-4,  # 只训练少量 LoRA 参数，通常比全量微调大一个量级
        max_length=1024,
        assistant_only_loss=True,  # 只在 assistant 回复 token 上算 loss
        bf16=True,
        optim="adamw_torch_fused",
        logging_steps=max(EVAL_STEPS // 10, 1),
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        save_total_limit=2,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=peft_model,  # 已注入 LoRA 的模型，训练时只更新 LoRA 参数
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_PATH / "best"))  # 只含 adapter_model.safetensors / adapter_config.json 与分词器


if __name__ == "__main__":
    with GpuClockLock():
        train()
