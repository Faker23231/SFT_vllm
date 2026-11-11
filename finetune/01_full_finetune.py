# -*- coding: utf-8 -*-
"""01 全参数微调（Full SFT），产物写入 outputs/qwen3-0.6b-full。

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

os.environ["TENSORBOARD_LOGGING_DIR"] = str(Path("/root/tf-logs") / time.strftime("%Y%m%d-%H%M%S"))  # 每次训练一个独立子目录
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/hf"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 减少显存碎片，必须在 import torch 之前设置

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.gpu_guard import GpuClockLock
from src.messages import build_messages

BASE_PATH = Path(__file__).parents[1]
MODEL_PATH = "Qwen/Qwen3-0.6B"
DATA_PATH = BASE_PATH / "data"
OUTPUT_PATH = BASE_PATH / "outputs" / "qwen3-0.6b-full"  # 目录名需与 eval/full_eval.py 保持一致

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
        fn_kwargs={"include_answer": True},  # 追加 assistant 消息作为监督信号
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
        attn_implementation="sdpa",  # 避免回退到 eager 把 attention 矩阵全量物化
    )

    # Windows(WDDM) 显存超卖不会报 OOM，而是换页到系统内存导致整机卡死；卡上限后能直接看到错误
    torch.cuda.set_per_process_memory_fraction(0.9, 0)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_PATH),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # 等效 batch = 2 x 8 = 16
        learning_rate=2e-5,
        max_length=1024,  # 本项目样本 652~716 token，降到 512 会截断 assistant 答案
        bf16=True,
        gradient_checkpointing=True,  # 不保存中间激活、反向时重算，激活显存降到 1/5~1/8
        gradient_checkpointing_kwargs={"use_reentrant": False},  # 新版非重入实现，配合梯度累积/检查点更稳
        optim="adamw_torch_fused",
        logging_steps=max(EVAL_STEPS // 10, 1),
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        save_total_limit=1,  # 每个检查点约 2.4 GB，少存以减少磁盘 IO
        save_only_model=True,  # 不存 optimizer/scheduler，代价是这些检查点不能续训
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,  # Windows 下多进程取数容易卡在 spawn/共享内存
        dataloader_pin_memory=False,
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,  # 新版 TRL 用 processing_class 取代 tokenizer 参数
    )
    trainer.train()
    trainer.save_model(str(OUTPUT_PATH / "best"))


if __name__ == "__main__":
    with GpuClockLock():  # 锁 SM 时钟规避 TDR 死机，退出时自动恢复
        train()
