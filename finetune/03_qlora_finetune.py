# -*- coding: utf-8 -*-
"""03 QLoRA 监督微调（4-bit 量化基座 + LoRA），产物写入 outputs/qwen3-0.6b-qlora。

与 02 相比只多三处：BitsAndBytesConfig 4-bit 量化配置、加载时传 quantization_config、
注入 LoRA 前调用 prepare_model_for_kbit_training；优化器换成 paged_adamw_8bit。

环境变量：
    SFT_TRAIN_LIMIT   >0 时只用前 N 条训练样本（快速演示用），默认全量
    SFT_EVAL_STEPS    每隔多少步验证/保存一次，默认 500

注意：/root/... 是原文中 Linux 服务器的路径；bitsandbytes 4-bit 训练需要 NVIDIA GPU。
"""

import os
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 保证从任意目录执行时 src 包可导入

os.environ["TENSORBOARD_LOGGING_DIR"] = str(Path("/root/tf-logs") / time.strftime("%Y%m%d-%H%M%S"))  # 每次实验一个独立目录
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/hf"

import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,  # 冻结基础参数，并把未量化参数转 fp32
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from src.gpu_guard import GpuClockLock
from src.message import to_message  # 原文如此导入

BASE_PATH = Path(__file__).parents[1]
MODEL_PATH = "Qwen/Qwen3-0.6B"
DATA_PATH = BASE_PATH / "data"
OUTPUT_PATH = BASE_PATH / "outputs" / "qwen3-0.6b-qlora"  # 只保存 Adapter，4-bit 基座不保存

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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)  # 分词器始终是普通精度，不量化

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 线性层以 4-bit 存放，显存约为 bf16 的 1/4
        bnb_4bit_quant_type="nf4",  # NF4 对正态分布友好，QLoRA 论文推荐
        bnb_4bit_use_double_quant=True,  # 对量化缩放常数再量化一次
        bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时反量化到 bf16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,  # 由 bitsandbytes 把线性层替换为 4-bit 实现
    )
    model = prepare_model_for_kbit_training(model)

    # Windows(WDDM) 显存超卖不报 OOM 而是换页到内存，会把整机拖死；卡上限后能直接看到错误
    torch.cuda.set_per_process_memory_fraction(0.8, 0)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,  # 实际缩放为 alpha/r = 2
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=str(OUTPUT_PATH),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # 等效 batch = 2 x 8 = 16
        gradient_checkpointing=True,  # QLoRA 必须配合它才真正省得下来
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=1e-4,
        max_length=1024,
        assistant_only_loss=True,  # 只在 assistant 回复 token 上算 loss
        bf16=True,
        optim="paged_adamw_8bit",  # 优化器状态分页到 CPU、8-bit 存储，降低显存峰值
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
        model=peft_model,  # 4-bit 基座 + LoRA
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_PATH / "best"))


if __name__ == "__main__":
    with GpuClockLock():
        train()
