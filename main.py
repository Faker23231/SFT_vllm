# -*- coding: utf-8 -*-
"""一条命令跑完整套实验：基线评测 → 全量微调 → LoRA → QLoRA，每步再评测/合并。

用法（项目根目录执行）：
    python main.py           # MASSIVE 全量数据，耗时以小时计
    python main.py --quick   # 快速演示：训练 400 条、评测 100 条

每个阶段单独起一个 Python 进程：阶段结束显存立刻释放，单个阶段跑崩也不会带死整条流水线。
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
QUICK = "--quick" in sys.argv

# 通过环境变量传给所有子进程，三个训练脚本和评测脚本都读这四个名字
ENV = {
    "SFT_TRAIN_LIMIT": 400 if QUICK else 0,  # 训练集只用前 N 条，0 = 全量
    "SFT_EVAL_LIMIT": 100 if QUICK else 0,  # 测试集只评测前 N 条，0 = 全量
    "SFT_EVAL_STEPS": 20 if QUICK else 500,  # 每隔多少步验证/保存一次
    "SFT_MAX_NEW_TOKENS": 128 if QUICK else 256,  # 单条最多生成 token 数（答案只有几十个）
}

# (阶段名, 脚本, 抓分数的正则；None 表示这步不出分数)
STEPS = [
    ("基线评测", "eval/baseline_eval.py", r"Baseline score:\s*([\d.]+)%"),
    ("全量微调", "finetune/01_full_finetune.py", None),
    ("全量微调评测", "eval/full_eval.py", r"Fine-tuned score:\s*([\d.]+)%"),
    ("LoRA 微调", "finetune/02_lora_finetune.py", None),
    ("LoRA 评测", "eval/lora_eval.py", r"LoRA fine-tuned score:\s*([\d.]+)%"),
    ("LoRA 合并", "eval/merge_lora.py", None),
    ("QLoRA 微调", "finetune/03_qlora_finetune.py", None),
    ("QLoRA 评测", "eval/qlora_eval.py", r"QLoRA score:\s*([\d.]+)%"),
    ("QLoRA 合并", "eval/merge_qlora.py", None),
]


def run(script):
    """执行一个脚本，实时打印输出，返回 (退出码, 全部输出, 耗时秒)。"""
    env = os.environ.copy()
    env.update({k: str(v) for k, v in ENV.items()})
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    start = time.time()
    proc = subprocess.Popen(
        [sys.executable, str(BASE / script)],
        cwd=str(BASE),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait()
    return proc.returncode, "".join(lines), time.time() - start


def main():
    print("=" * 70)
    print(f"运行模式：{'快速演示（--quick）' if QUICK else '全量数据'}")
    print("\n训练曲线：tensorboard --logdir D:\\root\\tf-logs --port 6006")

    print(f"参数：训练上限={ENV['SFT_TRAIN_LIMIT'] or '全量'}  "
          f"评测上限={ENV['SFT_EVAL_LIMIT'] or '全量'}  "
          f"验证间隔={ENV['SFT_EVAL_STEPS']} 步")
    print("=" * 70)

    results = []
    scores = {}

    for name, script, pattern in STEPS:
        print(f"\n{'=' * 70}\n[{name}]  {script}\n{'=' * 70}")
        code, output, cost = run(script)

        score = None
        if pattern:
            found = re.findall(pattern, output)
            score = float(found[-1]) if found else None
            if score is not None:
                scores[name] = score

        status = "成功" if code == 0 else f"失败(退出码 {code})"
        results.append((name, status, cost, score))
        print(f"\n--> {name} {status}，用时 {cost:.0f} 秒")

        if code != 0:
            print("\n".join(output.strip().splitlines()[-20:]))
            print("流程中止。修好这个阶段后再跑。")
            break

    print(f"\n{'=' * 70}\n执行汇总\n{'=' * 70}")
    print("阶段        状态          用时(秒)      分数")
    for name, status, cost, score in results:
        text = f"{score:.2f}%" if score is not None else "-"
        print(f"{name:<10}  {status:<10}  {cost:>8.0f}  {text:>8}")

    baseline = scores.get("基线评测")
    if baseline is not None:
        print("\n相对基线的提升：")
        for name in ("全量微调评测", "LoRA 评测", "QLoRA 评测"):
            if name in scores:
                print(f"  {name}：{scores[name]:.2f}%（{scores[name] - baseline:+.2f} 个百分点）")

    print("\n训练曲线：tensorboard --logdir D:\\root\\tf-logs --port 6006")


if __name__ == "__main__":
    main()
