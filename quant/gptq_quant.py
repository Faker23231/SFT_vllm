# -*- coding: utf-8 -*-
"""GPTQ W4A16 量化（llm-compressor 官方配方），产物可直接被 vLLM 加载。

三个容易翻车的点，脚本里都做了显式处理：
    1. 校准集必须用领域数据（本项目用 bench 的 dev 集），用通用语料量化精度会明显掉；
    2. 输入必须是 merge 后的 bf16 全量权重，不能是 adapter 或 4bit 基座；
    3. 校准显存 = 权重 + 激活，14B bf16 约 28G + 6G，单卡 40G 才稳（24G 要把
       --max-seq-len 降到 1024、--num-calibration-samples 降到 256）。

用法（实例上，40G 卡）：
    python quant/gptq_quant.py \
        --model /root/autodl-tmp/models/qwen3-14b-merged \
        --calib-dir bench \
        --output /root/autodl-tmp/models/qwen3-14b-gptq-w4a16

量化后部署：
    vllm serve <output> --quantization gptq_marlin --max-model-len 8192
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from bench import prompts  # noqa: E402

NODES = ["n1_intent", "n2_slot", "n3_reply", "n4_escalate"]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_calibration_texts(calib_dir: Path, per_node: int) -> list[str]:
    """把 dev 集按训练时的 prompt 格式拼成纯文本，作为量化校准语料。

    为什么不用原始 jsonl：量化校准看的是"模型实际会看到的输入分布"，
    拼上 system prompt 才和线上推理一致。
    """
    texts: list[str] = []
    for node in NODES:
        path = calib_dir / node / "dev.jsonl"
        if not path.exists():
            print(f"[warn] 缺少 {path}，跳过该节点")
            continue
        rows = load_jsonl(path)[:per_node]
        for row in rows:
            messages = prompts.build(node, row["input"], include_answer=False)["messages"]
            texts.append("\n".join(f"[{m['role']}] {m['content']}" for m in messages))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="GPTQ W4A16 量化")
    parser.add_argument("--model", required=True, help="merge 后的 bf16 全量权重目录")
    parser.add_argument("--output", required=True)
    parser.add_argument("--calib-dir", default=str(BASE / "bench"))
    parser.add_argument("--per-node", type=int, default=128, help="每个节点取多少条 dev 作校准")
    parser.add_argument("--num-calibration-samples", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--scheme", default="W4A16", choices=["W4A16", "W8A16", "W8A8"])
    parser.add_argument("--ignore", nargs="+", default=["lm_head"],
                        help="不量化的层，lm_head 与首尾层保留 bf16 通常能救回一部分精度")
    parser.add_argument("--dry-run", action="store_true", help="只准备校准集并打印配置")
    args = parser.parse_args()

    texts = build_calibration_texts(Path(args.calib_dir), args.per_node)
    print(f"[info] 校准样本 {len(texts)} 条（上限 {args.num_calibration_samples}），"
          f"max_seq_len={args.max_seq_len}, scheme={args.scheme}, group_size={args.group_size}")
    if not texts:
        raise SystemExit("校准集为空，先跑 bench/build_bench.py")
    if args.dry_run:
        print("[dry-run] 样例：\n" + texts[0][:300])
        return

    import torch
    from datasets import Dataset
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.transformers import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[info] gpu={torch.cuda.get_device_name(0)} "
          f"mem={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}G")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto", attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = Dataset.from_list([{"text": text} for text in texts])

    try:
        recipe = GPTQModifier(targets="Linear", scheme=args.scheme, ignore=args.ignore,
                              group_size=args.group_size)
    except TypeError:  # 老版本不接受 group_size 参数，退化为默认 128
        print("[warn] 当前 llm-compressor 不支持 group_size 参数，使用默认值")
        recipe = GPTQModifier(targets="Linear", scheme=args.scheme, ignore=args.ignore)

    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=min(args.num_calibration_samples, len(texts)),
        output_dir=args.output,
    )
    tokenizer.save_pretrained(args.output)
    (Path(args.output) / "quant_config.json").write_text(
        json.dumps(
            {
                "method": "GPTQ",
                "scheme": args.scheme,
                "group_size": args.group_size,
                "ignore": args.ignore,
                "calibration_samples": min(args.num_calibration_samples, len(texts)),
                "max_seq_length": args.max_seq_len,
                "source_model": args.model,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] 量化产物 → {args.output}")
    print("[next] vllm serve 该目录 --quantization gptq_marlin，确认日志出现 gptq_marlin kernel")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
