# -*- coding: utf-8 -*-
"""把 evalscope 的评测产物转换成 score_bench 需要的格式。

为什么需要这一步：
    evalscope 负责"按我们的 prompt 把模型跑一遍"并把结果落盘，但它的打分口径
    （一般按相似度或它的内置 metric）和本项目的 Joint-EM / 幻觉槽位率不一致。
    我们不硬改 evalscope 的 metric 体系（版本漂移风险高），而是把预测取出来，
    交给 eval/metrics.py 统一打分——两个阶段的数字因此永远可比。

对齐方式（踩过的坑）：
    - evalscope 并发执行，落盘顺序是乱序，`index` 字段不能当行号直接用；
    - 不同版本的输出字段名不同（reviews 里是 messages[-1]，predictions 里是
      model_output.choices[0].message.content）。
    因此这里**按 (system, user) 两条消息的原文**去对齐 bench 行，与顺序、版本都无关。

用法：
    python eval/evalscope_to_pred.py --work-dir reports/evalscope/baseline --tag baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from bench import prompts  # noqa: E402

BENCH = BASE / "bench"
NODES = ["n1_intent", "n2_slot", "n3_reply", "n4_escalate"]

# evalscope 不同版本对"模型输出"的字段命名不一致，按优先级探测
PRED_KEYS = ("predict", "prediction", "response", "generated", "output", "model_output")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def pick_text(row: dict) -> str:
    """从一条 evalscope 记录里取出模型输出文本（兼容 reviews / predictions 两种结构）。"""
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        for message in reversed(messages):
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                return message["content"]

    for key in PRED_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            choices = value.get("choices")
            if isinstance(choices, list) and choices:
                content = (choices[0].get("message") or {}).get("content")
                if isinstance(content, str):
                    return content
            content = value.get("content")
            if isinstance(content, str):
                return content
    return ""


def prompt_key(messages: list[dict]) -> tuple[str, str] | None:
    """(system, user) 两段原文作为对齐键：同一节点的 prompt 唯一确定一条样本。"""
    system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    if not user:
        return None
    # system.txt 落盘时被 strip 过，bench 侧 prompt 带首尾空白，两边都要归一
    return system.strip(), user.strip()


def build_index() -> tuple[dict[tuple[str, str], tuple[str, str]], dict[str, list[str]]]:
    """bench 侧：prompt 键 → (node, id)，同时保留每个节点的 id 顺序用于输出。"""
    lookup: dict[tuple[str, str], tuple[str, str]] = {}
    order: dict[str, list[str]] = {}
    for node in NODES:
        rows = load_jsonl(BENCH / node / "test.jsonl")
        order[node] = [row["id"] for row in rows]
        for row in rows:
            messages = prompts.build(node, row["input"], include_answer=False)["messages"]
            key = prompt_key(messages)
            if key:
                lookup[key] = (node, row["id"])
    return lookup, order


def candidate_files(work_dir: Path) -> list[Path]:
    patterns = [
        "**/reviews/**/*.jsonl",       # v1.x：reviews/<model>/<dataset>.jsonl
        "**/predictions/**/*.jsonl",   # 兜底：只存模型输出的目录
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(work_dir.glob(pattern)))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="evalscope 结果 → score_bench 输入")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--out-dir", default=str(BASE / "reports" / "preds"))
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    lookup, order = build_index()

    predictions: dict[str, dict[str, str]] = {node: {} for node in NODES}
    matched = 0
    unmatched = 0
    for path in candidate_files(work_dir):
        for row in load_jsonl(path):
            messages = row.get("messages")
            if not isinstance(messages, list):
                continue
            key = prompt_key(messages)
            hit = lookup.get(key) if key else None
            if hit is None:
                unmatched += 1
                continue
            node, bench_id = hit
            text = pick_text(row)
            if bench_id not in predictions[node] or not predictions[node][bench_id]:
                predictions[node][bench_id] = text
            matched += 1

    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for node in NODES:
        if not predictions[node]:
            print(f"[warn] {node}：没有匹配到任何预测，跳过（检查 --work-dir 是否指对）")
            continue
        rows = [{"id": bench_id, "prediction": predictions[node].get(bench_id, "")} for bench_id in order[node]]
        empty = sum(1 for row in rows if not row["prediction"])
        with (out_dir / f"{node}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[ok] {node}: {len(rows)} 条（空预测 {empty}）→ {out_dir / f'{node}.jsonl'}")

    print(f"\n匹配 {matched} 条，未匹配 {unmatched} 条")
    print(f"下一步：python eval/score_bench.py --pred-dir {out_dir} --tag {args.tag}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
