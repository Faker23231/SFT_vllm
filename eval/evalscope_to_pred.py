# -*- coding: utf-8 -*-
"""把 evalscope 的评测产物转换成 score_bench 需要的格式。

为什么需要这一步：
    evalscope 负责"按我们的 prompt 把模型跑一遍"并把结果落盘，但它的打分口径
    （一般按相似度或它的内置 metric）和本项目的 Joint-EM / 幻觉槽位率不一致。
    我们不硬改 evalscope 的 metric 体系（版本漂移风险高），而是按行序把预测
    取出来，交给 eval/metrics.py 统一打分——两个阶段的数字因此永远可比。

用法：
    python eval/evalscope_to_pred.py --work-dir reports/evalscope/baseline --tag baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
BENCH = BASE / "bench"
NODES = ["n1_intent", "n2_slot", "n3_reply", "n4_escalate"]

# evalscope 不同版本对"模型输出"的字段命名不一致，按优先级探测
PRED_KEYS = ("predict", "prediction", "response", "generated", "output", "model_output")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pick_text(row: dict) -> str:
    for key in PRED_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):  # 有些版本把输出包在 {"content": ...} 里
            content = value.get("content")
            if isinstance(content, str):
                return content
    return ""


def find_review_files(work_dir: Path, node: str) -> list[Path]:
    """evalscope 的落盘结构随版本变化，这里按"目录名含节点名 + jsonl"宽松匹配。"""
    patterns = [
        f"**/{node}/reviews/*.jsonl",
        f"**/*{node}*/reviews/*.jsonl",
        f"**/*{node}*/**/*.jsonl",
    ]
    for pattern in patterns:
        hits = sorted(work_dir.glob(pattern))
        if hits:
            return hits
    return []


def convert_node(work_dir: Path, node: str) -> list[dict]:
    files = find_review_files(work_dir, node)
    if not files:
        print(f"[warn] {node}：在 {work_dir} 下找不到 reviews，跳过")
        return []

    rows: list[dict] = []
    for path in files:
        rows.extend(load_jsonl(path))
    if not rows:
        return []

    bench_rows = load_jsonl(BENCH / node / "test.jsonl")
    # evalscope 通常保留输入顺序；若带 index 字段则按 index 对齐，更稳
    indexed = [row for row in rows if isinstance(row.get("index"), int)]
    if len(indexed) == len(rows):
        rows = sorted(rows, key=lambda row: row["index"])

    output = []
    for order, bench_row in enumerate(bench_rows):
        if order < len(rows):
            text = pick_text(rows[order])
        else:
            text = ""
        output.append({"id": bench_row["id"], "prediction": text})

    if len(rows) != len(bench_rows):
        print(f"[warn] {node}：reviews {len(rows)} 条 vs bench {len(bench_rows)} 条，按行序截断对齐")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="evalscope 结果 → score_bench 输入")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--out-dir", default=str(BASE / "reports" / "preds"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for node in NODES:
        rows = convert_node(Path(args.work_dir), node)
        if not rows:
            continue
        with (out_dir / f"{node}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[ok] {node}: {len(rows)} 条 → {out_dir / f'{node}.jsonl'}")
    print(f"\n下一步：python eval/score_bench.py --pred-dir {out_dir} --tag {args.tag}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
