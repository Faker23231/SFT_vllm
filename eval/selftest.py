# -*- coding: utf-8 -*-
"""指标自检：把金标当作预测喂进打分链路，结果必须全对。

为什么需要它：
    metrics.py 里"解析格式"和"gold 序列化格式"是两处独立写的代码，
    任何一处改动都可能让两者错位——表现就是"模型明明答对了却判错"，
    而这种 bug 在真实评测里极难发现（只会觉得模型效果差）。

预期结果：
    n1 accuracy=100%  n2 joint_em=100%  n3 约束通过=100%  n4 f1=100%
    任何一项不到 100%，说明 metrics 与 gold 格式不一致，先修这里再谈模型效果。

用法：
    python eval/selftest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from bench import prompts  # noqa: E402
from eval import metrics  # noqa: E402

BENCH = BASE / "bench"
NODES = ["n1_intent", "n2_slot", "n3_reply", "n4_escalate"]
EXPECT = {"n1_intent": "accuracy", "n2_slot": "joint_em", "n3_reply": "constraint_pass_rate",
          "n4_escalate": "f1"}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    failed = []
    for node in NODES:
        path = BENCH / node / "test.jsonl"
        if not path.exists():
            print(f"[skip] {node}：缺少 {path}")
            failed.append(node)
            continue

        records = []
        for row in load_jsonl(path):
            messages = prompts.build(node, row["input"], include_answer=True, gold=row["gold"])["messages"]
            records.append(
                {
                    "id": row["id"],
                    "gold": row["gold"],
                    "input": row["input"],
                    "constraints": row.get("constraints"),
                    "prediction": messages[-1]["content"],  # 金标当预测
                }
            )

        result = metrics.score_records(node, records, labels=prompts.INTENTS)
        key = EXPECT[node]
        value = result["metrics"][key]
        status = "OK" if value == 1.0 else "FAIL"
        if value != 1.0:
            failed.append(node)
            for record in records[:3]:
                if not _record_ok(node, record):
                    print(f"   失败样例 {record['id']}: gold={record['gold']} pred={record['prediction'][:80]}")
        print(f"[{status}] {node}: {key}={value * 100:.2f}%（{result['count']} 条）")

    if failed:
        print(f"\n自检未通过：{failed}，先修 metrics 或 gold 序列化，再做正式评测")
        return 1
    print("\n自检通过：打分链路与 gold 格式一致")
    return 0


def _record_ok(node: str, record: dict) -> bool:
    """逐条复核，用于打印失败样例。"""
    pred, gold = record["prediction"], record["gold"]
    if node == "n1_intent":
        return metrics.parse_intent(pred, prompts.INTENTS) == gold["label"]
    if node == "n2_slot":
        return metrics.joint_exact_match(metrics.parse_slots_answer(pred), gold)
    if node == "n3_reply":
        return metrics.check_constraints(pred, record.get("constraints"))[0]
    obj = metrics.extract_json(pred)
    return isinstance(obj, dict) and obj.get("escalate") == gold["escalate"]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
