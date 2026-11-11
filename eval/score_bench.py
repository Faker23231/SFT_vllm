# -*- coding: utf-8 -*-
"""对预测文件打分，产出报告与"失败样例"清单。

输入：
    bench/<node>/test.jsonl          结构化金标（build_bench.py 产出）
    <pred_dir>/<node>.jsonl          每行 {"id": "...", "prediction": "模型原始输出"}

输出：
    <out_dir>/<tag>_<node>.json      单节点打分结果
    <out_dir>/<tag>_summary.md       汇总表，可直接贴进文档
    <out_dir>/<tag>_<node>_failures.jsonl   失败样例，P4 差距分析的原材料

用法（项目根目录）：
    python eval/score_bench.py --pred-dir reports/preds/baseline --tag baseline
    python eval/score_bench.py --pred-dir reports/preds/ft --tag ft --node n2_slot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from bench.prompts import INTENTS  # noqa: E402
from eval import metrics  # noqa: E402

BENCH = BASE / "bench"
NODES = ["n1_intent", "n2_slot", "n3_reply", "n4_escalate"]
MAX_FAILURES = 200


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_predictions(path: Path) -> dict[str, str]:
    if not path.exists():
        print(f"[warn] 预测文件不存在：{path}，该节点按全空处理")
        return {}
    predictions = {}
    for row in load_jsonl(path):
        pred_id = row.get("id")
        text = row.get("prediction") or row.get("predict") or row.get("output") or ""
        if pred_id is not None:
            predictions[str(pred_id)] = str(text)
    return predictions


def is_failure(node: str, record: dict) -> bool:
    """判定单条是否失败，用于生成差距分析清单。"""
    pred = record["prediction"]
    gold = record["gold"]
    if node == "n1_intent":
        return metrics.parse_intent(pred, INTENTS) != gold.get("label")
    if node == "n2_slot":
        parsed = metrics.parse_slots_answer(pred)
        return not metrics.joint_exact_match(parsed, gold)
    if node == "n3_reply":
        ok, _ = metrics.check_constraints(pred, record.get("constraints"))
        return not ok
    if node == "n4_escalate":
        obj = metrics.extract_json(pred)
        flag = obj.get("escalate") if isinstance(obj, dict) else None
        return flag != gold.get("escalate")
    return False


def score_node(node: str, pred_path: Path, from_gold: bool = False) -> tuple[dict, list[dict]]:
    bench_path = BENCH / node / "test.jsonl"
    if not bench_path.exists():
        raise SystemExit(f"缺少基准集：{bench_path}，先跑 bench/build_bench.py")

    if from_gold:  # 冒烟用：金标当预测，验证报告链路，分数应全对
        from bench import prompts

        predictions = {}
        for item in load_jsonl(bench_path):
            messages = prompts.build(node, item["input"], include_answer=True, gold=item["gold"])["messages"]
            predictions[item["id"]] = messages[-1]["content"]
    else:
        predictions = load_predictions(pred_path)
    records = []
    for item in load_jsonl(bench_path):
        records.append(
            {
                "id": item["id"],
                "gold": item["gold"],
                "input": item.get("input") or {},
                "constraints": item.get("constraints"),
                "prediction": predictions.get(item["id"], ""),
            }
        )

    result = metrics.score_records(node, records, labels=INTENTS)
    failures = [
        {
            "id": rec["id"],
            "difficulty": rec.get("difficulty"),
            "category": next((r["category"] for r in load_jsonl(bench_path) if r["id"] == rec["id"]), None),
            "input": rec["input"],
            "gold": rec["gold"],
            "prediction": rec["prediction"][:400],
        }
        for rec in records
        if is_failure(node, rec)
    ][:MAX_FAILURES]
    return result, failures


def render_summary(results: list[dict], tag: str) -> str:
    lines = [f"# 评测汇总（{tag}）", "", "| 节点 | 条数 | 关键指标 |", "| --- | --- | --- |"]
    for result in results:
        node = result["node"]
        values = result["metrics"]
        if node == "n1_intent":
            key = f"Acc {values['accuracy']*100:.2f}% / Macro-F1 {values['macro_f1']*100:.2f}%"
        elif node == "n2_slot":
            key = (f"JointEM {values['joint_em']*100:.2f}% / JSON合法 {values['json_valid_rate']*100:.2f}% "
                   f"/ 幻觉槽位 {values['hallucination_slot_rate']*100:.2f}%")
        elif node == "n3_reply":
            key = f"约束通过 {values['constraint_pass_rate']*100:.2f}% / 平均 {values['avg_chars']:.0f} 字"
        else:
            key = (f"F1 {values['f1']*100:.2f}% / 漏判 {values['miss_rate']*100:.2f}% "
                   f"/ JSON合法 {values['json_valid_rate']*100:.2f}%")
        lines.append(f"| {node} | {result['count']} | {key} |")

    lines += ["", "## 明细", ""]
    for result in results:
        lines.append(metrics.format_report(result))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="对基准集预测结果打分")
    parser.add_argument("--pred-dir", required=True, help="预测文件目录，内含 <node>.jsonl")
    parser.add_argument("--tag", default="run", help="实验名，用于产物文件名（baseline / ft / gptq）")
    parser.add_argument("--node", default="all", choices=["all"] + NODES)
    parser.add_argument("--out-dir", default=str(BASE / "reports" / "bench"))
    parser.add_argument("--from-gold", action="store_true", help="金标当预测，用于冒烟验证报告链路")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = NODES if args.node == "all" else [args.node]

    results = []
    for node in nodes:
        result, failures = score_node(node, pred_dir / f"{node}.jsonl", from_gold=args.from_gold)
        results.append(result)
        (out_dir / f"{args.tag}_{node}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (out_dir / f"{args.tag}_{node}_failures.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for item in failures:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        metrics_text = "  ".join(
            f"{key}={value * 100:.2f}%" if isinstance(value, float) and 0 <= value <= 1
            else f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in result["metrics"].items()
            if not isinstance(value, dict)
        )
        print(f"[{node}] n={result['count']} 失败={len(failures)}\n    {metrics_text}")

    summary = render_summary(results, args.tag)
    (out_dir / f"{args.tag}_summary.md").write_text(summary, encoding="utf-8")
    print(f"\n汇总 → {out_dir / f'{args.tag}_summary.md'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
