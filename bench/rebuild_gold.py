# -*- coding: utf-8 -*-
"""用模型改写把 N3/N4 的模板金标换成「句式多样但标签不变」的正式测试集。

背景：`bench/build_bench.py --synth 300` 生成的 N3/N4 各 300 条里，288 条是模板拼接，
难度分布 97% 落在 medium，用来跑基线会低估长尾问题（见 bench/stats.md 的说明）。

做法（不是重新标注，只改输入的表达方式）：
    1. `data_pipeline/02_expand.py --backend openai/vllm --seed-factor 1 --times 3`
       以现有 test.jsonl 为种子，只改写「用户说的话」（N3 的 summary、N4 的 user 轮次），
       gold 一字不动 —— 标签因此天然保持一致，不需要再让人重标。
    2. 本脚本做质检 + 分层保留：
       - 改写后必须与原句不同、非空、长度合理；
       - N4 额外校验模型自标结果（02_expand 的 raw 列）与 gold 的 escalate 一致；
       - N3 校验 gold 仍满足 constraints（长度/必备信息/禁语）；
       - 每个种子最多保留 N 条，保证 category / difficulty 分布与原 test 集完全一致。
    3. 覆盖写出 `<node>/test.jsonl` 与 `<node>/test.evalscope.jsonl`，system.txt 不变。
       原始模板集在 git 里，需要回退直接 `git checkout -- bench/<node>/test.jsonl`。

用法：
    python data_pipeline/02_expand.py --backend openai --seed-factor 1 --times 3 \
        --seeds-dir /data/work/gold_seeds --out /data/work/gold_candidates.jsonl
    python bench/rebuild_gold.py --candidates /data/work/gold_candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from bench import prompts  # noqa: E402
from eval.metrics import check_constraints, extract_json  # noqa: E402

BENCH = BASE / "bench"
NODES = ["n3_reply", "n4_escalate"]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def user_text(node: str, record: dict) -> str:
    """取记录里「用户说的话」，N3 是 summary，N4 是最后一条 user 轮次。"""
    if node == "n3_reply":
        return (record["input"].get("summary") or "").strip()
    for turn in reversed(record["input"].get("dialog") or []):
        if turn.get("role") == "user":
            return (turn.get("text") or "").strip()
    return ""


def check_candidate(node: str, candidate: dict, seed: dict) -> tuple[bool, str]:
    new_text = user_text(node, candidate)
    if not new_text:
        return False, "改写为空"
    if new_text == user_text(node, seed):
        return False, "未改写"
    if not 4 <= len(new_text) <= 200:
        return False, f"长度异常({len(new_text)})"
    if "改写要求" in new_text or "原文：" in new_text:
        return False, "把 prompt 回显了"

    if node == "n4_escalate":
        parsed = extract_json(candidate.get("raw") or "")
        if parsed is None:
            return False, "自标无法解析为 JSON"
        if bool(parsed.get("escalate")) != bool(seed["gold"]["escalate"]):
            return False, "自标与 gold 不一致"
    else:
        ok, reasons = check_constraints(seed["gold"]["text"], seed.get("constraints"))
        if not ok:
            return False, "gold 违反自身约束：" + ",".join(reasons)
        if candidate.get("constraints") != seed.get("constraints"):
            return False, "constraints 被改动"
    return True, ""


def prompt_key(node: str, record: dict) -> tuple[str, str]:
    """整条 prompt 的指纹：不同种子改写后撞句时，靠它去重。"""
    messages = prompts.build(node, record["input"], include_answer=False)["messages"]
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    return system.strip(), user.strip()


def template_path(node: str) -> Path:
    """种子源：优先用不可变的模板副本，避免二次改写（test.jsonl 会被本脚本覆盖）。"""
    frozen = BENCH / node / "test.template.jsonl"
    return frozen if frozen.exists() else BENCH / node / "test.jsonl"


def rebuild(node: str, candidates: list[dict], keep_per_seed: int, limit: int = 0) -> tuple[list[dict], dict]:
    original = load_jsonl(template_path(node))
    by_seed: dict[str, list[dict]] = {}
    for row in candidates:
        by_seed.setdefault(row["seed_id"], []).append(row)

    selected: list[dict] = []
    rejected: Counter = Counter()
    seen: set[tuple[str, str]] = set()
    for seed in original:
        picked = 0
        for candidate in by_seed.get(seed["id"], []):
            if picked >= keep_per_seed:
                break
            ok, reason = check_candidate(node, candidate, seed)
            if not ok:
                rejected[reason] += 1
                continue
            fingerprint = prompt_key(node, candidate)
            if fingerprint in seen:  # 模板种子大量重复，改写后仍会撞成同一句
                rejected["与已保留样本重复"] += 1
                continue
            seen.add(fingerprint)
            selected.append(
                {
                    "id": f"{seed['id']}_r{picked}",
                    "node": node,
                    "difficulty": seed["difficulty"],
                    "category": seed["category"],
                    "input": candidate["input"],
                    "gold": seed["gold"],            # 金标不重写，标签天然一致
                    "constraints": seed.get("constraints"),
                    "synthetic": False,               # 已不是模板拼接
                    "rewritten_from": seed["id"],     # 可追溯：哪条模板改写而来
                }
            )
            picked += 1

    if limit and len(selected) > limit:
        # 固定种子抽样，保持 category 比例（按种子顺序截断会把 x 开头的人工种子全砍掉）
        random.Random(42).shuffle(selected)
        selected = selected[:limit]

    stats = {
        "node": node,
        "template_rows": len(original),
        "candidates": len(candidates),
        "kept": len(selected),
        "unique_prompts": len(seen),
        "rejected": dict(rejected),
        "difficulty": dict(Counter(row["difficulty"] for row in selected)),
        "category_kinds": len({row["category"] for row in selected}),
    }
    return selected, stats


def export_evalscope(node: str, records: list[dict]) -> None:
    rows = []
    for record in records:
        item = prompts.build(node, record["input"], include_answer=True, gold=record["gold"])
        rows.append({"query": item["messages"][1]["content"], "response": item["messages"][-1]["content"]})
    dump_jsonl(BENCH / node / "test.evalscope.jsonl", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="用改写样本重建 N3/N4 正式测试集")
    parser.add_argument("--candidates", required=True, help="02_expand.py 产出的候选 jsonl")
    parser.add_argument("--per-seed", type=int, default=3,
                        help="每个种子最多保留几条（默认 3，配合全量去重尽量把唯一 prompt 数做大）")
    parser.add_argument("--limit", type=int, default=0, help="每节点最多保留多少条（0 = 不限）")
    parser.add_argument("--nodes", nargs="+", default=NODES)
    parser.add_argument("--dry-run", action="store_true", help="只统计不落盘")
    args = parser.parse_args()

    candidates = load_jsonl(Path(args.candidates))
    report = []
    for node in args.nodes:
        rows, stats = rebuild(node, [c for c in candidates if c.get("node") == node],
                              args.per_seed, args.limit)
        report.append(stats)
        print(f"[{node}] 模板 {stats['template_rows']} → 候选 {stats['candidates']} → 保留 {stats['kept']}"
              f"（唯一 prompt {stats['unique_prompts']}）；"
              f"难度 {stats['difficulty']}；类别 {stats['category_kinds']} 类")
        if stats["rejected"]:
            print(f"        剔除原因：{stats['rejected']}")
        if args.dry_run:
            continue
        if not rows:
            print(f"[warn] {node} 无可用样本，保持原文件不变")
            continue
        dump_jsonl(BENCH / node / "test.jsonl", rows)
        export_evalscope(node, rows)
        print(f"        已写 {node}/test.jsonl 与 test.evalscope.jsonl")

    out = BENCH / "gold_rebuild_report.json"
    if not args.dry_run:
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告 → {out}")
        print("下一步：python eval/selftest.py && python eval/run_evalscope.py --dry-run")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
