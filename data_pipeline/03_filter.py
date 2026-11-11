# -*- coding: utf-8 -*-
"""第三步：拒绝采样与质检（纯本地、零卡时）。

五道关，逐级淘汰并记录保留率：
    1. 结构校验   能否解析出符合 schema 的答案
    2. 标签一致性 模型对新文本的自标结果是否与种子标签一致
    3. 幻觉过滤   N2 的槽位 value 必须能在原文中找到（否则是编造）
    4. 去重       同节点内近似重复（8-gram minhash key）+ 与评测集重叠
    5. 配额       按 category 截断，避免高频场景淹没长尾

用法：
    python data_pipeline/03_filter.py --cap-per-category 200
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from bench.prompts import INTENTS  # noqa: E402
from eval import metrics  # noqa: E402

PIPE = BASE / "data_pipeline"
BENCH = BASE / "bench"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def grams(text: str, n: int = 8) -> set[str]:
    clean = "".join(text.split())
    if len(clean) <= n:
        return {clean} if clean else set()
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def sim_key(text: str, n: int = 8, k: int = 4) -> str:
    """minhash 简化版：取最小的 k 个 gram 哈希作为指纹。

    比 O(n²) 两两比较快得多，且 crc32 保证跨进程稳定（内置 hash 有随机盐）。
    """
    values = sorted(zlib.crc32(gram.encode("utf-8")) for gram in grams(text, n))
    return "-".join(str(value) for value in values[:k]) or "empty"


def input_text(record: dict) -> str:
    return json.dumps(record["input"], ensure_ascii=False)


def quality_check(node: str, candidate: dict) -> tuple[bool, str]:
    """返回 (是否通过, 淘汰原因)。"""
    raw = candidate.get("raw") or ""
    gold = candidate["gold"]
    if not raw.strip():
        return False, "空输出"

    if node == "n1_intent":
        label = metrics.parse_intent(raw, INTENTS)
        if label is None:
            return False, "标签不在枚举内"
        if label != gold.get("label"):
            return False, "标签与种子不一致"
        return True, ""

    if node == "n2_slot":
        parsed = metrics.parse_slots_answer(raw)
        if parsed is None:
            return False, "JSON 不可解析"
        if parsed.get("intent") != gold.get("intent"):
            return False, "意图与种子不一致"
        if not metrics.joint_exact_match(parsed, gold):
            return False, "槽位与种子不一致"
        text = candidate["input"].get("text", "")
        if metrics.hallucinated_slots(parsed.get("slots"), text):
            return False, "slot value 不在原文中"
        return True, ""

    if node == "n3_reply":
        ok, violations = metrics.check_constraints(raw, candidate.get("constraints"))
        return (True, "") if ok else (False, f"约束违反:{violations[0]}")

    if node == "n4_escalate":
        obj = metrics.extract_json(raw)
        if not isinstance(obj, dict) or not isinstance(obj.get("escalate"), bool):
            return False, "JSON 不可解析"
        if obj["escalate"] != gold.get("escalate"):
            return False, "判定与种子不一致"
        return True, ""

    return False, f"未知节点:{node}"


def bench_grams(node: str) -> set[str]:
    path = BENCH / node / "test.jsonl"
    if not path.exists():
        return set()
    pool: set[str] = set()
    for row in load_jsonl(path):
        pool |= grams(input_text(row))
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description="拒绝采样与质检")
    parser.add_argument("--input", default=str(PIPE / "out" / "candidates.jsonl"))
    parser.add_argument("--out-dir", default=str(PIPE / "out" / "filtered"))
    parser.add_argument("--cap-per-category", type=int, default=200, help="每个 category 最多保留多少条")
    parser.add_argument("--leak-ratio", type=float, default=0.8, help="与评测集 8-gram 命中比例超过此值即丢弃")
    args = parser.parse_args()

    candidates = load_jsonl(Path(args.input))
    print(f"[in] 候选 {len(candidates)} 条")

    stats: dict[str, Counter] = defaultdict(Counter)
    kept: dict[str, list[dict]] = defaultdict(list)
    fingerprints: dict[str, set[str]] = defaultdict(set)
    bench_pool: dict[str, set[str]] = {}

    for candidate in candidates:
        node = candidate["node"]
        stats[node]["total"] += 1

        ok, reason = quality_check(node, candidate)
        if not ok:
            stats[node][f"drop:{reason.split(':')[0]}"] += 1
            continue
        stats[node]["after_quality"] += 1

        text = input_text(candidate)
        key = sim_key(text)
        if key in fingerprints[node]:
            stats[node]["drop:重复"] += 1
            continue
        fingerprints[node].add(key)

        if node not in bench_pool:
            bench_pool[node] = bench_grams(node)
        hit = grams(text)
        if bench_pool[node] and hit:
            ratio = len(hit & bench_pool[node]) / len(hit)
            if ratio >= args.leak_ratio:
                stats[node]["drop:与评测集重叠"] += 1
                continue
        stats[node]["after_dedup"] += 1

        kept[node].append(candidate)

    # 配额：每个 category 截断
    final: dict[str, list[dict]] = {}
    for node, rows in kept.items():
        per_category: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            per_category[row["gold"].get("label") or row["gold"].get("intent")
                         or row.get("style", "unknown")].append(row)
        selected = []
        for category, items in per_category.items():
            selected.extend(items[: args.cap_per_category])
            stats[node][f"cap:{category}"] = min(len(items), args.cap_per_category)
        final[node] = selected
        stats[node]["final"] = len(selected)
        dump_jsonl(Path(args.out_dir) / f"{node}.jsonl", selected)

    # 报告
    lines = ["# 合成数据质检报告", "", "| 节点 | 候选 | 质量通过 | 去重后 | 最终 |", "| --- | --- | --- | --- | --- |"]
    total_final = 0
    for node, counter in stats.items():
        lines.append(f"| {node} | {counter['total']} | {counter['after_quality']} "
                     f"| {counter['after_dedup']} | {counter['final']} |")
        total_final += counter["final"]
    lines += ["", "## 淘汰原因明细", ""]
    for node, counter in stats.items():
        drops = {key.replace("drop:", ""): value for key, value in counter.items() if key.startswith("drop:")}
        lines.append(f"- {node}：{drops or '无'}")
    lines += ["", f"最终保留 {total_final} 条。"]

    report_path = Path(args.out_dir) / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for node, counter in stats.items():
        print(f"[{node}] 候选 {counter['total']} → 质量 {counter['after_quality']} "
              f"→ 去重 {counter['after_dedup']} → 最终 {counter['final']}")
    print(f"\n报告 → {report_path}\n下一步：python data_pipeline/04_mix_split.py")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
