# -*- coding: utf-8 -*-
"""第一步：准备种子数据。

    N1/N2：从 data/train.jsonl 按意图分层抽真实样本（每类 N 条）
    N3/N4：data_pipeline/seeds/*.jsonl 人工种子（与 bench 评测集场景刻意错开）

种子必须来自真实分布，不用模型生成——模型生成的种子会让后续"扩写"变成同义反复，
评测时看不出真实提升。

用法：
    python data_pipeline/01_seed.py --per-class 8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

DATA = BASE / "data"
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


def bench_source_text(node: str) -> str:
    """把评测集的输入拼成文本，用于种子/评测重叠检查。"""
    path = BENCH / node / "test.jsonl"
    if not path.exists():
        return ""
    return "\n".join(json.dumps(row["input"], ensure_ascii=False) for row in load_jsonl(path))


def build_massive_seeds(per_class: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in load_jsonl(DATA / "train.jsonl"):
        groups[row["intent"]].append(row)

    seeds = []
    for intent in sorted(groups):
        for index, row in enumerate(groups[intent][:per_class]):
            input_text = {"text": row["text"]}
            seeds.append(
                {
                    "id": f"n1_{intent}_{index:02d}",
                    "node": "n1_intent",
                    "category": intent,
                    "input": input_text,
                    "gold": {"label": intent},
                    "constraints": None,
                }
            )
            seeds.append(
                {
                    "id": f"n2_{intent}_{index:02d}",
                    "node": "n2_slot",
                    "category": intent,
                    "input": input_text,
                    "gold": {"intent": intent, "slots": row.get("slots") or []},
                    "constraints": None,
                }
            )
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="生成种子池")
    parser.add_argument("--per-class", type=int, default=8, help="N1/N2 每类抽多少条真实样本")
    parser.add_argument("--out-dir", default=str(PIPE / "out" / "seeds"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    payload: dict[str, list[dict]] = defaultdict(list)
    for seed in build_massive_seeds(args.per_class):
        payload[seed["node"]].append(seed)

    for node, filename in (("n3_reply", "n3_reply_train.jsonl"), ("n4_escalate", "n4_escalate_train.jsonl")):
        for row in load_jsonl(PIPE / "seeds" / filename):
            payload[node].append({**row, "constraints": row.get("constraints")})

    for node, seeds in payload.items():
        dump_jsonl(out_dir / f"{node}.jsonl", seeds)
        print(f"[ok] {node}: {len(seeds)} 条种子")

    # 种子与评测集重叠检查：种子进了训练，就不能出现在 bench 里。
    # 用"命中比例"而非"是否有交集"——短句只有一个 gram，有交集是必然的，没有判别意义。
    for node in payload:
        bench_grams = set()
        path = BENCH / node / "test.jsonl"
        if path.exists():
            for row in load_jsonl(path):
                bench_grams |= grams(json.dumps(row["input"], ensure_ascii=False))
        overlap = 0
        for seed in payload[node]:
            text = json.dumps(seed["input"], ensure_ascii=False)
            hit = grams(text)
            if hit and bench_grams and len(hit & bench_grams) / len(hit) >= 0.8:
                overlap += 1
        if overlap:
            print(f"[warn] {node}: {overlap} 条种子与评测集高度重叠，请人工确认")
        else:
            print(f"[ok] {node}: 与评测集无高度重叠")

    print(f"\n种子目录：{out_dir}\n下一步：python data_pipeline/02_expand.py --backend mock")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
