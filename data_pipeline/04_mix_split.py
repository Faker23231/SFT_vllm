# -*- coding: utf-8 -*-
"""第四步：真实/合成配比 + 切分 + 转成训练格式（纯本地）。

产出直接给 finetune/04_domain_qlora.py 吃：
    data/domain_sft_<name>_train.jsonl        {"messages": [...]}
    data/domain_sft_<name>_validation.jsonl   同上
    data/domain_sft_<name>_manifest.json      配比、条数、去重统计、哈希

配比只做两档（1:1 与 1:3），对应消融实验里的"合成数据的边际收益"。

用法：
    python data_pipeline/04_mix_split.py --real-per-class 20 --ratios 1:1 1:3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from bench import prompts  # noqa: E402

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


def build_real(per_class: int) -> list[dict]:
    """真实样本：MASSIVE 训练集按意图分层抽样，同时产出 N1/N2 两个任务。"""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in load_jsonl(DATA / "train.jsonl"):
        groups[row["intent"]].append(row)

    records = []
    for intent in sorted(groups):
        for index, row in enumerate(groups[intent][:per_class]):
            text = {"text": row["text"]}
            records.append(
                {
                    "id": f"real_n1_{intent}_{index}",
                    "node": "n1_intent",
                    "category": intent,
                    "input": text,
                    "gold": {"label": intent},
                    "constraints": None,
                }
            )
            records.append(
                {
                    "id": f"real_n2_{intent}_{index}",
                    "node": "n2_slot",
                    "category": intent,
                    "input": text,
                    "gold": {"intent": intent, "slots": row.get("slots") or []},
                    "constraints": None,
                }
            )
    return records


def to_messages(record: dict) -> dict:
    item = prompts.build(record["node"], record["input"], include_answer=True, gold=record["gold"])
    return {"messages": item["messages"], "node": record["node"], "category": record["category"]}


def leak_against_bench(records: list[dict]) -> dict:
    """训练集与评测集的 8-gram 命中情况，作为上线前的最后一道检查。"""
    report = {}
    for node in ("n1_intent", "n2_slot", "n3_reply", "n4_escalate"):
        path = BENCH / node / "test.jsonl"
        if not path.exists():
            continue
        pool: set[str] = set()
        for row in load_jsonl(path):
            pool |= grams(json.dumps(row["input"], ensure_ascii=False))
        subset = [row for row in records if row["node"] == node]
        hits = 0
        for row in subset:
            item = grams(json.dumps(row["input"], ensure_ascii=False))
            if item and len(item & pool) / len(item) >= 0.8:
                hits += 1
        report[node] = {"checked": len(subset), "suspicious": hits}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="配比与切分")
    parser.add_argument("--filtered-dir", default=str(PIPE / "out" / "filtered"))
    parser.add_argument("--real-per-class", type=int, default=20, help="每类真实样本数")
    parser.add_argument("--ratios", nargs="+", default=["1:1", "1:3"])
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--allow-short", action="store_true",
                        help="合成不足时仍按实际数量产出（只用于验证格式，正式训练不要开）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    real = build_real(args.real_per_class)
    synthetic: list[dict] = []
    for path in sorted(Path(args.filtered_dir).glob("*.jsonl")):
        for row in load_jsonl(path):
            synthetic.append({**row, "category": row.get("category")
                              or row["gold"].get("label") or row["gold"].get("intent") or "unknown"})
    print(f"[in] 真实 {len(real)} 条，合成 {len(synthetic)} 条")
    if not synthetic:
        print("[warn] 合成集为空：先跑 02_expand.py 与 03_filter.py")
    backends = {row.get("backend") for row in synthetic}
    if backends == {"mock"}:
        print("[warn] 合成集来自 mock 后端（规则模板改写），只能验证流程；"
              "正式训练前必须用 --backend vllm 重跑 02_expand.py")

    rng = random.Random(args.seed)
    for ratio in args.ratios:
        real_part, synth_part = (int(x) for x in ratio.split(":"))
        need = int(len(real) * synth_part / real_part)
        pool = synthetic[:]
        rng.shuffle(pool)
        chosen = pool[:need]
        if len(chosen) < need:
            message = f"[warn] 配比 {ratio}: 需要合成 {need} 条，实际只有 {len(chosen)} 条"
            if not args.allow_short:
                # 合成不足时硬产出，会让消融误以为做了 1:3，这里直接跳过更诚实
                print(message + "，跳过该配比（加 --allow-short 可强制产出，仅用于格式验证）")
                continue
            print(message + "，按实际数量产出（manifest 里 actual_ratio 已如实记录）")

        mixed = real + chosen
        rng.shuffle(mixed)
        cut = int(len(mixed) * (1 - args.val_ratio))
        train, validation = mixed[:cut], mixed[cut:]

        name = f"{real_part}to{synth_part}"
        dump_jsonl(DATA / f"domain_sft_{name}_train.jsonl", [to_messages(row) for row in train])
        dump_jsonl(DATA / f"domain_sft_{name}_validation.jsonl", [to_messages(row) for row in validation])

        manifest = {
            "name": name,
            "ratio": ratio,
            "actual_ratio": f"{len(real)}:{len(chosen)}",  # 合成不足时会小于目标配比
            "real": len(real),
            "synthetic_available": len(synthetic),
            "synthetic_used": len(chosen),
            "synthetic_backend": sorted(backends),
            "train": len(train),
            "validation": len(validation),
            "nodes": dict(Counter(row["node"] for row in mixed)),
            "categories": len({row["category"] for row in mixed}),
            "leak_check": leak_against_bench(mixed),
            "sha256_train": hashlib.sha256(
                (DATA / f"domain_sft_{name}_train.jsonl").read_bytes()
            ).hexdigest()[:16],
            "seed": args.seed,
        }
        (DATA / f"domain_sft_{name}_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[ok] {name}: train {len(train)} / val {len(validation)} 条 → data/domain_sft_{name}_*.jsonl")

    print("\n下一步（实例上）：python finetune/04_domain_qlora.py --dataset data/domain_sft_1to1_train.jsonl")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
