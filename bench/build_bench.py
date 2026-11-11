# -*- coding: utf-8 -*-
"""构造四个节点的基准测试集（纯本地、零卡时）。

数据来源：
    n1_intent / n2_slot   data/test.jsonl（MASSIVE 中文子集）分层抽样
    n3_reply / n4_escalate bench/seeds/*.jsonl 人工种子（可 --synth 扩量，仅用于打通流程）

产物（每个节点一个目录）：
    bench/<node>/test.jsonl            结构化金标，本地打分用
    bench/<node>/dev.jsonl             调 prompt / 选检查点用
    bench/<node>/test.evalscope.jsonl  evalscope 直接吃（query/response 两列）
    bench/<node>/system.txt            system prompt，传给 evalscope 保证口径一致
    bench/stats.json / stats.md        类别、难度、长度分布与泄漏检查报告

用法（项目根目录）：
    python bench/build_bench.py                    # 默认配额
    python bench/build_bench.py --synth 300        # N3/N4 用模板扩到 300 条（仅流程验证）
    python bench/build_bench.py --n1-test 100 --n2-test 100 --dev 20   # 快速试跑
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import prompts  # noqa: E402

BENCH = BASE / "bench"
DATA = BASE / "data"
NODES = ["n1_intent", "n2_slot", "n3_reply", "n4_escalate"]
RATIOS = {"easy": 0.5, "medium": 0.3, "hard": 0.2}
MAX_CHARS = {"n1_intent": 24, "n2_slot": 256, "n3_reply": 192, "n4_escalate": 96}
BAN_WORDS = ["肯定", "绝对", "包您满意", "百分百"]


# ---------------------------------------------------------------- 基础工具


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def difficulty_of(text: str, slot_count: int) -> str:
    """启发式难度：文本越长、槽位越多越难。比随机打标签更有解释性，也方便复现。"""
    length = len(text)
    if length >= 18 or slot_count >= 3:
        return "hard"
    if length <= 9 and slot_count <= 1:
        return "easy"
    return "medium"


def grams(text: str, n: int = 8) -> set[str]:
    """字符级 n-gram。中文短句固定 13-gram 会切不出东西，默认 8。"""
    clean = "".join(text.split())
    if len(clean) <= n:
        return {clean} if clean else set()
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


# ---------------------------------------------------------------- 分层抽样


class Sampler:
    """按意图分层 + 按难度配比抽样，避免高频意图淹没长尾。

    两段式：
        1. 先每个意图至少取 1 条（长尾保护），再轮转补足；
        2. 在候选池里按 easy/medium/hard 配额挑选，实现难度分层。
    """

    def __init__(self, rows: list[dict], seed: int = 42) -> None:
        self.groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            self.groups[row["intent"]].append(row)
        rng = random.Random(seed)
        for items in self.groups.values():
            rng.shuffle(items)
        self.order = sorted(self.groups, key=lambda key: -len(self.groups[key]))

    def _round_robin(self, limit: int) -> list[dict]:
        taken: list[dict] = []
        cursors = {key: 0 for key in self.order}
        while len(taken) < limit:
            moved = False
            for key in self.order:
                if len(taken) >= limit:
                    break
                index = cursors[key]
                if index < len(self.groups[key]):
                    taken.append(self.groups[key][index])
                    cursors[key] = index + 1
                    moved = True
            if not moved:
                break
        for key in self.order:
            self.groups[key] = self.groups[key][cursors[key] :]
        return taken

    def take(self, total: int, ratios: dict[str, float] = RATIOS) -> list[dict]:
        candidates = self._round_robin(min(total * 4, 100000))
        by_difficulty: dict[str, list[dict]] = defaultdict(list)
        for row in candidates:
            by_difficulty[row["difficulty"]].append(row)

        picked: list[dict] = []
        for level, ratio in ratios.items():
            picked.extend(by_difficulty[level][: int(round(total * ratio))])

        if len(picked) < total:  # 某档不够时用其它档补齐，不硬凑难度
            chosen = {id(row) for row in picked}
            for row in candidates:
                if len(picked) >= total:
                    break
                if id(row) not in chosen:
                    picked.append(row)

        remaining = [row for row in candidates if id(row) not in {id(item) for item in picked}]
        for row in remaining:  # 未选中的候选还给池子，供 dev 集使用
            self.groups[row["intent"]].append(row)

        random.Random(7).shuffle(picked)
        return picked[:total]


# ---------------------------------------------------------------- 四个节点构造


def build_n1(rows: list[dict]) -> list[dict]:
    records = []
    for index, row in enumerate(rows, start=1):
        slots = row.get("slots") or []
        records.append(
            {
                "id": f"n1_{index:04d}",
                "node": "n1_intent",
                "difficulty": difficulty_of(row["text"], len(slots)),
                "category": row["intent"],
                "input": {"text": row["text"]},
                "gold": {"label": row["intent"]},
                "constraints": None,
                "synthetic": False,
            }
        )
    return records


def build_n2(rows: list[dict]) -> list[dict]:
    records = []
    for index, row in enumerate(rows, start=1):
        slots = row.get("slots") or []
        records.append(
            {
                "id": f"n2_{index:04d}",
                "node": "n2_slot",
                "difficulty": difficulty_of(row["text"], len(slots)),
                "category": row["intent"],
                "input": {"text": row["text"]},
                "gold": {"intent": row["intent"], "slots": slots},
                "constraints": None,
                "synthetic": False,
            }
        )
    return records


N3_SYNTH = [
    ("订单{oid}的退款正在审核中，预计1到3个工作日完成，请留意短信通知。", "refund_progress"),
    ("您好，订单{oid}已安排补发，物流单号会在今天更新，请您稍后查看。", "reship"),
    ("非常抱歉，订单{oid}的包裹在途中滞留，已为您登记加急处理。", "logistics_stuck"),
    ("您好，订单{oid}的发票已开具，稍后会发送到您预留的邮箱。", "invoice_done"),
    ("您好，订单{oid}的换货申请已受理，请在7天内寄回商品。", "exchange"),
    ("您好，订单{oid}的优惠活动已结束，当前价格以页面显示为准。", "promotion"),
    ("您好，订单{oid}需补充收件人手机号才能安排派送。", "info_missing"),
    ("您好，订单{oid}支持7天无理由退换，需保持商品完好。", "rule"),
]

N4_SYNTH = [
    (True, "用户要求转人工处理赔偿", [("user", "这事必须你们主管来处理，给我转人工")]),
    (True, "投诉且情绪激烈", [("user", "太气人了，我要投诉到底"), ("assistant", "请先冷静")]),
    (True, "涉及金额纠纷", [("user", "你们凭什么扣我钱"), ("assistant", "我帮您核实")]),
    (False, "常规物流查询", [("user", "我的包裹到哪个城市了"), ("assistant", "我帮您查询")]),
    (False, "商品参数咨询", [("user", "这个支持无线充电吗"), ("assistant", "该型号支持")]),
    (False, "发票问题", [("user", "发票抬头能开公司的吗"), ("assistant", "可以，请填写抬头")]),
    (False, "优惠券咨询", [("user", "券为什么用不了"), ("assistant", "限指定商品")]),
    (False, "退换货规则", [("user", "退货运费谁承担"), ("assistant", "质量问题我们承担")]),
]


def synth_n3(count: int, start: int) -> list[dict]:
    records = []
    for index in range(count):
        text, category = N3_SYNTH[index % len(N3_SYNTH)]
        oid = f"S{10000 + index}"
        records.append(
            {
                "id": f"n3_x{start + index:04d}",
                "node": "n3_reply",
                "difficulty": "medium",
                "category": category,
                "input": {"summary": "用户咨询订单进展", "action": category, "order_id": oid},
                "gold": {"text": text.format(oid=oid)},
                "constraints": {"max_chars": 80, "must_contain": [oid], "must_not_contain": BAN_WORDS},
                "synthetic": True,
            }
        )
    return records


def synth_n4(count: int, start: int) -> list[dict]:
    records = []
    for index in range(count):
        flag, reason, turns = N4_SYNTH[index % len(N4_SYNTH)]
        dialog = [{"role": role, "text": text} for role, text in turns]
        records.append(
            {
                "id": f"n4_x{start + index:04d}",
                "node": "n4_escalate",
                "difficulty": "medium",
                "category": "synthetic",
                "input": {"dialog": dialog},
                "gold": {"escalate": flag, "reason": reason},
                "constraints": None,
                "synthetic": True,
            }
        )
    return records


def build_from_seeds(node: str, total: int, synth: int, dev_size: int) -> tuple[list, list]:
    path = BENCH / "seeds" / ("n3_reply.jsonl" if node == "n3_reply" else "n4_escalate.jsonl")
    seeds = load_jsonl(path)
    random.Random(11).shuffle(seeds)

    dev_count = min(dev_size, max(len(seeds) // 4, 1))
    dev, rest = seeds[:dev_count], seeds[dev_count:]

    test = list(rest)
    if synth:
        maker = synth_n3 if node == "n3_reply" else synth_n4
        need = max(synth - len(test), 0)
        test += maker(need, start=len(test) + 1)

    if len(test) < total:
        print(f"[warn] {node} 只有 {len(test)} 条，未达目标 {total}；上卡后用 data_pipeline 扩写补齐")
    return test, dev


# ---------------------------------------------------------------- 校验与报告


def leak_check(records_by_node: dict[str, list[dict]], train_path: Path, n: int = 8, min_len: int = 16) -> dict:
    """与训练集做泄漏检查。

    两条判据分开看，避免 MASSIVE 这类高度模板化的语料产生大量假阳性：
        1. 精确重复：去掉空白后与训练集某条完全相同 → 真泄漏，必须在抽样阶段就剔除；
        2. n-gram 重叠：只在文本长度 >= 2n 时判定，短句（如"今天"）的 gram 集合
           只有一个元素，与训练集必然重合，没有判别意义。
    """
    train_grams: set[str] = set()
    train_texts: set[str] = set()
    for row in load_jsonl(train_path):
        train_grams |= grams(row["text"], n)
        train_texts.add("".join(row["text"].split()))

    report = {}
    for node, records in records_by_node.items():
        suspicious, exact = [], 0
        for record in records:
            text = record["input"].get("text") or json.dumps(record["input"], ensure_ascii=False)
            clean = "".join(text.split())
            if clean in train_texts:
                exact += 1
                continue
            if len(clean) < min_len:  # 太短，n-gram 判据不可靠
                continue
            hit = grams(text, n)
            overlap = len(hit & train_grams) / len(hit)
            if overlap >= 0.8:
                suspicious.append({"id": record["id"], "overlap": round(overlap, 2), "text": text[:40]})
        report[node] = {
            "checked": len(records),
            "exact_duplicate": exact,
            "suspicious": len(suspicious),
            "examples": suspicious[:5],
        }
    return report


def make_stats(name: str, records: list[dict]) -> dict:
    categories = Counter(record["category"] for record in records)
    difficulties = Counter(record["difficulty"] for record in records)
    lengths = [len(record["input"].get("text") or json.dumps(record["input"], ensure_ascii=False))
               for record in records]
    return {
        "node": name,
        "count": len(records),
        "synthetic": sum(1 for record in records if record.get("synthetic")),
        "categories": len(categories),
        "long_tail": sorted([key for key, value in categories.items() if value < 5]),
        "difficulty": dict(difficulties),
        "difficulty_ratio": {key: round(value / max(len(records), 1), 3)
                             for key, value in difficulties.items()},
        "length": {"min": min(lengths) if lengths else 0,
                   "avg": round(sum(lengths) / len(lengths), 1) if lengths else 0,
                   "max": max(lengths) if lengths else 0},
        "top_categories": categories.most_common(8),
    }


def export_evalscope(node: str, records: list[dict]) -> None:
    rows = []
    for record in records:
        item = prompts.build(node, record["input"], include_answer=True, gold=record["gold"])
        system = item["messages"][0]["content"]
        query = item["messages"][1]["content"]
        answer = item["messages"][-1]["content"]
        rows.append({"query": query, "response": answer})
        if record is records[0]:
            (BENCH / node / "system.txt").write_text(system, encoding="utf-8")
    dump_jsonl(BENCH / node / "test.evalscope.jsonl", rows)


def write_reports(stats: list[dict], leak: dict, meta: dict) -> None:
    (BENCH / "stats.json").write_text(
        json.dumps({"meta": meta, "nodes": stats, "leak": leak}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = ["# 基准集统计", "", f"- 生成时间：{meta['created_at']}", f"- 随机种子：{meta['seed']}"]
    if any(item["synthetic"] for item in stats):
        lines.append("- 注意：本版含模板合成样本，只用于打通评测流程；"
                     "正式基线必须换成人工筛选样本后重跑")
    lines.append("")
    for item in stats:
        lines += [
            f"## {item['node']}",
            "",
            f"- 条数：{item['count']}（模板合成 {item['synthetic']} 条）",
            f"- 类别数：{item['categories']}",
            f"- 难度分布：{item['difficulty']} → {item['difficulty_ratio']}",
            f"- 长度 min/avg/max：{item['length']['min']}/{item['length']['avg']}/{item['length']['max']}",
            f"- 长尾类（<5 条）：{', '.join(item['long_tail'][:12]) or '无'}",
            "",
        ]
    lines += ["## 泄漏检查（与 data/train.jsonl 的 8-gram 重叠）", ""]
    for node, info in leak.items():
        lines.append(f"- {node}：检查 {info['checked']} 条，疑似泄漏 {info['suspicious']} 条")
    (BENCH / "stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- 主流程


def main() -> None:
    parser = argparse.ArgumentParser(description="构造四节点基准集")
    parser.add_argument("--n1-test", type=int, default=400)
    parser.add_argument("--n2-test", type=int, default=400)
    parser.add_argument("--n3-test", type=int, default=300)
    parser.add_argument("--n4-test", type=int, default=300)
    parser.add_argument("--dev", type=int, default=50, help="每个节点的 dev 条数")
    parser.add_argument("--synth", type=int, default=0, help="用模板把 N3/N4 扩到的条数（仅流程验证）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = load_jsonl(DATA / "test.jsonl")
    train_texts = {"".join(row["text"].split()) for row in load_jsonl(DATA / "train.jsonl")}
    before = len(source)
    source = [row for row in source if "".join(row["text"].split()) not in train_texts]  # 剔精确重复
    if before != len(source):
        print(f"[info] 剔除与训练集完全重复的样本 {before - len(source)} 条")
    for row in source:  # 抽样按难度分层，原始数据没有这个字段，先算好
        row["difficulty"] = difficulty_of(row["text"], len(row.get("slots") or []))
    present = {row["intent"] for row in source}
    missing = [label for label in prompts.INTENTS if label not in present]
    extra = sorted(present - set(prompts.INTENTS))
    if missing or extra:
        print(f"[warn] 意图集不一致：数据缺 {missing[:5]}，数据多 {extra[:5]}")

    sampler = Sampler(source, seed=args.seed)
    # 两次独立 take：N1 与 N2 用不同样本，且各自都做长尾保护（每类至少 1 条）
    n1_rows = sampler.take(args.n1_test + args.dev)
    n2_rows = sampler.take(args.n2_test + args.dev)
    for name, rows, total in (("n1_intent", n1_rows, args.n1_test + args.dev),
                              ("n2_slot", n2_rows, args.n2_test + args.dev)):
        if len(rows) < total:
            print(f"[warn] {name} 可用样本 {len(rows)} 条，少于目标 {total} 条")

    n1_records = build_n1(n1_rows)
    n2_records = build_n2(n2_rows)

    n1_test, n1_dev = n1_records[: args.n1_test], n1_records[args.n1_test :]
    n2_test, n2_dev = n2_records[: args.n2_test], n2_records[args.n2_test :]

    n3_test, n3_dev = build_from_seeds("n3_reply", args.n3_test, args.synth, args.dev)
    n4_test, n4_dev = build_from_seeds("n4_escalate", args.n4_test, args.synth, args.dev)

    payload = {
        "n1_intent": (n1_test, n1_dev),
        "n2_slot": (n2_test, n2_dev),
        "n3_reply": (n3_test, n3_dev),
        "n4_escalate": (n4_test, n4_dev),
    }

    for node, (test, dev) in payload.items():
        dump_jsonl(BENCH / node / "test.jsonl", test)
        dump_jsonl(BENCH / node / "dev.jsonl", dev)
        export_evalscope(node, test)
        print(f"[ok] {node}: test={len(test)} dev={len(dev)}")

    stats = [make_stats(node, test) for node, (test, _) in payload.items()]
    leak = leak_check({node: test for node, (test, _) in payload.items()}, DATA / "train.jsonl")
    meta = {
        "seed": args.seed,
        "created_at": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_sha256": hashlib.sha256((DATA / "test.jsonl").read_bytes()).hexdigest()[:16],
    }
    write_reports(stats, leak, meta)

    total = sum(len(test) for test, _ in payload.values())
    print(f"\n合计 {total} 条 → {BENCH / 'stats.md'}")
    for node, info in leak.items():
        if info["suspicious"]:
            print(f"[warn] {node} 疑似泄漏 {info['suspicious']} 条：{info['examples'][:2]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
