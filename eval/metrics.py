# -*- coding: utf-8 -*-
"""四个业务节点的评测指标，全项目唯一实现。

谁在用：
    eval/score_bench.py     本地对预测文件打分
    data_pipeline/03_filter  合成数据质检（复用解析与幻觉判定）
    EvalScope 自定义 metric  如需注册，import 本文件即可，保证口径一致

设计原则：
    1. 只依赖标准库，任何环境都能跑（训练机、本地、CI）。
    2. 解析函数对模型输出宽容（容忍 Markdown 代码块、前后寒冰文字），
       但对"答案是否正确"严格（Joint EM 完全一致才算对）。
    3. 每个指标都返回 0~1 的浮点数，报告里统一乘 100 转成百分比。
"""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_QUOTES = "\"'`“”‘’「」"


def normalize(text: str) -> str:
    """去掉所有空白，用于槽位 value 与原文的比对（中文无需保留空格）。"""
    return re.sub(r"\s+", "", text or "")


def extract_json(text: str) -> dict | None:
    """从模型输出里抠出第一个 JSON 对象，失败返回 None。

    依次尝试：整体解析 → 代码块内解析 → 首个 '{' 到末个 '}' 之间的片段。
    """
    if not text:
        return None
    candidates = [text.strip()]
    fenced = _FENCE.findall(text)
    candidates.extend(block.strip() for block in fenced)

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            pass

    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def parse_intent(text: str, labels: list[str]) -> str | None:
    """从输出里识别意图标签；不在候选集内返回 None（计入非法输出率）。"""
    if not text:
        return None
    cleaned = text.strip().strip(_QUOTES + "。.,，:：")

    for label in labels:  # 先整体匹配，避免 "alarm_set" 被 "alarm_query" 的子串规则误伤
        if cleaned == label:
            return label

    obj = extract_json(cleaned)
    if isinstance(obj, dict):
        for key in ("intent", "label", "intent_label"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip() in labels:
                return value.strip()

    first_line = cleaned.splitlines()[0] if cleaned else ""
    for label in labels:  # 退化为"输出里出现的唯一标签"
        if re.search(rf"(?<![a-z_]){re.escape(label)}(?![a-z_])", first_line):
            return label
    return None


def parse_slots_answer(text: str) -> dict | None:
    """解析 N2 的结构化答案，返回 {"intent":..., "slots":[...]} 或 None。"""
    obj = extract_json(text)
    if obj is None:
        return None
    slots = obj.get("slots")
    if slots is None:
        return None
    if not isinstance(slots, list):
        return None
    cleaned = []
    for item in slots:
        if not isinstance(item, dict):
            continue
        name, value = item.get("name"), item.get("value")
        if isinstance(name, str) and isinstance(value, str):
            cleaned.append({"name": name.strip(), "value": value.strip()})
    intent = obj.get("intent")
    return {"intent": intent.strip() if isinstance(intent, str) else intent, "slots": cleaned}


def slot_set(slots) -> set[tuple[str, str]]:
    out = set()
    for item in slots or []:
        if isinstance(item, dict):
            out.add((str(item.get("name", "")).strip(), normalize(str(item.get("value", "")))))
    return out


def joint_exact_match(pred: dict | None, gold: dict) -> bool:
    """intent 与 slots（含顺序）完全一致才算对，与 eval/baseline_eval.py 口径一致。"""
    if pred is None:
        return False
    pred_norm = {"intent": pred.get("intent"), "slots": slot_list(pred.get("slots"))}
    gold_norm = {"intent": gold.get("intent"), "slots": slot_list(gold.get("slots"))}
    return pred_norm == gold_norm


def slot_list(slots) -> list[tuple[str, str]]:
    """把 slots 归一成有序 (name, value) 列表，用于顺序敏感比较。"""
    out = []
    for item in slots or []:
        if isinstance(item, dict):
            out.append((str(item.get("name", "")).strip(), normalize(str(item.get("value", "")))))
    return out


def slot_prf(pred_slots, gold_slots) -> tuple[float, float, float]:
    """槽位集合级 Precision / Recall / F1（无序，允许同名多值）。"""
    pred, gold = slot_set(pred_slots), slot_set(gold_slots)
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def hallucinated_slots(pred_slots, source_text: str) -> list[str]:
    """返回原文中找不到出处的槽位 value 列表（幻觉槽位）。"""
    source = normalize(source_text)
    bad = []
    for item in pred_slots or []:
        value = normalize(str(item.get("value", ""))) if isinstance(item, dict) else ""
        if value and value not in source:
            bad.append(str(item.get("value", "")))
    return bad


def macro_f1(gold_labels: list, pred_labels: list) -> float:
    labels = sorted({x for x in gold_labels + pred_labels if x is not None})
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = fp = fn = 0
        for gold, pred in zip(gold_labels, pred_labels):
            if pred == label and gold == label:
                tp += 1
            elif pred == label and gold != label:
                fp += 1
            elif pred != label and gold == label:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def binary_metrics(gold_flags: list[bool], pred_flags: list[bool]) -> dict:
    tp = sum(1 for g, p in zip(gold_flags, pred_flags) if g and p)
    fp = sum(1 for g, p in zip(gold_flags, pred_flags) if not g and p)
    fn = sum(1 for g, p in zip(gold_flags, pred_flags) if g and not p)
    tn = sum(1 for g, p in zip(gold_flags, pred_flags) if not g and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(gold_flags) if gold_flags else 0.0,
        # 漏判率：该转人工却没转，业务上代价最高
        "miss_rate": fn / (tp + fn) if tp + fn else 0.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def check_constraints(text: str, constraints: dict | None) -> tuple[bool, list[str]]:
    """N3 话术约束校验：长度、必备信息、禁用词。"""
    if not constraints:
        return True, []
    body = (text or "").strip()
    violations = []

    limit = constraints.get("max_chars")
    if isinstance(limit, int) and len(body) > limit:
        violations.append(f"超长({len(body)}>{limit})")

    for token in constraints.get("must_contain", []) or []:
        if token and token not in body:
            violations.append(f"缺少必备信息:{token}")

    for token in constraints.get("must_not_contain", []) or []:
        if token and token in body:
            violations.append(f"出现禁用词:{token}")

    return not violations, violations


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score_records(node: str, records: list[dict], labels: list[str] | None = None) -> dict:
    """对某节点的预测结果打分。

    records 每项：{"id", "gold", "input", "prediction"(字符串)}
    input 中需带原文的字段：N1/N2 用 input["text"]，其余节点可缺省。
    """
    if node == "n1_intent":
        return _score_n1(records, labels or [])
    if node == "n2_slot":
        return _score_n2(records)
    if node == "n3_reply":
        return _score_n3(records)
    if node == "n4_escalate":
        return _score_n4(records)
    raise ValueError(f"未知节点：{node}")


def _score_n1(records: list[dict], labels: list[str]) -> dict:
    gold_labels, pred_labels, invalid = [], [], 0
    for rec in records:
        gold = rec["gold"].get("label") or rec["gold"].get("intent")
        pred = parse_intent(rec.get("prediction") or "", labels)
        gold_labels.append(gold)
        pred_labels.append(pred)
        invalid += 1 if pred is None else 0
    total = len(records)
    correct = sum(1 for g, p in zip(gold_labels, pred_labels) if g == p and g is not None)
    return {
        "node": "n1_intent",
        "count": total,
        "metrics": {
            "accuracy": correct / total if total else 0.0,
            "macro_f1": macro_f1(gold_labels, pred_labels),
            "invalid_rate": invalid / total if total else 0.0,
        },
    }


def _score_n2(records: list[dict]) -> dict:
    joint, valid = 0, 0
    precisions, recalls, f1s, intent_hits = [], [], [], []
    slot_total, halluc_slots, halluc_samples = 0, 0, 0
    total = len(records)

    for rec in records:
        gold = rec["gold"]
        text = (rec.get("input") or {}).get("text", "")
        pred = parse_slots_answer(rec.get("prediction") or "")
        valid += 1 if pred is not None else 0

        if pred is not None:
            joint += 1 if joint_exact_match(pred, gold) else 0
            intent_hits.append(1.0 if pred.get("intent") == gold.get("intent") else 0.0)
            p, r, f = slot_prf(pred.get("slots"), gold.get("slots"))
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
            bad = hallucinated_slots(pred.get("slots"), text)
            slot_total += len(pred.get("slots") or [])
            halluc_slots += len(bad)
            halluc_samples += 1 if bad else 0

    return {
        "node": "n2_slot",
        "count": total,
        "metrics": {
            "joint_em": joint / total if total else 0.0,
            "json_valid_rate": valid / total if total else 0.0,
            "intent_acc": mean(intent_hits),
            "slot_precision": mean(precisions),
            "slot_recall": mean(recalls),
            "slot_f1": mean(f1s),
            "hallucination_slot_rate": halluc_slots / slot_total if slot_total else 0.0,
            "hallucination_sample_rate": halluc_samples / total if total else 0.0,
        },
    }


def _score_n3(records: list[dict]) -> dict:
    passed, lengths = 0, []
    violation_counter: dict[str, int] = {}
    for rec in records:
        text = (rec.get("prediction") or "").strip()
        ok, violations = check_constraints(text, rec.get("constraints"))
        passed += 1 if ok else 0
        lengths.append(len(text))
        for item in violations:
            key = item.split(":")[0]
            violation_counter[key] = violation_counter.get(key, 0) + 1
    total = len(records)
    return {
        "node": "n3_reply",
        "count": total,
        "metrics": {
            "constraint_pass_rate": passed / total if total else 0.0,
            "avg_chars": mean(lengths),
            "empty_rate": sum(1 for rec in records if not (rec.get("prediction") or "").strip()) / total
            if total
            else 0.0,
        },
        "violations": violation_counter,
    }


def _score_n4(records: list[dict]) -> dict:
    gold_flags, pred_flags, valid = [], [], 0
    reasons = 0
    for rec in records:
        gold = bool(rec["gold"].get("escalate"))
        obj = extract_json(rec.get("prediction") or "")
        flag = None
        if isinstance(obj, dict) and isinstance(obj.get("escalate"), bool):
            flag = obj["escalate"]
            reasons += 1 if str(obj.get("reason", "")).strip() else 0
        valid += 1 if flag is not None else 0
        gold_flags.append(gold)
        pred_flags.append(bool(flag))
    result = binary_metrics(gold_flags, pred_flags)
    result["json_valid_rate"] = valid / len(records) if records else 0.0
    result["reason_complete_rate"] = reasons / valid if valid else 0.0
    return {"node": "n4_escalate", "count": len(records), "metrics": result}


def format_report(result: dict) -> str:
    """把单个节点的打分结果渲染成 Markdown 表格行组。"""
    lines = [f"### {result['node']}（{result['count']} 条）", "", "| 指标 | 值 |", "| --- | --- |"]
    for key, value in result["metrics"].items():
        if isinstance(value, dict):
            continue
        if isinstance(value, float):
            lines.append(f"| {key} | {value * 100:.2f}% |")
        else:
            lines.append(f"| {key} | {value:.2f} |")
    confusion = result["metrics"].get("confusion")
    if confusion:
        lines.append(f"| confusion | TP={confusion['tp']} FP={confusion['fp']} "
                     f"FN={confusion['fn']} TN={confusion['tn']} |")
    for key, value in (result.get("violations") or {}).items():
        lines.append(f"| 违规:{key} | {value} 次 |")
    return "\n".join(lines)
