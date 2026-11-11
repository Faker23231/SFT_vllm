# -*- coding: utf-8 -*-
"""四个节点的提示词与 messages 构造，训练数据与评测数据共用这一份。

为什么要单独抽出来：
    prompt 一变，历史评测分数全部作废。所有脚本（build_bench / data_pipeline /
    finetune / run_evalscope）都从这里取，保证"训练时见过的格式 = 评测时的格式"。

N2 直接复用 src/messages.py 的 SYSTEM_PROMPT（与现有 SFT 实验一致），
N1/N3/N4 是本项目新增节点，prompt 写在下面。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.messages import SYSTEM_PROMPT as SLOT_SYSTEM_PROMPT  # noqa: E402

# 与 data/*.jsonl 实际出现的 59 类一致（MASSIVE 中文子集不含 cooking_query）；
# build_bench.py 会用数据里的真实标签集校验这份列表，不一致会告警
INTENTS = [
    "alarm_query", "alarm_remove", "alarm_set",
    "audio_volume_down", "audio_volume_mute", "audio_volume_other", "audio_volume_up",
    "calendar_query", "calendar_remove", "calendar_set",
    "cooking_recipe",
    "datetime_convert", "datetime_query",
    "email_addcontact", "email_query", "email_querycontact", "email_sendemail",
    "general_greet", "general_joke", "general_quirky",
    "iot_cleaning", "iot_coffee", "iot_hue_lightchange", "iot_hue_lightdim",
    "iot_hue_lightoff", "iot_hue_lighton", "iot_hue_lightup",
    "iot_wemo_off", "iot_wemo_on",
    "lists_createoradd", "lists_query", "lists_remove",
    "music_dislikeness", "music_likeness", "music_query", "music_settings",
    "news_query",
    "play_audiobook", "play_game", "play_music", "play_podcasts", "play_radio",
    "qa_currency", "qa_definition", "qa_factoid", "qa_maths", "qa_stock",
    "recommendation_events", "recommendation_locations", "recommendation_movies",
    "social_post", "social_query",
    "takeaway_order", "takeaway_query",
    "transport_query", "transport_taxi", "transport_ticket", "transport_traffic",
    "weather_query",
]

INTENT_SYSTEM_PROMPT = (
    "你是客服助手的意图识别模块。根据用户话语，从候选意图中选择唯一最匹配的一个。\n\n"
    "候选意图：\n"
    + "、".join(INTENTS)
    + "\n\n规则：\n"
    "1. 只输出一个意图标签，不要解释，不要加引号，不要输出 JSON 或 Markdown。\n"
    "2. 必须逐字使用候选意图中的一个标签，不得自造。\n"
    "3. 语义模糊时选择最接近的一个，不要输出多个。"
)

REPLY_SYSTEM_PROMPT = (
    "你是客服回复生成模块。根据给定的事实与处置结论，生成一条可直接发给用户的话术。\n\n"
    "规则：\n"
    "1. 只输出话术正文，不要解释过程，不要 Markdown，不要输出 JSON。\n"
    "2. 不超过 80 个汉字。\n"
    "3. 若给出订单号，必须原样包含在话术中。\n"
    "4. 只能承诺 action 中已授权的处置，不得新增补偿、退款额度或时限。\n"
    "5. 禁止使用“肯定”“绝对”“包您满意”“百分百”等过度承诺用语。\n"
    "6. 语气礼貌简洁，先回应诉求再给出结论。"
)

ESCALATE_SYSTEM_PROMPT = (
    "你是客服对话的风险判定模块。根据对话片段判断是否需要转人工。\n\n"
    "判定为 true 的情形（满足任一即可）：\n"
    "1. 用户明确要求转人工或投诉。\n"
    "2. 涉及退款争议、金额纠纷、法律风险、人身安全。\n"
    "3. 同一问题已连续两轮未解决。\n"
    "4. 用户情绪激烈（辱骂、威胁、重复感叹）。\n\n"
    "输出要求：只输出一行 JSON，不要 Markdown 代码块，不要额外解释。\n"
    '{"escalate": true 或 false, "reason": "不超过 20 字的理由"}'
)


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def build_n1(inputs: dict, include_answer: bool = False, gold: dict | None = None) -> dict:
    messages = [
        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"用户文本：{inputs['text']}"},
    ]
    if include_answer and gold is not None:
        messages.append({"role": "assistant", "content": str(gold["label"])})
    return {"messages": messages}


def build_n2(inputs: dict, include_answer: bool = False, gold: dict | None = None) -> dict:
    messages = [
        {"role": "system", "content": SLOT_SYSTEM_PROMPT},
        {"role": "user", "content": f"用户文本：{inputs['text']}"},
    ]
    if include_answer and gold is not None:
        messages.append(
            {"role": "assistant", "content": _dump({"intent": gold["intent"], "slots": gold["slots"]})}
        )
    return {"messages": messages}


def build_n3(inputs: dict, include_answer: bool = False, gold: dict | None = None) -> dict:
    facts = {
        "用户诉求": inputs.get("summary", ""),
        "处置结论": inputs.get("action", ""),
        "订单号": inputs.get("order_id") or "无",
    }
    messages = [
        {"role": "system", "content": REPLY_SYSTEM_PROMPT},
        {"role": "user", "content": "已知信息：\n" + _dump(facts)},
    ]
    if include_answer and gold is not None:
        messages.append({"role": "assistant", "content": str(gold["text"])})
    return {"messages": messages}


def build_n4(inputs: dict, include_answer: bool = False, gold: dict | None = None) -> dict:
    dialog = inputs.get("dialog") or []
    lines = []
    for turn in dialog:
        role = "用户" if turn.get("role") == "user" else "客服"
        lines.append(f"{role}：{turn.get('text', '')}")
    messages = [
        {"role": "system", "content": ESCALATE_SYSTEM_PROMPT},
        {"role": "user", "content": "对话片段：\n" + "\n".join(lines)},
    ]
    if include_answer and gold is not None:
        messages.append(
            {"role": "assistant", "content": _dump({"escalate": gold["escalate"], "reason": gold.get("reason", "")})}
        )
    return {"messages": messages}


BUILDERS = {
    "n1_intent": build_n1,
    "n2_slot": build_n2,
    "n3_reply": build_n3,
    "n4_escalate": build_n4,
}

# 评测时的输出长度上限，run_evalscope.sh 与本地推理共用，避免两处不一致
MAX_TOKENS = {"n1_intent": 24, "n2_slot": 256, "n3_reply": 192, "n4_escalate": 96}


def build(node: str, inputs: dict, include_answer: bool = False, gold: dict | None = None) -> dict:
    return BUILDERS[node](inputs, include_answer=include_answer, gold=gold)


if __name__ == "__main__":
    # 手动检查四个节点的实际输入，改 prompt 后跑一下确认格式
    samples = {
        "n1_intent": {"text": "星期五早上九点叫醒我"},
        "n2_slot": {"text": "星期五早上九点叫醒我"},
        "n3_reply": {"summary": "用户申请退款", "action": "订单已发货，无法退款，说明退换货规则", "order_id": "A12345"},
        "n4_escalate": {"dialog": [{"role": "user", "text": "我要投诉！你们的东西有问题"},
                                   {"role": "assistant", "text": "请描述具体问题"}]},
    }
    golds = {
        "n1_intent": {"label": "alarm_set"},
        "n2_slot": {"intent": "alarm_set", "slots": [{"name": "date", "value": "星期五"}]},
        "n3_reply": {"text": "您好，订单 A12345 已发货，暂不支持退款，可签收后按退换货规则申请。"},
        "n4_escalate": {"escalate": True, "reason": "用户投诉，情绪激烈"},
    }
    for node, inputs in samples.items():
        item = build(node, inputs, include_answer=True, gold=golds[node])
        print(f"===== {node} =====")
        for message in item["messages"]:
            print(f"[{message['role']}] {message['content'][:120]}...")
        print()
