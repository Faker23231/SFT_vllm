# -*- coding: utf-8 -*-
"""第二步：种子扩写（两轮推理）。

    第 1 轮：按"改写要求"把种子输入改写成新的表达（语义不变）
    第 2 轮：让模型对改写后的输入重新标注，作为 03_filter 的一致性检验依据

三个后端（同一份产物格式，随时切换）：
    mock    本地规则变换 + 直接沿用种子标签，零卡时，用来验证整条流水线
    vllm    vllm.LLM 离线批量推理，实例上的正式方案（吞吐最高）
    openai  并发调 OpenAI 兼容接口，适合"服务已经起着、顺手合成"

用法：
    python data_pipeline/02_expand.py --backend mock --times 2          # 本地验证
    python data_pipeline/02_expand.py --backend vllm --model /models/qwen3-14b --times 6
    python data_pipeline/02_expand.py --backend openai --api-url http://127.0.0.1:8000/v1/chat/completions
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from bench import prompts  # noqa: E402

PIPE = BASE / "data_pipeline"
STYLES = [
    "改写成口语化表达，带语气词，但不改变原意",
    "加入轻微错别字或输入法错误，模拟真实用户快速打字",
    "尽量简短，省略主语或部分修饰成分",
    "加入冗余的背景铺垫，但结论不变",
]
MAX_TOKENS = {"n1_intent": 24, "n2_slot": 256, "n3_reply": 192, "n4_escalate": 96}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 改写目标定位


def rewrite_target(node: str, record: dict) -> str:
    """每种节点只改写"用户说的话"，gold 与 action 保持不变。"""
    if node in ("n1_intent", "n2_slot"):
        return record["input"]["text"]
    if node == "n3_reply":
        return record["input"].get("summary", "")
    dialog = record["input"].get("dialog") or []
    for turn in dialog:
        if turn.get("role") == "user":
            return turn.get("text", "")
    return ""


def apply_rewrite(node: str, record: dict, new_text: str) -> dict:
    import copy

    updated = copy.deepcopy(record)
    if node in ("n1_intent", "n2_slot"):
        updated["input"]["text"] = new_text
    elif node == "n3_reply":
        updated["input"]["summary"] = new_text
    else:
        for turn in updated["input"].get("dialog") or []:
            if turn.get("role") == "user":
                turn["text"] = new_text
                break
    return updated


def rewrite_messages(node: str, text: str, style: str) -> list[dict]:
    system = (
        "你是数据增强助手。把用户给定的一句话改写为新的表达，要求：只改变说法，"
        "不改变意图、不改变关键信息（时间、地点、对象、数量）、不增删要点、不解释。"
        "只输出改写后的一句话，不要引号、不要前缀说明。"
    )
    user = f"改写要求：{style}\n原文：{text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def annotate_messages(node: str, record: dict) -> list[dict]:
    return prompts.build(node, record["input"], include_answer=False)["messages"]


# ---------------------------------------------------------------- mock 后端


MOCK_PREFIX = ["请问一下，", "那个，", "麻烦问下，", "我想问，"]
MOCK_SUFFIX = ["啊", "呢", "吧", "谢谢"]


def mock_rewrite(text: str, style_index: int) -> str:
    rng = random.Random(hash((text, style_index)) & 0xFFFF)
    head = MOCK_PREFIX[style_index % len(MOCK_PREFIX)]
    tail = MOCK_SUFFIX[style_index % len(MOCK_SUFFIX)]
    body = text.rstrip("。！？!?")
    if style_index % 4 == 2 and len(body) > 6:  # 简短风格：砍掉前 2 个字
        body = body[2:]
    if rng.random() < 0.5:
        return f"{head}{body}"
    return f"{body}{tail}"


def gold_text(node: str, gold: dict) -> str:
    if node == "n1_intent":
        return gold["label"]
    if node == "n2_slot":
        return json.dumps({"intent": gold["intent"], "slots": gold["slots"]}, ensure_ascii=False)
    if node == "n3_reply":
        return gold["text"]
    return json.dumps({"escalate": gold["escalate"], "reason": gold.get("reason", "")}, ensure_ascii=False)


def run_mock(seeds: list[dict], times: int) -> list[dict]:
    candidates = []
    for record in seeds:
        node = record["node"]
        text = rewrite_target(node, record)
        for index in range(times):
            style = index % len(STYLES)
            new_text = mock_rewrite(text, index)
            candidate = apply_rewrite(node, record, new_text)
            candidates.append(
                {
                    "id": f"{record['id']}_m{index}",
                    "node": node,
                    "seed_id": record["id"],
                    "style": style,
                    "input": candidate["input"],
                    "gold": record["gold"],
                    "constraints": record.get("constraints"),
                    # mock 直接给正确答案，模拟"标注一致"的理想情况，跑通链路用
                    "raw": gold_text(node, record["gold"]),
                }
            )
    return candidates


# ---------------------------------------------------------------- vLLM 后端


def run_vllm(seeds: list[dict], times: int, args) -> list[dict]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization)
    sampling = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

    def render(messages: list[dict]) -> str:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
        except TypeError:  # 老版本 transformers 没有 enable_thinking
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 第 1 轮：批量改写
    jobs = []
    for record in seeds:
        text = rewrite_target(record["node"], record)
        for index in range(times):
            jobs.append((record, index, STYLES[index % len(STYLES)], text))

    prompts_text = [render(rewrite_messages(record["node"], text, style))
                    for record, _, style, text in jobs]
    started = time.time()
    outputs = llm.generate(prompts_text, sampling)
    elapsed = time.time() - started
    token_count = sum(len(item.outputs[0].token_ids) for item in outputs)
    print(f"[vllm] 改写 {len(jobs)} 条，耗时 {elapsed:.1f}s，约 {token_count / max(elapsed, 1):.0f} tok/s")

    candidates = []
    for (record, index, style, original), output in zip(jobs, outputs):
        new_text = output.outputs[0].text.strip().splitlines()[0].strip()
        if not new_text:
            new_text = original
        candidate = apply_rewrite(record["node"], record, new_text)
        candidates.append(
            {
                "id": f"{record['id']}_v{index}",
                "node": record["node"],
                "seed_id": record["id"],
                "style": index % len(STYLES),
                "input": candidate["input"],
                "gold": record["gold"],
                "constraints": record.get("constraints"),
                "raw": "",
            }
        )

    # 第 2 轮：对改写结果重新标注
    annotate_prompts = [render(annotate_messages(item["node"], item)) for item in candidates]
    sampling_ann = SamplingParams(temperature=0.0, max_tokens=args.annotate_max_tokens)
    outputs = llm.generate(annotate_prompts, sampling_ann)
    for item, output in zip(candidates, outputs):
        item["raw"] = output.outputs[0].text.strip()
    return candidates


# ---------------------------------------------------------------- OpenAI 后端


def chat(api_url: str, api_key: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
    payload = json.dumps(
        {"model": "default", "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    ).encode("utf-8")
    request = urllib.request.Request(
        api_url, data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def run_openai(seeds: list[dict], times: int, args) -> list[dict]:
    jobs = []
    for record in seeds:
        text = rewrite_target(record["node"], record)
        for index in range(times):
            jobs.append((record, index, STYLES[index % len(STYLES)], text))

    def do_rewrite(job):
        record, index, style, text = job
        try:
            new_text = chat(args.api_url, args.api_key, rewrite_messages(record["node"], text, style), 512, 0.8)
            new_text = new_text.splitlines()[0].strip() or text
        except Exception as exc:  # 网络抖动不中断整批
            print(f"[warn] 改写失败 {record['id']}#{index}: {exc}")
            new_text = text
        return record, index, style, new_text

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rewritten = list(pool.map(do_rewrite, jobs))

    candidates = []
    for record, index, style, new_text in rewritten:
        candidate = apply_rewrite(record["node"], record, new_text)
        candidates.append(
            {
                "id": f"{record['id']}_o{index}",
                "node": record["node"],
                "seed_id": record["id"],
                "style": style,
                "input": candidate["input"],
                "gold": record["gold"],
                "constraints": record.get("constraints"),
                "raw": "",
            }
        )

    def do_annotate(item):
        try:
            item["raw"] = chat(args.api_url, args.api_key,
                               annotate_messages(item["node"], item),
                               args.annotate_max_tokens, 0.0)
        except Exception as exc:
            print(f"[warn] 标注失败 {item['id']}: {exc}")
            item["raw"] = ""
        return item

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(do_annotate, candidates))
    return candidates


# ---------------------------------------------------------------- 主流程


def main() -> None:
    parser = argparse.ArgumentParser(description="种子扩写")
    parser.add_argument("--backend", default="mock", choices=["mock", "vllm", "openai"])
    parser.add_argument("--seeds-dir", default=str(PIPE / "out" / "seeds"))
    parser.add_argument("--out", default=str(PIPE / "out" / "candidates.jsonl"))
    parser.add_argument("--times", type=int, default=2, help="每条种子生成几份改写")
    parser.add_argument("--limit", type=int, default=0, help="只用前 N 条种子（先小样本估时长）")
    parser.add_argument("--model", default="/root/autodl-tmp/models/qwen3-14b")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--annotate-max-tokens", type=int, default=256)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    seeds: list[dict] = []
    for path in sorted(Path(args.seeds_dir).glob("*.jsonl")):
        rows = load_jsonl(path)
        # N3/N4 人工种子少，自动多扩几倍，避免合成集被 N1/N2 淹没
        factor = 5 if rows and rows[0]["node"] in ("n3_reply", "n4_escalate") else 1
        for record in rows:
            seeds.extend([record] * factor)
        print(f"[seed] {path.name}: {len(rows)} 条 × {factor}")

    if args.limit:
        seeds = seeds[: args.limit]
    print(f"[seed] 合计 {len(seeds)} 条，预计候选 {len(seeds) * args.times} 条")

    started = time.time()
    if args.backend == "mock":
        candidates = run_mock(seeds, args.times)
    elif args.backend == "vllm":
        candidates = run_vllm(seeds, args.times, args)
    else:
        candidates = run_openai(seeds, args.times, args)

    for candidate in candidates:  # 记录来源后端，04 会据此拦截 mock 数据进训练
        candidate["backend"] = args.backend
    dump_jsonl(Path(args.out), candidates)
    print(f"[ok] 候选 {len(candidates)} 条，耗时 {time.time() - started:.1f}s → {args.out}")
    print("下一步：python data_pipeline/03_filter.py")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
