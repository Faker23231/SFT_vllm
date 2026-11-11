# -*- coding: utf-8 -*-
"""在 GPU 实例上驱动 EvalScope：四个节点评测 + 通用能力回归 + 性能压测。

为什么用 Python 而不是 shell：
    --dataset-args 要传 JSON，system prompt 里有引号和中文，shell 拼字符串极易出错；
    这里统一用 json.dumps 构造，顺手支持 --dry-run 在本地打印命令做预检（零卡时）。

本地预检（不装 evalscope 也能跑）：
    python eval/run_evalscope.py --dry-run

实例上执行：
    python eval/run_evalscope.py --model qwen3-14b --api-url http://127.0.0.1:8000/v1/chat/completions \
        --tag baseline --general --perf

版本提示：
    evalscope 的 --eval-type 在不同版本里可能是 openai_api / openai_api_models，
    首次执行前先 `evalscope eval --help` 确认；本脚本失败时会原样打印 stderr。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from bench.prompts import MAX_TOKENS  # noqa: E402

BENCH = BASE / "bench"
NODES = ["n1_intent", "n2_slot", "n3_reply", "n4_escalate"]


def read_system(node: str) -> str:
    path = BENCH / node / "system.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def eval_command(node: str, args, work_dir: Path) -> list[str]:
    if not (BENCH / node / "test.evalscope.jsonl").exists():
        raise SystemExit(f"缺少 {node} 的 test.evalscope.jsonl，先跑 bench/build_bench.py")

    dataset_args = {
        "general_qa": {
            "local_path": str(BENCH / node / "test.evalscope.jsonl"),
            # 与训练/服务完全相同的 system prompt，缺了这项评测就不可比
            "system_prompt": read_system(node),
        }
    }
    generation = {"temperature": 0.0, "max_tokens": MAX_TOKENS[node], "do_sample": False}

    command = [
        "evalscope", "eval",
        "--model", args.model,
        "--api-url", args.api_url,
        "--api-key", args.api_key,
        "--eval-type", args.eval_type,
        "--datasets", "general_qa",
        "--dataset-args", json.dumps(dataset_args, ensure_ascii=False),
        "--generation-config", json.dumps(generation, ensure_ascii=False),
        "--work-dir", str(work_dir / node),
    ]
    if args.limit:
        command += ["--limit", str(args.limit)]
    return command


def general_command(dataset: str, args, work_dir: Path) -> list[str]:
    return [
        "evalscope", "eval",
        "--model", args.model,
        "--api-url", args.api_url,
        "--api-key", args.api_key,
        "--eval-type", args.eval_type,
        "--datasets", dataset,
        "--limit", str(args.general_limit),
        "--work-dir", str(work_dir / f"general_{dataset}"),
    ]


def perf_command(args, work_dir: Path) -> list[str]:
    base_url = args.api_url.rsplit("/v1/", 1)[0]
    command = [
        "evalscope", "perf",
        "--model", args.model,
        "--url", f"{base_url}/v1/chat/completions",
        "--api", "openai",
        "--dataset", "openqa",
        "--parallel", *[str(value) for value in args.parallel],
        "--number", str(args.number),
        "--max-tokens", "256",
        "--min-prompt-length", "128",
        "--max-prompt-length", "1024",
    ]
    if args.dry_run:  # perf 只支持写文件输出，dry-run 时仅打印
        command += ["--output-dir", str(work_dir / "perf")]
    return command


def run(command: list[str], dry_run: bool) -> bool:
    print("\n$ " + " ".join(command))
    if dry_run:
        return True
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[fail] 退出码 {result.returncode}：{command[0]} {command[1] if len(command) > 1 else ''}")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="EvalScope 评测驱动")
    parser.add_argument("--model", default="qwen3-14b")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--eval-type", default="openai_api")
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--work-dir", default=str(BASE / "reports" / "evalscope"))
    parser.add_argument("--node", default="all", choices=["all"] + NODES)
    parser.add_argument("--limit", type=int, default=0, help="每节点只评测前 N 条，0 表示全量")
    parser.add_argument("--general", action="store_true", help="加跑 gsm8k/ceval 通用能力回归")
    parser.add_argument("--general-limit", type=int, default=200)
    parser.add_argument("--perf", action="store_true", help="跑并发压测")
    parser.add_argument("--parallel", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--number", type=int, default=50, help="每档并发的请求数")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    args = parser.parse_args()

    work_dir = Path(args.work_dir) / args.tag
    work_dir.mkdir(parents=True, exist_ok=True)

    nodes = NODES if args.node == "all" else [args.node]
    results = []
    for node in nodes:
        results.append((node, run(eval_command(node, args, work_dir), args.dry_run)))

    if args.general:
        for dataset in ("gsm8k", "ceval"):
            results.append((dataset, run(general_command(dataset, args, work_dir), args.dry_run)))

    if args.perf:
        results.append(("perf", run(perf_command(args, work_dir), args.dry_run)))

    failed = [name for name, ok in results if not ok]
    print("\n===== 汇总 =====")
    for name, ok in results:
        print(f"  {name:<12} {'OK' if ok else 'FAIL'}")
    if failed:
        print(f"\n失败项：{failed}；检查 API 是否可达、--eval-type 是否与当前 evalscope 版本一致")

    print(f"\n结果目录：{work_dir}")
    print("下一步：")
    print(f"  python eval/evalscope_to_pred.py --work-dir {work_dir} --tag {args.tag}")
    print(f"  python eval/score_bench.py --pred-dir reports/preds/{args.tag} --tag {args.tag}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
