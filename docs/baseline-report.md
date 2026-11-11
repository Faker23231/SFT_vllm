# S1 基线报告（vLLM + EvalScope，Qwen3-14B / A100 40G）

> 采集时间：2026-11-11
> 口径：`temperature=0`、`do_sample=false`、关闭 thinking（`deploy/chat_template_qwen3_nothink.jinja`）、
> system prompt 取自 `bench/<node>/system.txt`，打分统一走 `eval/metrics.py`。

## 1. 环境（对应 docs/env-matrix.md 的 S1 段）

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA A100-PCIE-40GB（40960 MiB） |
| 驱动 / nvcc | 560.35.03 / CUDA 12.8 (V12.8.93) |
| python / venv | 3.12.13（`/data/venv`，uv 镜像装不上，改用 pip + 腾讯云镜像） |
| torch | 2.9.0+cu128 |
| vllm | 0.11.2 |
| transformers / datasets | 4.57.6 / 4.8.4 |
| evalscope | 1.14.1（压测需 `evalscope[perf]`） |
| peft / trl / accelerate | **未安装**，S2 训练前需补 |

**踩坑记录（直接影响选型）**：默认 `pip install vllm` 装到 0.33.0，带的是 CUDA 13 版 torch，
在 560 驱动上 `torch.cuda.is_available()` 直接 False（`found version 12060`）。
必须钉到 **vllm 0.11.2（torch 2.9.0+cu128）** 才能在 CUDA 12.6 驱动上跑。

## 2. vLLM 启动日志关键四数（bf16 基线）

| 项 | 值 | 解读 |
| --- | --- | --- |
| `GPU KV cache size` | **50,608 tokens** | 36G（0.90×40G）里扣掉 28G 权重后剩下的 KV 预算 |
| `Maximum concurrency for 8,192 tokens/request` | **6.18x** | `max-num-seqs=32` 只是上限，实际并发被 KV 卡在 6 左右 |
| attention backend | **FLASH_ATTN** | sm80 上走 FlashAttention，没落到 FlashInfer |
| 权重显存 | ~28G | 14B bf16 |

结论：**8K 上下文下 32 并发物理上不可能**，这也解释了压测里并发 16/32 时 TTFT 的陡增。

## 3. 能力基线（EvalScope + 统一打分）

| 节点 | 条数 | 关键指标 | 判读 |
| --- | --- | --- | --- |
| N1 意图 | 400 | Acc **78.00%** / Macro-F1 **79.13%** / 非法输出 1.00% | 长尾意图混淆，可微调 |
| N2 槽位 | 400 | JointEM **18.00%** / JSON 合法 **100%** / intent_acc 70.50% / Slot-F1 **35.90%** / 幻觉槽位 0.62% | 格式已合规，**value 抽取是主要瓶颈** |
| N3 话术 | 300 | 约束通过 **100.00%** / 平均 39.08 字 / 空输出 0 | 规则化约束都能满足 → 不需要微调 |
| N4 转人工 | 300 | F1 **100.00%** / 漏判 0.00% / JSON 合法 100%（TP=146 FP=0 FN=0 TN=154） | **饱和，无区分度**，见 §5 |
| gsm8k（通用回归） | 200 | accuracy **95.50%** | 作为"微调后是否变傻"的监控线 |
| ceval（通用回归） | — | **未跑成** | evalscope 从 modelscope 拉数据集卡死（14M 后无进展），已终止；后续用 `--data-source huggingface` 或本地数据补 |

## 4. 服务性能基线（并发 1~32，random 128~1024 prompt，max_tokens=256）

| 并发 | RPS | 完成吞吐 tok/s | TTFT avg / p50 / p99 (ms) | TPOT avg (ms) | 端到端 avg (s) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.19 | 41.88 | 101 / 104 / 160 | 23.52 | 5.19 |
| 4 | 0.73 | 151.35 | 147 / 136 / 410 | 24.76 | 5.27 |
| 8 | 1.28 | 268.11 | 224 / 172 / 682 | 26.87 | 5.82 |
| 16 | 2.09 | 464.03 | 445 / 270 / 1444 | 29.87 | 7.06 |
| 32 | 2.97 | 586.26 | 973 / 731 / 2601 | 38.13 | 8.19 |

- 全部档位成功率 100%。
- **TTFT P99 < 800ms 只在并发 ≤8 成立**，对应 **≈1.28 QPS**；并发 16 起 P99 破 1.4s，超过多数客服场景的体感阈值。
- 吞吐在并发 32 时仍在涨（586 tok/s），但延迟已不可用 —— 这是典型的"用延迟换吞吐"区间。
- 这张表是第七步量化收益的对照组：W4A16 后 KV cache 可放大 ~3.5 倍，同 SLA 下可承载更高并发。

产物：`outputs/20261111_182448/qwen3-14b/{performance_summary.txt, perf_report.html, parallel_*/benchmark_data.db}`

## 5. 本阶段发现的两个数据问题（必须写进汇报）

1. **原 N4 基准集严重重复**：`bench/n4_escalate/test.jsonl` 300 条里，**唯一 prompt 只有 20 条**
   （模板拼接时占位符为空导致大量同句）。这会让人误以为"300 条评测"其实只测了 20 种输入。
   → 已用 `bench/rebuild_gold.py` + 模型改写重建：N4 300 条**全部唯一**，N3 300 条全部唯一，
   金标不重写（只改用户表达），因此标签一致性天然成立；N4 还剔除了 105 条"模型自标与 gold 不一致"的候选。
2. **N4 基线饱和（F1 100%）**：重建后依然满分，说明当前 N4 样本的风险信号过于显式（用户直接说"要投诉/转人工"）。
   → 第四步的结论应当是：**N4 暂不微调**，先补"隐含风险/多轮升级"类难样本，否则微调收益无法量化。
   这与计划书"预期微调 N1/N2/N4"的假设不同，属于用数据推翻假设，写进 gap-analysis 更有说服力。

## 6. 卡时

| 阶段 | 实际耗时 | 说明 |
| --- | --- | --- |
| 环境安装 | ~55 min | 含两次装错（cu13 torch、依赖回溯）与一次磁盘占满，主要成本在踩坑 |
| 权重下载 | ~13 min | hf-mirror，28G |
| N3/N4 金标重建（GPU） | ~4 min | 2 轮改写共 5400 次请求，并发 16~24 |
| 四节点 + gsm8k 基线 | ~9 min | 1400 + 200 条 |
| 压测（5 档并发） | ~7 min | 250 次请求 |
| 合计（不含踩坑） | **~30 min** | 预算 4.5h，实际远低于预算，余量来自"本地写完再上卡" |

## 7. 下一步（退出 S1）

1. 本地写 `docs/gap-analysis.md`：按 §3 的失败样例（`reports/bench/baseline_*_failures.jsonl`）归类失败模式，
   定下"微调 N1/N2、不微调 N3、N4 先补数据"的结论。
2. S2 开机前补 `peft trl accelerate llmcompressor`（S1 没装，S2 训练和量化要用）。
3. S2 复用同一镜像 + 同一套 `serve.sh` 参数，保证与本节数字可比。

## 8. 关机前状态快照（2026-11-11）

| 项 | 位置 / 值 |
| --- | --- |
| 代码仓库（含未提交改动） | `/data/SFT`（`/root/SFT` 是容器根盘，关机即失） |
| venv | `/data/venv`（python 3.12.13 + vllm 0.11.2 + torch 2.9.0+cu128 + evalscope 1.14.1） |
| 模型权重 | `/data/models/qwen3-14b`（28G，bf16） |
| 金标扩写候选 | `/data/work/gold_candidates{,6}.jsonl` |
| 日志 | `/data/logs/{serve_base,baseline,perf_run,expand6,evalscope}.log` |
| 评测产物 | `reports/bench/baseline_summary.md`、`reports/preds/baseline/`、`reports/evalscope/baseline/`、`outputs/20261111_182448/` |
| SSH 私钥 | `/data/ssh/id_ed25519`（已加入 GitHub 账号，仅本实例使用） |

重启后复现基线：

```bash
source /data/venv/bin/activate
cd /data/SFT
MODEL_PATH=/data/models/qwen3-14b MODE=base bash deploy/serve.sh > /data/logs/serve_bf16.log 2>&1 &
python eval/run_evalscope.py --tag baseline --general --perf   # 需要 evalscope[perf]
python eval/evalscope_to_pred.py --work-dir reports/evalscope/baseline --tag baseline
python eval/score_bench.py --pred-dir reports/preds/baseline --tag baseline
```

S2 开机前必须先补：`pip install peft trl accelerate llmcompressor`（S1 未安装，训练/量化缺依赖）。
