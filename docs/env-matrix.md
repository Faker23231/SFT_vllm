# 环境矩阵（env-matrix）

> 换实例后照着这张表复原环境，避免"上次能跑这次不能"的黑盒问题。
> 每条记录都必须来自实际输出，不要凭记忆填。

## 1. 本机（仅用于写代码、打分、看日志，不跑 vLLM）

| 项 | 值 | 采集方式 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU，16303 MiB | `nvidia-smi --query-gpu=name,memory.total --format=csv` |
| 驱动 | 621.14 | 同上 |
| 计算能力 | sm_120 (12, 0) | `torch.cuda.get_device_capability()` |
| Python | 3.x（Anaconda） | `python -V` |
| PyTorch | 2.13.0+cu130 | `torch.__version__` |
| 用途 | 数据构造、指标打分、命令 dry-run | — |

本机不部署 vLLM（不支持原生 Windows），也不做训练（16G 显存不足以跑 14B QLoRA 的舒适配置）。

## 2. 实例（S1 / S2 各填一次）

### S1（部署 + 基线）

| 项 | 值 |
|---|---|
| 平台 / 卡型 | NVIDIA A100-PCIE-40GB（40960 MiB，单卡） |
| 镜像 | 自带 python 3.12.13 + 驱动 560.35.03（CUDA 12.6 驱动，nvcc 12.8） |
| 实例创建时间 | 2026-11-11 16:51 |
| 关机时间 | 待填 |
| 计费卡时 | 待填（目标 ≤4.5h） |

**重要坑**：`pip install vllm` 默认装 0.33.0（CUDA 13 版 torch），在 560 驱动上
`torch.cuda.is_available()` = False（`found version 12060`）。必须钉 **vllm==0.11.2**（torch 2.9.0+cu128）。
另外 pip 用 `mirrors.cloud.tencent.com` 约 16~90MB/s，aliyun/pypi 只有 0.4MB/s；
`rm -rf` 在本环境会被转进 `/data/.Trash-0`（占 22G 不释放），清理要用 `find ... -delete`。

### S2（合成 + 训练 + 量化 + 总评）

| 项 | 值 |
|---|---|
| 卡型 | 同 S1（用镜像复原，不换卡型） |
| 卡时 | 待填（目标 ≤24h） |

## 3. 版本矩阵（在实例上执行 `python -c ...` 采集）

```bash
python - <<'PY'
import platform, torch, transformers, peft, trl, vllm
print("python", platform.python_version())
print("torch", torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability())
print("transformers", transformers.__version__)
print("peft", peft.__version__, "trl", trl.__version__)
print("vllm", vllm.__version__)
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvcc --version | tail -2
```

| 组件 | S1 实测 | S2 实测 | 备注 |
|---|---|---|---|
| python | 3.12.13 | | 3.11 |
| torch | 2.9.0+cu128 | | cu128；cu13 版在 560 驱动上不可用 |
| vllm | 0.11.2 | | 0.33.0 需 CUDA 13 驱动，勿用 |
| transformers | 4.57.6 | | |
| peft / trl | 未装 | | trl ≥ 0.20 才有 `assistant_only_loss`；S2 前补装 |
| llm-compressor | 未装 | | GPTQ 量化；S2 前补装 |
| evalscope | 1.14.1 | | 压测需额外 `evalscope[perf]`（缺 sse_starlette 会 ImportError） |
| CUDA / 驱动 | 12.8 / 560.35.03 | | nvcc V12.8.93 |

## 4. vLLM 关键日志（每次起服务都记一次）

| 项 | bf16 | GPTQ W4A16 | 说明 |
|---|---|---|---|
| 模型路径 | `/data/models/qwen3-14b` | | 微调后待填 |
| `GPU KV cache size` | 50,608 tokens | | 36G−28G 权重后的 KV 预算 |
| `Maximum concurrency` | 6.18x @8192 | | max-num-seqs=32 实际被 KV 卡在 6 |
| attention backend | FLASH_ATTN | | sm80 走 FlashAttention |
| 非 torch 显存 | 见启动日志 Memory profiling | | CUDA graph / activation |
| 启动耗时 | ~70s（含 8 shard 加载） | | |
| `gptq_marlin` 是否命中 | — | | 量化没走 marlin 等于白量化 |
| `gptq_marlin` 是否命中 | — | | 量化没走 marlin 等于白量化 |

采集命令：

```bash
vllm serve ... 2>&1 | tee logs/serve_$(date +%H%M).log
grep -E "KV cache size|concurrency|attention backend|non-torch" logs/serve_*.log
```

## 5. 已知坑位（踩过就记下来）

| 现象 | 原因 | 处理 |
|---|---|---|
| 服务启动 OOM | `gpu-memory-utilization` 过高或 `max-model-len` 过大 | 降到 0.85 / 4096 再看 KV cache 容量 |
| 量化后吞吐没有提升 | 未命中 `gptq_marlin` kernel | 服务加 `--quantization gptq_marlin` |
| 评测分数异常高 | 训练集与评测集重叠 | 跑 `bench/stats.md` 的泄漏检查 |
| 训练 OOM | batch 太大或 packing 后单条超长 | batch 4→2，`max_length` 4096→2048 |
| 合成数据质量差 | 改写温度过低 / 一致性过滤缺失 | 温度 0.8，务必跑 `03_filter.py` |
