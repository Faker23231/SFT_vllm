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
| 平台 / 卡型 | 待填（A100 40G / L20 48G / 4090 48G） |
| 镜像 | 待填（PyTorch 2.x + CUDA 12.4/12.8） |
| 实例创建时间 | 待填 |
| 关机时间 | 待填 |
| 计费卡时 | 待填（目标 ≤4.5h） |

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
| python | | | 3.11 |
| torch | | | cu128 |
| vllm | | | |
| transformers | | | |
| peft / trl | | | trl ≥ 0.20 才有 `assistant_only_loss` |
| llm-compressor | | | GPTQ 量化 |
| evalscope | | | `--eval-type` 取值需核对版本 |
| CUDA / 驱动 | | | |

## 4. vLLM 关键日志（每次起服务都记一次）

| 项 | bf16 | GPTQ W4A16 | 说明 |
|---|---|---|---|
| 模型路径 | | | |
| `GPU KV cache size` | | | 决定理论并发 |
| `Maximum concurrency` | | | 与 `max-num-seqs` 的关系 |
| attention backend | | | A100 一般为 FlashAttention，4090 为 FlashInfer/XFormers |
| 非 torch 显存 | | | CUDA graph / activation |
| 启动耗时 | | | |
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
