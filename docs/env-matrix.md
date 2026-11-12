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
| 卡型 | 同 S1（A100 40G，同一镜像，未换卡型） |
| 开机时间 | 2026-11-12 09:10 |
| 卡时 | 进行中（目标 ≤24h） |

**S2 依赖安装（有版本三角，照抄即可）**：

```bash
pip install peft trl accelerate llmcompressor "torch==2.9.0" -i https://mirrors.cloud.tencent.com/pypi/simple
pip install "compressed-tensors==0.12.2"          # vllm 0.11.2 硬钉，必须先降回来
pip install "llmcompressor==0.8.1" --no-deps      # 0.10 要 compressed-tensors 0.14，与 vllm 冲突
pip install "trl==0.23.0"                          # trl 1.x 要 accelerate>=1.11/datasets>=4.7，与 llmcompressor 0.8.1 互斥
pip install "accelerate==1.10.1" "datasets==4.1.1" "pillow==11.3.0" "nvidia-ml-py==13.580.82"
pip install bitsandbytes tensorboard               # QLoRA 与 loss 曲线必需（S1 镜像里没有）
```

> 遗留：`pip check` 仍会报 llmcompressor 0.8.1 对 torch≤2.8 / transformers≤4.56 的**声明上限**。
> torch 不能降（vllm 0.11.2 钉 2.9.0），实测 `from llmcompressor import oneshot` 与 GPTQModifier 正常导入，
> 量化阶段以实际跑通为准；若量化报错优先怀疑这里。

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
| python | 3.12.13 | 3.12.13 | 同一 venv `/data/venv` |
| torch | 2.9.0+cu128 | 2.9.0+cu128 | cu13 版在 560 驱动上不可用 |
| vllm | 0.11.2 | 0.11.2 | 0.33.0 需 CUDA 13 驱动，勿用 |
| transformers | 4.57.6 | 4.57.6 | |
| peft / trl | 未装 | 0.21.2 / 0.23.0 | trl 1.x 与 llmcompressor 0.8.1 的 accelerate/datasets 要求互斥 |
| llm-compressor | 未装 | 0.8.1 | 配 compressed-tensors 0.12.2；oneshot 改用 `from llmcompressor import oneshot` |
| compressed-tensors | 0.12.2 | 0.12.2 | vllm 0.11.2 硬钉 ==0.12.2 |
| accelerate / datasets | — | 1.10.1 / 4.1.1 | 被 llmcompressor 0.8.1 上限压住 |
| bitsandbytes | 未装 | 0.50.2 | QLoRA 4bit 必需 |
| evalscope | 1.14.1 | 1.14.1 | 压测需额外 `evalscope[perf]`（缺 sse_starlette 会 ImportError） |
| CUDA / 驱动 | 12.8 / 560.35.03 | 12.8 / 560.35.03 | nvcc V12.8.93 |

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
| 训练 OOM（CE logits 一次要 7.6G） | packing + batch 4 × 4096 token，151k 词表 logits 转 fp32 | batch 4→2、grad_accum 4（等效 batch 不变）；实测显存 31.4G/39.5G |
| `assistant_only_loss=True` 报 "no assistant tokens" | Qwen3 原生 chat template 缺 `{% generation %}` | 用 `deploy/chat_template_qwen3_nothink.jinja`（已加 generation 标记），训练脚本自动加载 |
| `pip install llmcompressor` 后 vllm 起不来 | llmcompressor 0.10 拉 compressed-tensors 0.14 | 见 §2 S2 依赖四连；`--no-deps` 装 0.8.1 |
| 合成数据质量差 | 改写温度过低 / 一致性过滤缺失 | 温度 0.8，务必跑 `03_filter.py` |
