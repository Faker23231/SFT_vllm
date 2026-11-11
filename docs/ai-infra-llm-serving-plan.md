# 微调项目实施大纲（最小卡时租卡 · 聚焦七步主线）

> 约束：**训练/部署/量化全部在租用的单卡 Linux 实例上完成**，本机只写代码、SSH 操作、看日志；**不引入 RAG 节点**；**技术栈只保留落地必需的几项**。
> 核心策略：**大模型 + 单一卡型 + 极短卡时**。模型选到 14B 提升简历说服力，靠压缩卡时（而非降低模型规模）控制成本。

---

## 0. 模型与卡型决策

### 0.1 为什么选 Qwen3-14B

| 候选 | 量化后权重 | 单卡可行性 | 简历说服力 |
|---|---|---|---|
| 4B / 8B | 2.6G / 5G | 24G 绰绰有余 | 弱：小模型微调已成标配，几乎不产生量化权衡问题 |
| **14B（选定）** | **~8G** | **40G 卡可覆盖训练/服务/量化全流程** | **强：主流 dense 尺寸，量化收益真实（28G→8G，可降到 24G 卡承接）** |
| 30B-A3B（MoE） | ~18G | 量化需 80G 卡（bf16 60G） | 更强，但必须换卡型 → 违反"少租卡" |

结论：**主模型 `Qwen3-14B`**（推理时 `enable_thinking=False`，与本仓库现有脚本一致）。
它同时满足三个条件：① 大到能讲清「量化 28G→8G、落地卡型 40G→24G」的降本故事；② 小到 **一张卡跑完全流程**；③ 40G 显存下不会因 OOM 反复返工（返工才是最贵的开销）。

> 若后续有余力想升级到 `Qwen3-30B-A3B`：它的 W4A16 只需 ~18G，单张 4090 即可服务，简历更亮；但自建量化必须上 80G 卡（bf16 60G），属于"租更大卡"而非"少租卡"。建议先完成 14B 全链路，再决定是否追加。

### 0.2 卡型：一种卡跑到底

| 项 | 选择 | 说明 |
|---|---|---|
| 卡型 | **A100 40G（或 L20 48G / 4090 48G 版）** | 一张卡覆盖：QLoRA 训练、bf16 服务、GPTQ 校准、量化后服务 |
| 数量 | **单卡** | 不使用多卡并行，避免 NCCL/启动成本 |
| 实例次数 | **2 次** | S1 部署+基线；S2 合成+训练+量化+终评。中间不换卡、不重建环境 |
| 计费方式 | 按需（不用抢占式） | 抢占实例被回收会导致长任务重跑，反而更贵 |

### 0.3 卡时与成本（关键：靠压缩卡时省钱，不靠降模型）

| 阶段 | 卡时 | 说明 |
|---|---|---|
| S1：环境 + vLLM 部署 + 冒烟 | 2h | 装环境、拉模型、起服务、读日志 |
| S1：EvalScope 基线（4 节点 + 通用能力） | 1.5h | 服务常开，API 方式评测 |
| S1：perf 压测（并发 1~32） | 1h | 与基线同一服务实例，不重起 |
| S2：数据合成（离线批量推理） | 2.5h | 5 万条候选 |
| S2：3 组 QLoRA 消融 + 合并 | 5h | 14B QLoRA 单组约 1.2h |
| S2：GPTQ 量化（含 2 次重试） | 3h | bf16 28G + 校准激活 |
| S2：复评（微调后 + 量化后）+ 终评压测 | 2.5h | 复用服务 |
| 缓冲（调试/重跑） | 6h | 必留，租卡的意外都在这 |
| **合计** | **~24h** | **A100 40G 按 3.5 元/h ≈ 85 元；L20 按 2 元/h ≈ 50 元** |

对照：8B + 24G 卡的方案约 60h × 2 元 ≈ 120 元。**模型翻倍、成本反而更低**，原因就是卡时被压到 24h（见 §8 的压缩手段）。

### 0.4 明确的非目标（写进 README，防项目发散）

不做 RAG、不做 Agent 编排、不做 K8s/多机多卡、不做投机解码/PD 分离、不做 DPO。
只做「单卡可复现的：服务 → 评测 → 微调 → 量化 → 复评」闭环。

---

## 1. 第一步：vLLM 部署开源模型

### 1.1 实例初始化（一次做完，立刻存镜像）

```bash
# 1) 全部路径放数据盘，系统盘很容易被 28G 权重写满
df -h && mkdir -p /root/autodl-tmp/{models,work,logs}

# 2) 环境：uv 比 pip 快一个数量级
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U uv
uv venv /root/autodl-tmp/venv --python 3.11
source /root/autodl-tmp/venv/bin/activate
uv pip install vllm evalscope llm-compressor

# 3) 模型（14B bf16 约 28G，用镜像站 + 多线程）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-14B --local-dir /root/autodl-tmp/models/qwen3-14b
```

**省钱纪律（比技术选型更影响成本）**
1. 装完**立刻保存镜像**，S2 直接基于镜像开实例，省掉 40 分钟重装。
2. 长任务一律 `tmux`，断网不中断。
3. **本地写代码 → 实例只跑**：任何脚本先在本地 `--dry-run`/小样本校验格式，再上传跑全量。
4. 每个阶段产物落盘后立刻 `rsync -az` 回本地，实例异常也不用重跑。
5. 记录 `nvidia-smi` / `nvcc -V` / torch 版本到 `docs/env-matrix.md`，S2 开实例时对照。

### 1.2 起服务

```bash
vllm serve /root/autodl-tmp/models/qwen3-14b \
  --served-model-name qwen3-14b \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --port 8000
```

**必须读懂启动日志的四个数**（面试高频）：
- `GPU KV cache size: xxx tokens` —— 决定理论最大并发；
- `Maximum concurrency for 8192 tokens per request: xx` —— 与 `max-num-seqs` 的真实关系；
- 权重显存 / 非 torch 显存占用；
- 实际 attention backend（A100 一般是 FlashAttention，记录实际值）。

### 1.3 冒烟与最小压测

```bash
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model":"qwen3-14b",
  "messages":[{"role":"system","content":"只输出JSON"},{"role":"user","content":"用户说：帮我取消明天的闹钟"}],
  "temperature":0,"max_tokens":128}'

curl -s localhost:8000/metrics | grep -E "time_to_first_token|inter_token_latency|num_requests_running"
```

**验收（P1 门槛）**：`env-matrix.md`、`serve.sh`、冒烟脚本齐全；延迟三次可复现；`/metrics` 能取到 TTFT；镜像已保存。

---

## 2. 第二步：为每个调用大模型的节点构造基准测试集

先明确**项目里谁在调大模型**。本项目共 4 个节点（均不含检索）：

| 节点 | 任务 | 输出形态 | 规模 |
|---|---|---|---|
| **N1 意图识别** | 用户话语 → 封闭意图枚举（客服场景 30~60 类） | 单个 label | 400 |
| **N2 槽位/信息抽取** | 话语 → 结构化字段 | 严格 JSON | 400 |
| **N3 回复生成** | 上下文 + 结论 → 客服话术 | 自然语言 + 约束 | 300 |
| **N4 转人工判定** | 对话片段 → 是否升级人工 | 二分类 + 理由 | 300 |

### 2.1 构造规范（决定评测可信度）

1. **来源分层**：公开集（`data/test.jsonl` 的 MASSIVE 子集）+ 业务改写样本 + 边界构造样本（错别字/口语/省略/否定/多意图）。
2. **分层配比**：easy / medium / hard 约 50/30/20，意图按类别**均衡配额**，长尾类每类 ≥ 5 条。
3. **防泄漏**：与训练数据做 8-gram 重叠检查 + MinHash 去重，报告中写明剔除条数。
4. **标签质量**：全部人工复核（300~400 条/节点），报标注一致率；不用模型自标当金标。
5. **每条样本带** `id` / `difficulty` / `category`，便于出分层报告。
6. **基准集必须进 git**：实例随时释放，数据不能只存在卡上。

### 2.2 指标定义（写入 `eval/metrics.py`，所有脚本共用）

| 节点 | 指标 |
|---|---|
| N1 | Accuracy、Macro-F1、非法输出率（不在枚举内） |
| N2 | Joint Exact Match（复用 `eval/baseline_eval.py` 的判定）、JSON 合法率、Slot F1、**幻觉槽位率**（value 未出现在原文） |
| N3 | 约束满足率（长度/必备信息/禁语）、人工或评委模型 1~5 分（抽 100 条） |
| N4 | Precision/Recall/F1、**漏判率**（该转人工却没转） |

**验收（P2 门槛）**：`bench/` 下 4 节点 × {dev,test} 齐全；`docs/bench-spec.md` 含规范、去重报告、分层统计；数据已提交。

> 本阶段**零卡时**，全部在本地完成。这是压缩总成本的最大一块（约 1400 条标注 + 代码）。

---

## 3. 第三步：用 EvalScope 做基准测试

### 3.1 能力评测（服务常开，一次跑完 4 个节点）

```bash
evalscope eval \
  --model qwen3-14b \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets general_qa \
  --dataset-args '{"general_qa":{"local_path":"/root/autodl-tmp/work/bench/n2_slot/test.jsonl"}}' \
  --generation-config '{"temperature":0,"max_tokens":256}' \
  --work-dir /root/autodl-tmp/reports/baseline/n2
```

要点：
- 本地 jsonl 用 `query` / `response` 两列，`response` 放序列化的标准答案。
- **自定义指标**：`general_qa` 默认按相似度打分，反映不了 JSON/EM。实现 `DataAdapter` 子类，把 `eval/metrics.py` 的 Joint-EM、JSON 合法率、幻觉槽位率注册进去，确保训练侧与 EvalScope 侧用同一份判定代码。
- 一个 `bash` 脚本串行跑 4 节点，避免手敲 4 次命令、也避免服务重启。
- **通用能力回归**：加跑 `gsm8k`（limit 200）与 `ceval`（limit 200），作为"微调后是否变傻"的监控线。
- 参数名随版本变动，先 `evalscope eval --help` 对齐。

### 3.2 服务指标压测（同一实例，紧接着跑）

```bash
evalscope perf \
  --model qwen3-14b \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --api openai \
  --dataset openqa \
  --parallel 1 4 8 16 32 \
  --number 50 --max-tokens 256 \
  --min-prompt-length 128 --max-prompt-length 1024
```

得到并发 × 吞吐 × 延迟（TTFT/TPOT），找出「TTFT P95 < 800ms」对应的最大 QPS。这张表是量化收益的对照组。

**验收（P3 门槛）**：`reports/baseline/` 下 4 节点分数表 + 通用能力 + 并发曲线 → `docs/baseline-report.md`；随后**关闭实例**（S1 结束，卡时 4.5h）。

---

## 4. 第四步：评估当前模型，判断哪些场景需要微调

不要「全都微调」，用证据 + 成本决定：

| 节点 | 基线表现 | 主要失败模式 | 决策 | 依据 |
|---|---|---|---|---|
| N1 意图 | Macro-F1 尚可但长尾差 | 相似意图混淆 | **微调** | 标签封闭、类别多、数据可造 |
| N2 槽位 | JSON 合法率不足、幻觉槽位 | 格式漂移 + 编造 value | **微调 + 约束解码对照** | 格式问题可先由约束解码兜住 |
| N3 生成 | 约束违反 | 话术跑偏、缺必备信息 | **暂不微调** | 主观指标，收益难量化；先改 prompt |
| N4 转人工 | 漏判率高 | 风险信号弱 | **微调** | 漏判代价高，二分类易验证 |

**必须做的对照实验（N2）**：① 纯 prompt；② 只加约束解码（`--guided-decoding-backend xgrammar` + `response_format.json_schema`）；③ 只微调；④ 微调 + 约束解码。
→ 结论通常是「约束解码解决格式、微调解决语义」，据此定上线组合。这是全项目最能体现判断力的一步。

> 14B 的基线通常已明显好于 8B，失败样本更集中在长尾与格式，因此**更容易得出"只微调 2 个节点"的克制结论**——这比全节点微调更有说服力。

**验收（P4 门槛）**：`docs/gap-analysis.md`，含失败样本归类、每节点方案与预期收益。本阶段同样**零卡时**（基于已有报告分析）。

---

## 5. 第五步：构造微调数据（种子扩写）

只针对 P4 判定要微调的节点（N1、N2、N4），N3 不动。

### 5.1 流程（`data_pipeline/`，4 个脚本）

1. **`01_seed.py` — 种子**：N1/N2 从 `data/train.jsonl` 按意图每类抽 8 条真实样本；N3/N4 用 `data_pipeline/seeds/` 的人工种子（场景与评测集刻意错开）。种子来自真实分布，不用模型生成。
2. **`02_expand.py` — 扩写**（两轮推理：改写 → 重新标注）：用**实例上的 vLLM 离线批量推理**（`vllm.LLM`，吞吐远高于逐条 HTTP），四种改写风格：
   - 口语化、加入错别字、尽量简短、加入冗余铺垫。
   `temperature=0.8`，目标 1~3 万条候选。`--backend mock` 可在本地零卡时跑通全链路。
   **省钱关键**：先用 `--limit 200` 校准单条耗时与 tokens/s，再估算全量时长（14B 约 1500~2500 tok/s，与预算表口径一致）。
3. **`03_filter.py` — 拒绝采样与质检**（最容易被追问的一步）：
   - 结构校验：能否解析出符合 schema 的答案；
   - **标签一致性**：模型对改写文本的自标结果必须与种子标签一致；
   - **幻觉过滤**：槽位 value 必须能在原文定位；
   - **去重**：8-gram minhash 指纹（同节点内）+ 与 test 集命中比例 ≥0.8 直接丢；
   - **配额**：按 category 截断，防高频类淹没长尾。
   每级保留率写进 `out/filtered/report.md`（本地 mock 实测：2104 → 1846）。
4. **`04_mix_split.py` — 配比 + 切分**：真实:合成 = 1:1 / 1:3 两档，9:1 切 train/validation，直接输出 `{"messages": [...]}` 供训练脚本加载；manifest 里记录 `actual_ratio`（合成不足时如实反映）与 train 集 sha256。

### 5.2 数据卡（`docs/data-card.md`）

来源、生成方式（模型/参数/条数）、各级过滤保留率、去重统计、类别分布、许可与隐私、已知偏差。

**验收（P5 门槛）**：data card 完整；train/test 8-gram 零重叠；每类 ≥ 20 条；数据已回传本地。

---

## 6. 第六步：微调并复跑评估集（S2 开始）

### 6.1 训练脚本

复制 `finetune/03_qlora_finetune.py` 为 `finetune/04_domain_qlora.py`，改动：

| 参数 | 原值 | 新值 | 理由 |
|---|---|---|---|
| `MODEL_PATH` | Qwen3-0.6B | Qwen3-14B | 主模型 |
| `per_device_train_batch_size` | 2 | 4 | 40G 显存富余 |
| `gradient_accumulation_steps` | 8 | 4 | 等效 batch=16 |
| `max_length` | 1024 | 4096 | 槽位 JSON + 上下文 |
| `packing` | 关 | **开** | 提升 token 利用率 30~50%，直接省训练时间 |
| 其余 | `assistant_only_loss=True`、`gradient_checkpointing=True`、`optim="paged_adamw_8bit"`、`report_to="tensorboard"` 保持不变 |

14B QLoRA 在 40G 上的显存估算：4bit 权重 ~9G + LoRA/优化器 ~1G + 激活（grad ckpt）~10G ≈ 20G，余量充足；不必为显存牺牲 batch。

### 6.2 消融（3 组，串行跑，别并行开实例）

| 变量 | 取值 | 目的 |
|---|---|---|
| r | 16 / 32 | 容量是否够 |
| 数据配比 | 真实:合成 1:1 / 1:3 | 合成数据的边际收益 |
| target_modules | 仅 attention / all-linear | 领域对齐深度 |

每组训完 merge（`eval/merge_lora.py`），**用量化前的 bf16 合并权重**评测。

### 6.3 复评（口径必须与 P3 完全一致）

```
同一份 bench test → 同一版 metrics.py → 同一 temperature=0 → 同一 max_tokens
```

| 节点 | 基线 | 微调后 | Δ | gsm8k Δ |
|---|---|---|---|---|
| N1 Macro-F1 | __ | __ | +__pp | __ |
| N2 Joint-EM | __ | __ | +__pp | __ |
| N2 JSON 合法率 | __ | __ | +__pp | __ |
| N4 漏判率 | __ | __ | -__pp | __ |

**判定**：目标节点提升 ≥ 5pp 且通用能力回退 ≤ 2pp 视为成功；回退过大 → 降 lr 或混入 10% 通用数据。

**验收（P6 门槛）**：`docs/train-report.md`（超参表、loss 曲线、消融表、复评表）；合并权重已备份。

---

## 7. 第七步：GPTQ 量化 + 再次 vLLM 部署

### 7.1 量化

- 输入：P6 merge 后的 **bf16 全精度权重**（量化 adapter 或 4bit 基座是错的）。
- 校准集：**本项目 bench 的 dev 集**（实际仅 108 条，不够 512）+ 从 `data/train.jsonl` 补领域样本凑满 512 条，`max_seq_len=2048`；来源比例写进 `quant_config.json`。
  - 注意：不要用 bench test 补，否则等于让量化"见过考题"，精度数字失真。
- 工具：llm-compressor，`GPTQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"])`，`group_size=128`。

```bash
python quant/gptq_quant.py \
  --model /root/autodl-tmp/models/qwen3-14b-merged \
  --calib /root/autodl-tmp/work/bench/n2_slot/dev.jsonl \
  --num-calibration-samples 512 \
  --output /root/autodl-tmp/models/qwen3-14b-gptq-w4a16
```

14B bf16 权重 28G + 校准激活约 6G ≈ 34G，40G 卡单卡可完成，不会 OOM。

### 7.2 部署量化模型

```bash
vllm serve /root/autodl-tmp/models/qwen3-14b-gptq-w4a16 \
  --served-model-name qwen3-14b-gptq \
  --quantization gptq_marlin \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64
```

确认启动日志走 `gptq_marlin`（否则 4bit 反量化开销会吃掉收益）。

### 7.3 量化验收（三张表，缺一不可）

**① 精度回归**（同一 bench，对比 bf16 微调后）

| 节点 | bf16 微调后 | GPTQ W4A16 | Δ |
|---|---|---|---|
| N2 Joint-EM | __ | __ | -__pp |
| N2 JSON 合法率 | __ | __ | -__pp |
| N1 Macro-F1 | __ | __ | -__pp |
| gsm8k | __ | __ | -__pp |

**② 服务收益**（同并发档位）

| 指标 | bf16 | W4A16 |
|---|---|---|
| 权重显存 | ~28G | ~8G |
| 可落地最低卡型 | 40G | **24G（单卡 4090）** |
| TTFT P50/P95 @c=8 | __ | __ |
| tokens/s @c=8 / c=16 | __ | __ |
| 最大可用 `max-num-seqs` | 32 | 64 |

**③ 结论**：精度损失 ≤ 1.5pp 前提下，吞吐提升 __ 倍、显存降低 __%、**可用卡型从 40G 下探到 24G**，给出明确上线结论；掉点超标则做误差归因（校准集覆盖 / group_size / 敏感层）并复测一次。

**验收（P7 门槛）**：`docs/quant-report.md` + 量化模型稳定 serve + 三张表齐全 + 权重回传本地。

---

## 8. 排期与"少租卡"执行策略（4 周，共 2 次开实例）

| 周 | 任务 | 开实例 | 卡时 |
|---|---|---|---|
| W1 | 本地：脚本改造 + bench 雏形 + dry-run | 否 | 0 |
| W2 | 本地：1400 条 bench 完成 + metrics.py + 数据 pipeline 代码 | 否 | 0 |
| W3 | **S1**：环境 + 部署 + 基线 + 压测 → 关机 | 是 | 4.5h |
| W4 | 本地：差距分析 + 数据 pipeline 跑通校验（小样本） | 否 | 0 |
| W5 | **S2**：合成 → 训练 3 组 → 合并 → 量化 → 复评压测 → 关机 | 是 | ~24h |

**6 条压缩卡时的具体手段（每条都可写进简历）**

1. **本地写、实例跑**：所有脚本先在本地 dry-run，实例上只跑全量，杜绝"边调边烧卡"。
2. **一次实例跑完全链路**：S2 内串行执行，中间不释放实例（释放后重装环境 + 重传 28G 权重 ≈ 1h 白烧）。
3. **镜像固化**：S1 结束存镜像，S2 直接开机即用。
4. **离线批量推理做合成**：`vllm.LLM` 离线接口的吞吐是逐条 HTTP 的数倍，先小样本校准再全量。
5. **训练开 `packing=True`**：短样本拼批，token 利用率提升 30~50%，单组训练从 ~1.8h 降到 ~1.2h。
6. **服务只起两次**：bf16 一次（基线 + 微调后复评），量化后一次（量化复评 + 终评压测），复用同一进程。

---

## 9. 仓库结构（✅ 表示租卡前已完成并在本地跑通）

```
SFT/
├── data/                       # 已有 MASSIVE 切分
│   └── domain_sft_*_train.jsonl      ✅ 配比后的训练集（messages 格式）
├── data_pipeline/              ✅
│   ├── 01_seed.py                     ✅ 种子池（N1/N2 真实抽样 + N3/N4 人工种子）
│   ├── 02_expand.py                   ✅ 扩写（mock 已跑通；vllm/openai 后端待上卡）
│   ├── 03_filter.py                   ✅ 质检 + 去重 + 配额
│   ├── 04_mix_split.py                ✅ 配比 + 切分 + 转训练格式
│   └── seeds/                         ✅ N3/N4 训练种子（与评测集场景错开）
├── bench/                      ✅
│   ├── build_bench.py                 ✅ 分层抽样 + 难度分层 + 泄漏检查
│   ├── prompts.py                     ✅ 四节点 prompt（训练/评测/serving 共用）
│   ├── seeds/                         ✅ N3/N4 人工种子 16 条
│   ├── stats.md / stats.json          ✅ 分布与泄漏报告
│   └── <node>/{test,dev}.jsonl        ✅ 1400 条基准集
├── eval/
│   ├── baseline_eval.py        # 已有 MASSIVE 基线（保留）
│   ├── metrics.py              ✅ 四节点指标唯一实现
│   ├── selftest.py             ✅ 指标自检（金标当预测，必须 100%）
│   ├── score_bench.py          ✅ 打分 + 失败样例导出
│   ├── run_evalscope.py        ✅ EvalScope 驱动（支持 --dry-run 本地预检）
│   └── evalscope_to_pred.py    ✅ evalscope 产物 → 统一打分输入
├── finetune/01~03_*.py         # 已有
├── finetune/04_domain_qlora.py ✅ 14B QLoRA（待上卡执行）
├── quant/gptq_quant.py         ✅ GPTQ W4A16（待上卡执行）
├── deploy/{setup.sh,serve.sh,sync.sh}   ✅ 环境初始化 / 双模式启动 / 产物回传
├── reports/{bench,preds,evalscope}/
└── docs/{env-matrix,bench-spec,gap-analysis,data-card,train-report,quant-report}.md
```

`main.py` 现有编排（分阶段子进程 + 正则抓分 + 汇总提升）扩展：加「节点基准评测」「量化」「量化复评」三个阶段即可一键复现。

---

## 9.1 租卡前已完成清单（W1 + W2，全部本地零卡时）

| 产物 | 验证方式 | 结果 |
|---|---|---|
| 四节点基准集 1400 条（+ dev 108 条） | `python bench/build_bench.py --synth 300` | test 400/400/300/300，dev 50/50/4/4 |
| 意图集一致性 | 构建时校验 59 类 | 一致（MASSIVE 子集不含 `cooking_query`） |
| 泄漏检查 | 与 `data/train.jsonl` 做 8-gram + 精确重复 | 剔除精确重复 223 条；n1 剩 1 条高重叠（模板化邮箱句式，已记录） |
| 指标实现 | `python eval/selftest.py` | 四节点全部 100.00% |
| 打分链路 | `python eval/score_bench.py --from-gold` | 汇总报告与失败样例正常产出 |
| EvalScope 命令 | `python eval/run_evalscope.py --dry-run` | 7 条命令（4 节点 + gsm8k + ceval + perf）全部构造正确 |
| 数据流水线 | `01 → 02(mock) → 03 → 04` 全跑 | 2104 候选 → 质量/去重后 1846 → 训练集 3778 条（1:1） |
| 训练/量化脚本 | 语法与参数自检 | 待上卡执行 |

**尚未完成、需要上卡的**（按卡时顺序）

1. `02_expand.py --backend vllm`：把 1846 条合成扩到 1~3 万条（mock 数据仅供跑通流程，不可用于训练）。
2. N3/N4 的正式金标：当前 300 条里 288 条是模板合成，需扩写后人工筛 300 条替换，然后**重跑全部基线**。
3. `04_domain_qlora.py`：3 组消融训练。
4. `gptq_quant.py`：GPTQ 量化 + 服务复评。
5. `docs/` 里三份待填报告：`gap-analysis.md`、`train-report.md`、`quant-report.md`。

---

## 10. 风险与预案

| 风险 | 影响 | 预案 |
|---|---|---|
| 实例释放后环境/产物丢失 | 全部重来 | 装完存镜像；每阶段 `rsync` 回本地；权重上传对象存储 |
| 系统盘被 28G 权重写满 | 服务/训练中断 | 全路径走数据盘；`serve.sh` 前置 `df -h` 检查 |
| 忘记关机持续计费 | 成本失控 | 长任务 `tmux` + 明确预期时长；跑完立即关机；每周核账单 |
| 14B 训练 OOM | 中断 | batch 4→2、`max_length` 4096→2048、关 packing；40G 卡余量本应充足 |
| GPTQ 校准显存不足 | 量化失败 | 校准 512→256 条、`max_seq_len` 2048→1024、group_size 128 |
| 合成数据与 test 相似 | 指标虚高 | 8-gram + MinHash 检查并报数；评测口径固定 |
| 两套评测口径不一致 | 数字对不上 | 训练侧与 EvalScope 侧共用 `eval/metrics.py` |
| 量化掉点超阈值 | 收益为负 | 换领域校准集、group_size 64、`lm_head`/首尾层保留 bf16 |
| 掉进"再训一组试试" | 卡时翻倍 | 消融上限 3 组，事先定好判定标准，达到即停 |

---

## 11. 简历写法（跑完填实测数）

> **LLM 推理服务与领域微调闭环（单卡 40G 完成，总卡时 24h）**
> - 基于 **vLLM** 部署 Qwen3-14B-Instruct，依据启动日志的 KV cache 容量推算最大并发，完成显存规划与 attention backend 选型，暴露 OpenAI 兼容接口支撑 4 个 LLM 调用节点。
> - 为意图识别、槽位抽取、话术生成、转人工判定 4 个节点设计 1400 条分层基准集（难度分层 + 长尾配额，train/test 8-gram 零重叠），定义 Joint-EM、JSON 合法率、幻觉槽位率、漏判率等指标。
> - 用 **EvalScope** 建立能力与性能双基线（并发 1~32 的吞吐/延迟曲线），定位 N1 长尾混淆、N2 格式漂移与幻觉、N4 漏判，通过「prompt / 约束解码 / 微调 / 微调+约束解码」四组对照确定上线方案。
> - 设计种子扩写数据流水线：vLLM 离线批量推理合成 __ 万条候选（__ tok/s），经 schema 校验、标签一致性复核、幻觉过滤与 MinHash 去重后保留 __ 条（保留率 __%）；配比消融确定真实:合成最优比例。
> - QLoRA 微调 14B 模型后目标节点 Joint-EM __% → __%（+__pp）、JSON 合法率 __% → __%，gsm8k/ceval 回退 <2pp。
> - 用 **llm-compressor** 完成 **GPTQ W4A16** 量化（领域校准集 512 条、group_size 128），权重显存 28G → 8G，**可落地卡型由 40G 下探到 24G 消费级卡**，同 SLA 下吞吐 ×__，精度损失 __pp，经 `gptq_marlin` 部署。
> - 通过本地 dry-run、镜像固化、离线批量合成与训练 packing，将全流程卡时压缩到 24h（约 __ 元），实现单卡可复现的完整实验闭环。

**面试必须能答**：
1. PagedAttention 解决什么问题？block size 大/小的权衡？
2. KV cache 容量怎么算？为什么 `max-num-seqs` 不能随便调大？
3. 14B 微调用 QLoRA 而非全参，显存账怎么算的？
4. 为什么要微调而不是只做 prompt/约束解码？四组对照说明了什么？
5. GPTQ 校准集为什么必须用领域数据？group_size 影响什么？
6. 量化后精度掉了怎么归因、怎么补救？
