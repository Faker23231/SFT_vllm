# 基准集规范（bench-spec）

> 本文件是评测的"宪法"：指标怎么算、样本怎么来、什么情况算泄漏、口径怎么固定。
> 任何一条改动都要记到文末的变更记录里，否则历史分数不可比。

## 1. 节点定义

| 节点 | 线上位置 | 输入 | 输出 | test 规模 |
|---|---|---|---|---|
| `n1_intent` | 用户消息进入后的第一步 | 用户话语 | 59 类意图中的 1 个 | 400 |
| `n2_slot` | 意图识别之后 | 用户话语 | `{"intent":..., "slots":[{"name","value"}]}` | 400 |
| `n3_reply` | 处置结论确定之后 | 用户诉求 + 处置动作 + 订单号 | 一段客服话术（≤80 字） | 300 |
| `n4_escalate` | 每轮对话结束时 | 对话片段 | `{"escalate": bool, "reason": str}` | 300 |

合计 1400 条（+ 每节点 50 条 dev，N3/N4 为 4 条）。

## 2. 数据来源

| 节点 | 来源 | 处理 |
|---|---|---|
| n1 / n2 | `data/test.jsonl`（MASSIVE 中文子集 2974 条） | 剔除与 `data/train.jsonl` 完全重复的 223 条 → 按意图分层 + 难度分层抽样 |
| n3 / n4 | `bench/seeds/*.jsonl` 人工种子 16 条 | 场景与训练种子刻意错开；`--synth` 模板扩量仅用于打通流程 |

**分层规则**

1. 按意图分层：每个意图先保证 1 条（长尾保护），再按剩余量轮转补齐。
2. 按难度分层：目标 easy 50% / medium 30% / hard 20%，某档不足时由其它档补足（不硬凑）。
3. 难度启发式（`bench/build_bench.py:difficulty_of`）：
   - `hard`：文本 ≥18 字，或槽位 ≥3 个
   - `easy`：文本 ≤9 字且槽位 ≤1 个
   - 其余 `medium`

**已知限制**

- N1 覆盖 57/59 类：`audio_volume_other`、`general_joke` 等极端长尾类在 400 条配额内被高频类挤掉，评测报告里按 Macro-F1 而非 Accuracy 看更稳妥。
- N3/N4 的 300 条中 288 条是模板合成，**只能用于验证流程，不能作为正式金标**。上卡后用 `data_pipeline` 扩写并人工筛 300 条，替换后需重新跑全部基线。

## 3. 指标定义

所有实现集中在 `eval/metrics.py`，本地打分与 evalscope 侧共用同一份代码。

| 节点 | 指标 | 定义 |
|---|---|---|
| n1 | accuracy | 精确匹配 |
| | macro_f1 | 59 类等权平均，长尾类不被高频类掩盖 |
| | invalid_rate | 输出不在 59 类枚举内的比例（格式失败） |
| n2 | joint_em | intent 与 slots（含顺序）完全一致，与 `eval/baseline_eval.py` 口径一致 |
| | json_valid_rate | 能解析出 `{"intent","slots"}` 的比例 |
| | slot_f1 | 槽位 (name, value) 集合级 P/R/F1（无序） |
| | hallucination_slot_rate | value 在原文中找不到出处的槽位占比（按槽位） |
| | hallucination_sample_rate | 至少含一个幻觉槽位的样本占比 |
| n3 | constraint_pass_rate | 同时满足长度 / 必备信息 / 禁用词的比例 |
| | avg_chars / empty_rate | 平均字数、空输出率 |
| | （人工）quality_1to5 | 抽 100 条人工或评委模型打分，不进自动化报告 |
| n4 | precision / recall / f1 | 以 `escalate=true` 为正类 |
| | miss_rate | FN/(TP+FN)，**该转人工却没转**，业务代价最高 |
| | json_valid_rate / reason_complete_rate | 格式与理由完整度 |

## 4. 评测口径（不可变项）

```
system prompt  = bench/<node>/system.txt（与训练时完全一致）
temperature    = 0
do_sample      = False
max_tokens     = {n1:24, n2:256, n3:192, n4:96}
解析方式       = eval/metrics.py 的 extract_json / parse_intent
```

这五项任何一项变化，历史分数作废。改 prompt 必须重跑基线。

## 5. 泄漏控制

| 检查 | 方法 | 判定 |
|---|---|---|
| 训练/测试精确重复 | 去空白后全等 | 抽样前直接剔除（已剔除 223 条） |
| 近似泄漏 | 8-gram 命中比例 | ≥0.8 判为疑似；短文本（<16 字）不参与判定 |
| 合成数据 | `data_pipeline/03_filter.py` 与 `04_mix_split.py` 各查一次 | 命中即丢弃 |

中文短句用 8-gram 而非 13-gram：13 字以下的文本切不出有效 13-gram，检查会退化为全真或全假。

## 6. 执行流程

```bash
# 1. 构造基准集（本地，零卡时）
python bench/build_bench.py                       # 正式版：N3/N4 只放人工种子
python bench/build_bench.py --synth 300           # 流程演练：N3/N4 模板扩到 300

# 2. 实例上评测（服务需先起）
python eval/run_evalscope.py --tag baseline --general --perf
python eval/run_evalscope.py --dry-run            # 本地预检命令

# 3. 转预测 → 打分
python eval/evalscope_to_pred.py --work-dir reports/evalscope/baseline --tag baseline
python eval/score_bench.py --pred-dir reports/preds/baseline --tag baseline
```

产物：
- `reports/bench/<tag>_summary.md`：汇总表，直接贴文档
- `reports/bench/<tag>_<node>_failures.jsonl`：失败样例，P4 差距分析的原材料

## 7. 变更记录

| 日期 | 变更 | 影响 |
|---|---|---|
| 2026-11-10 | 初版：4 节点 / 1400 条 / 8-gram 泄漏检查 | 建立基线口径 |
