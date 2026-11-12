# 数据卡（第五步产物）

> 日期：2026-11-12　实例：A100 40G　合成后端：vLLM 0.11.2 离线批量推理（Qwen3-14B bf16，enable_thinking=False）

## 1. 用途与范围

只训练 **N1 意图识别** 与 **N2 槽位抽取**（依据 `docs/gap-analysis.md` 的决策：N3 约束已 100% 通过不微调；N4 基线饱和、先补难样本）。
N3/N4 的数据**不进训练集**，仅保留基准集用于复评。

## 2. 来源

| 部分 | 来源 | 规模 | 说明 |
| --- | --- | --- | --- |
| 真实样本 | `data/train.jsonl`（MASSIVE 子集）按意图分层抽样，每类 20 条 | 2352 条（59 类 × 20 × 2 任务） | 同一句文本同时产出 N1 标签与 N2 槽位两个训练样本 |
| 合成样本 | 种子（真实样本每类 8 条，共 476 条/节点）→ 模型改写 → 模型重标注 | 候选 45696 → 质检后 8703 | 种子与评测集刻意错开；改写与重标注各一轮推理 |

## 3. 生成方式（可复现）

```bash
# 1) 种子（零卡时）
python data_pipeline/01_seed.py
# 2) 扩写：两轮离线推理（改写 temperature=0.8, top_p=0.95；重标注 temperature=0）
python data_pipeline/02_expand.py --backend vllm --model /data/models/qwen3-14b \
    --seeds-dir data_pipeline/out/seeds_train --times 12 --out data_pipeline/out/candidates_full.jsonl
python data_pipeline/02_expand.py ... --times 36 --out data_pipeline/out/candidates_more.jsonl
# 3) 质检（结构/标签一致性/幻觉/去重/配额）
python data_pipeline/03_filter.py --input data_pipeline/out/candidates_all.jsonl
# 4) 配比 + 切分
python data_pipeline/04_mix_split.py --real-per-class 20 --ratios 1:1 1:3
```

- 四种改写风格轮转：口语化 / 错别字 / 极简 / 冗余铺垫。
- 实测吞吐：改写轮约 **2091~2233 tok/s**，45696 条候选两轮推理合计约 10 分钟卡时。
- 校准先行：先 `--limit 200` 测速与自标一致率（69%），再全量。

## 4. 各级过滤保留率

| 节点 | 候选 | 质量通过 | 去重后 | 最终保留率 |
| --- | --- | --- | --- | --- |
| N1 意图 | 22848 | 17203（75.3%） | 7132（41.5%） | **31.2%** |
| N2 槽位 | 22848 | 4403（19.3%） | 1571（6.9%） | **6.9%** |

淘汰原因（可叠加）：

- **N1**：重复 10069、标签与种子不一致 5362、标签不在枚举内 283、与评测集重叠 2。
- **N2**：槽位与种子不一致 12217、意图与种子不一致 6228、重复 2831、与评测集重叠 1。

> N2 保留率低是**预期**而非缺陷：槽位口径要求严（模型改写后常自行增删槽位），
> 这正是 N2 基线 Slot-F1 仅 35.9% 的同一根因，严过滤保证训练标签口径与金标一致。

## 5. 训练集配比与规模

| 名称 | 真实:合成 | train / validation | N1 / N2 占比 |
| --- | --- | --- | --- |
| `domain_sft_1to1` | 2352 : 2352（1:1） | 4233 / 471 | 2807 / 1426 |
| `domain_sft_1to3` | 2352 : 7056（1:3，实际 2352:7056 足额） | 8467 / 941 | 6260 / 2207 |

- 切分：按 9:1 随机切 train/validation，seed=42。
- manifest 记录 `actual_ratio` 与 train 集 `sha256`（`data/domain_sft_*_manifest.json`）。

## 6. 去重与泄漏检查

- 合成集内部：同节点 8-gram MinHash 指纹去重。
- 与基准集：8-gram 命中比例 ≥0.8 直接丢弃（N1/N2 各命中 2/1 条，已剔除）。
- 最终训练集 vs `bench/*/test.jsonl`：**8-gram suspicious = 0**（manifest 的 `leak_check` 全 0）。

## 7. 类别分布与配额

- 覆盖 60 个 category（59 意图 + unknown 汇总项），真实样本每类 20 条保证长尾类下限。
- `03_filter` 按 category 截断，防高频意图（`calendar_query`/`play_music` 等）淹没长尾。
- N1 高危类别（`calendar_set`/`music_likeness`/`email_query`/`datetime_convert`）在合成中占比高于基线失败分布，见 `docs/gap-analysis.md` §2.2。

## 8. 已知偏差与风险

1. **合成标签继承种子口径**：槽位粗细粒度沿用 MASSIVE 金标（如 `music_genre` 粗粒度），改写后若模型重标注粒度漂移会被一致性过滤剔除，代价是 N2 合成量偏少（1571 条）。
2. **模板化残留**：真实样本为 MASSIVE 语音助手域，与客服业务话术有分布差异；改写只增加表达多样性，不改变任务域。
3. **模型自标≠人工金标**：合成标签正确性依赖"两轮一致性"这一较弱保证；评测集全部为人工/MASSIVE 金标，故指标可信度不受影响。
4. N3/N4 本轮未训练，若后续补"隐含风险"难样本需重建基准集并重跑基线。

## 9. 许可与隐私

- 真实样本来自公开数据集 MASSIVE（CC BY 4.0），无真实用户隐私数据。
- 合成样本由本地模型生成，不含个人信息；无爬取数据。
