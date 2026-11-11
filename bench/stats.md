# 基准集统计

- 生成时间：2026-11-10 20:20:07
- 随机种子：42
- 注意：本版含模板合成样本，只用于打通评测流程；正式基线必须换成人工筛选样本后重跑

## n1_intent

- 条数：400（模板合成 0 条）
- 类别数：59
- 难度分布：{'easy': 194, 'medium': 120, 'hard': 86} → {'easy': 0.485, 'medium': 0.3, 'hard': 0.215}
- 长度 min/avg/max：2/11.2/44
- 长尾类（<5 条）：audio_volume_down, audio_volume_other, general_greet, iot_hue_lighton, iot_wemo_off, music_dislikeness, transport_taxi, weather_query

## n2_slot

- 条数：400（模板合成 0 条）
- 类别数：54
- 难度分布：{'easy': 194, 'medium': 120, 'hard': 86} → {'easy': 0.485, 'medium': 0.3, 'hard': 0.215}
- 长度 min/avg/max：1/10.8/48
- 长尾类（<5 条）：audio_volume_down, audio_volume_up, datetime_convert, email_addcontact, email_querycontact, iot_cleaning, iot_wemo_off, iot_wemo_on

## n3_reply

- 条数：300（模板合成 288 条）
- 类别数：20
- 难度分布：{'hard': 4, 'easy': 4, 'medium': 292} → {'hard': 0.013, 'easy': 0.013, 'medium': 0.973}
- 长度 min/avg/max：63/69.6/97
- 长尾类（<5 条）：account, address_change, after_sale_status, coupon_rule, invoice, logistics_delay, membership, payment_fail, privacy, product_spec, reject_parcel, urge_shipment

## n4_escalate

- 条数：300（模板合成 288 条）
- 类别数：13
- 难度分布：{'easy': 5, 'medium': 292, 'hard': 3} → {'easy': 0.017, 'medium': 0.973, 'hard': 0.01}
- 长度 min/avg/max：59/88.0/108
- 长尾类（<5 条）：abuse, address_change, coupon, legal, logistics_query, membership, money_dispute, product_spec, refund_dispute, return_rule, safety, urge

## 泄漏检查（与 data/train.jsonl 的 8-gram 重叠）

- n1_intent：检查 400 条，疑似泄漏 1 条
- n2_slot：检查 400 条，疑似泄漏 0 条
- n3_reply：检查 300 条，疑似泄漏 0 条
- n4_escalate：检查 300 条，疑似泄漏 0 条
