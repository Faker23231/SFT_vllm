#!/usr/bin/env bash
# 单个消融/微调产物的「merge → 起 bf16 服务 → 四节点复评 → 打分 → 清理」一键脚本。
#
# 用法：
#   TAG=ft-1to1-r16 ATT=${DATA_ROOT}/SFT/outputs/abl-1to1-r16-linear/best \
#     KEEP_MERGED=1 bash eval/ablation_eval.sh
#
# 设计要点：
#   1. merge 与评测共用同一份 bf16 权重，评测口径与 S1 baseline 完全一致（serve.sh MODE=bf16）；
#   2. 服务就绪用 /health 轮询，不靠 sleep 猜时间；
#   3. 评测完默认删除 28G merged 权重（磁盘只容得下一份），KEEP_MERGED=1 时保留；
#   4. 每个实验的产物固定在 reports/{evalscope,preds,bench}/<TAG>/，便于横向对比。

set -euo pipefail

TAG="${TAG:?需要 TAG，例如 TAG=ft-1to1-r16}"
ATT="${ATT:?需要 ATT=adapter 目录，例如 outputs/abl-1to1-r16-linear/best}"
DATA_ROOT="${DATA_ROOT:-/data}"
REPO="${REPO:-$DATA_ROOT/SFT}"
PORT="${PORT:-8000}"
KEEP_MERGED="${KEEP_MERGED:-0}"
MERGED="${MERGED:-$REPO/outputs/merged-$TAG}"
PY="$DATA_ROOT/venv/bin/python"

cd "$REPO"

echo "==> [1/5] merge adapter → $MERGED"
if [ ! -f "$MERGED/config.json" ]; then
  "$PY" eval/merge_lora.py --adapter "$ATT" --output "$MERGED"
else
  echo "    已存在，跳过 merge"
fi

echo "==> [2/5] 起 bf16 服务（MODE=bf16, port=$PORT）"
MODEL_PATH="$MERGED" MODE=bf16 PORT="$PORT" DATA_ROOT="$DATA_ROOT" \
  nohup bash deploy/serve.sh > "$DATA_ROOT/logs/serve_${TAG}.log" 2>&1 &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null || true' EXIT

for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "    服务就绪（等待 ${i}0s 内）"
    break
  fi
  if ! kill -0 $SERVE_PID 2>/dev/null; then
    echo "[fail] 服务进程已退出，看 $DATA_ROOT/logs/serve_${TAG}.log"; exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "[fail] 服务未就绪"; exit 1; }

echo "==> [3/5] 四节点评测 → reports/evalscope/$TAG"
"$PY" eval/run_evalscope.py --tag "$TAG"

echo "==> [4/5] 结果转换 + 打分"
"$PY" eval/evalscope_to_pred.py --work-dir "reports/evalscope/$TAG" --tag "$TAG"
"$PY" eval/score_bench.py --pred-dir "reports/preds/$TAG" --tag "$TAG"

echo "==> [5/5] 收尾"
kill $SERVE_PID 2>/dev/null || true
trap - EXIT
sleep 10
if [ "$KEEP_MERGED" = "1" ]; then
  echo "    保留 merged 权重：$MERGED"
else
  find "$MERGED" -mindepth 1 -delete 2>/dev/null || true
  rmdir "$MERGED" 2>/dev/null || true
  echo "    已清理 merged 权重（KEEP_MERGED=1 可保留）"
fi
echo "[done] $TAG 复评完成：reports/bench/${TAG}_summary.md"
