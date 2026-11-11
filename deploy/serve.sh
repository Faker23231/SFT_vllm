#!/usr/bin/env bash
# vLLM 启动脚本（bf16 / GPTQ 两套参数，避免手敲参数不一致导致评测不可比）
#
#   MODE=bf16 bash deploy/serve.sh     # 基线 & 微调后复评
#   MODE=gptq bash deploy/serve.sh     # 量化模型复评 & 终评压测
#
# 量化前后必须用同一套 max-model-len / gpu-memory-utilization，
# 只有 max-num-seqs 允许因显存变宽而调大——否则吞吐对比没有意义。

set -euo pipefail

MODE="${MODE:-bf16}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"

case "$MODE" in
  bf16)
    MODEL_PATH="${MODEL_PATH:-$DATA_ROOT/models/qwen3-14b-ft}"   # 微调 merge 后的权重
    SERVED_NAME="${SERVED_NAME:-qwen3-14b}"
    MAX_SEQS="${MAX_SEQS:-32}"
    QUANT_ARGS=()
    ;;
  gptq)
    MODEL_PATH="${MODEL_PATH:-$DATA_ROOT/models/qwen3-14b-gptq-w4a16}"
    SERVED_NAME="${SERVED_NAME:-qwen3-14b-gptq}"
    MAX_SEQS="${MAX_SEQS:-64}"
    QUANT_ARGS=(--quantization gptq_marlin)
    ;;
  *)
    echo "MODE 只能是 bf16 或 gptq"; exit 1 ;;
esac

if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "模型目录不存在或不完整：$MODEL_PATH"; exit 1
fi

echo "==> 显存检查"
df -h "$DATA_ROOT" | tail -1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

echo "==> 启动 $SERVED_NAME ($MODE)"
echo "    model=$MODEL_PATH max_len=$MAX_LEN max_seqs=$MAX_SEQS gpu_util=$GPU_UTIL"

# 提示：启动日志要重点看这四行
#   GPU KV cache size / Maximum concurrency / non-torch memory / attention backend
exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --dtype bfloat16 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-seqs "$MAX_SEQS" \
  "${QUANT_ARGS[@]}" \
  --port "$PORT"
