#!/usr/bin/env bash
# vLLM 启动脚本（base / bf16 / gptq 三套参数，避免手敲参数不一致导致评测不可比）
#
#   MODE=base bash deploy/serve.sh     # 原始 Qwen3-14B：S1 基线 + 压测
#   MODE=bf16 bash deploy/serve.sh     # 微调 merge 后的 bf16：S2 复评（与 base 同一套参数）
#   MODE=gptq bash deploy/serve.sh     # GPTQ W4A16：S2 量化复评 & 终评压测
#
# 量化前后必须用同一套 max-model-len / gpu-memory-utilization，
# 只有 max-num-seqs 允许因显存变宽而调大——否则吞吐对比没有意义。
#
# 本地零卡时预检（不装 vllm 也能验证参数解析）：
#   CHECK_ONLY=1 MODEL_PATH=/tmp/fake bash deploy/serve.sh

set -euo pipefail

MODE="${MODE:-base}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"
CHECK_ONLY="${CHECK_ONLY:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Qwen3 默认开 thinking，评测/训练口径要求关闭；模板层钉死，客户端不用每次传参
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$SCRIPT_DIR/chat_template_qwen3_nothink.jinja}"

case "$MODE" in
  base)
    MODEL_PATH="${MODEL_PATH:-$DATA_ROOT/models/qwen3-14b}"      # 原始权重，S1 基线用这个
    SERVED_NAME="${SERVED_NAME:-qwen3-14b}"
    MAX_SEQS="${MAX_SEQS:-32}"
    QUANT_ARGS=()
    ;;
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
    echo "MODE 只能是 base / bf16 / gptq"; exit 1 ;;
esac

echo "==> 生效参数（$MODE）"
echo "    model=$MODEL_PATH"
echo "    served_name=$SERVED_NAME max_len=$MAX_LEN max_seqs=$MAX_SEQS gpu_util=$GPU_UTIL port=$PORT"
echo "    quant_args=${QUANT_ARGS[*]:-无}"
echo "    chat_template=${CHAT_TEMPLATE:-模型自带}"

if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "模型目录不存在或不完整：$MODEL_PATH"
  if [ -d "$DATA_ROOT/models" ]; then
    echo "可用模型目录："
    ls -1 "$DATA_ROOT/models"
  fi
  echo "提示：S1 基线用 MODE=base（models/qwen3-14b），微调后复评用 MODE=bf16（models/qwen3-14b-ft）"
  [ "$CHECK_ONLY" = "1" ] || exit 1
  echo "（CHECK_ONLY=1，仅预检参数，不退出）"
fi

if [ "$CHECK_ONLY" = "1" ]; then
  echo "==> CHECK_ONLY 模式，跳过显存检查与启动"
  exit 0
fi

echo "==> 显存检查"
df -h "$DATA_ROOT" | tail -1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

echo "==> 启动 $SERVED_NAME ($MODE)"

# 提示：启动日志要重点看这四行，抄进 docs/env-matrix.md §4
#   GPU KV cache size / Maximum concurrency for 8192 tokens / non-torch memory / attention backend
CHAT_ARGS=()
if [ -n "${CHAT_TEMPLATE:-}" ] && [ -f "${CHAT_TEMPLATE}" ]; then
  CHAT_ARGS=(--chat-template "$CHAT_TEMPLATE")
fi

exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --dtype bfloat16 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-seqs "$MAX_SEQS" \
  ${QUANT_ARGS[@]+"${QUANT_ARGS[@]}"} \
  ${CHAT_ARGS[@]+"${CHAT_ARGS[@]}"} \
  --port "$PORT"
