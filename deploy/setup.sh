#!/usr/bin/env bash
# 实例初始化（S1 开卡后第一件事）。做完立刻在平台上保存镜像，S2 直接用。
#
#   bash deploy/setup.sh                 # 默认装到 /root/autodl-tmp
#   DATA_ROOT=/data bash deploy/setup.sh # 指定数据盘
#
# 关键纪律：所有东西都放数据盘。系统盘通常只有 30~50G，14B 权重 28G 会直接写满。

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
MODEL_DIR="${MODEL_DIR:-$DATA_ROOT/models/qwen3-14b}"
VENV="${VENV:-$DATA_ROOT/venv}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> 磁盘检查"
df -h "$DATA_ROOT" || { echo "数据盘不存在：$DATA_ROOT"; exit 1; }
mkdir -p "$DATA_ROOT"/{models,work,logs,reports}

echo "==> GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "==> Python 环境"
PY_VERSION="${PY_VERSION:-3.12}"   # 跟随镜像自带版本；需要时 PY_VERSION=3.11 让 uv 自动下载
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple >/dev/null
pip install -U uv >/dev/null
# venv 是独立环境，镜像预装的 torch 不会进来；torch 版本由下一步的 vllm 决定
uv venv "$VENV" --python "$PY_VERSION"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
uv pip install -U pip
# 安装顺序很重要：vLLM 对 torch 版本 pin 得很死，先手动装 torch 再装 vllm，
# 会导致 uv 把 torch 降级重装（几百 MB）。正确做法是让 vllm 决定 torch 版本。
# uv >= 0.7 用 --torch-backend 指定 CUDA 版本；老版本会报错，去掉该参数即可（默认 cu12x）。
uv pip install vllm --torch-backend=cu128
uv pip install evalscope llm-compressor transformers peft trl datasets accelerate tensorboard

echo "==> 依赖版本（记进 docs/env-matrix.md）"
python - <<'PY'
import platform, torch, transformers, peft, trl
print("python", platform.python_version())
print("torch", torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability())
print("transformers", transformers.__version__, "peft", peft.__version__, "trl", trl.__version__)
PY
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
vllm --version || true

echo "==> 下载模型到 $MODEL_DIR"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$DATA_ROOT/hf"
if [ -f "$MODEL_DIR/config.json" ]; then
  echo "已存在，跳过下载"
else
  huggingface-cli download "$MODEL" --local-dir "$MODEL_DIR" --exclude "*.pth" "original/*"
fi

cat <<EOF

==> 完成。接下来：
    1) 启动服务：bash deploy/serve.sh            # 默认 bf16
    2) 冒烟：curl -s localhost:8000/v1/models | head
    3) 立刻在平台保存镜像（S2 省 40 分钟重装）
    4) 本地代码同步上来：bash deploy/sync.sh push
EOF
