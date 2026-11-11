#!/usr/bin/env bash
# 代码/产物同步：把实例上的结果拉回本地，或把本地改动推上实例。
# 租卡方案里"忘了回传"是最大的隐性成本——实例一释放，训练几小时的结果就没了。
#
# 注意：本脚本依赖 rsync，Windows 本地默认没有（Git for Windows 也不带）。
# Windows 侧请改用 Windows 自带的 OpenSSH scp（注意 scp 的端口是大写 -P）：
#   scp -P <端口> -r root@<主机>:/root/autodl-tmp/SFT/reports ./reports_remote
#   scp -P <端口> sft_code.tar.gz root@<主机>:/root/autodl-tmp/
# 本脚本适合在实例上执行，或在装了 rsync 的 WSL/Git Bash 里执行。
#
#   REMOTE=root@connect.xxx:22 REMOTE_DIR=/root/autodl-tmp/SFT bash deploy/sync.sh push
#   REMOTE=root@connect.xxx:22 REMOTE_DIR=/root/autodl-tmp/SFT bash deploy/sync.sh pull
#
# Windows 本地可用 Git Bash 执行，或直接在实例上执行 pull 的反向版本。

set -euo pipefail

ACTION="${1:-pull}"
REMOTE="${REMOTE:?请设置 REMOTE，例如 root@connect.xxx:22}"
REMOTE_DIR="${REMOTE_DIR:-/root/autodl-tmp/SFT}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$ACTION" in
  push)
    echo "==> 本地 → 实例（代码与数据，不含 outputs/models）"
    rsync -avz --progress \
      --exclude 'outputs/' --exclude 'models/' --exclude '__pycache__/' --exclude '*.pyc' \
      -e "ssh -p ${REMOTE##*:}" \
      "$LOCAL_DIR/" "${REMOTE%@*}:$REMOTE_DIR/"
    ;;
  pull)
    echo "==> 实例 → 本地（报告、产物、日志）"
    mkdir -p "$LOCAL_DIR/reports/remote" "$LOCAL_DIR/outputs/remote"
    rsync -avz --progress \
      -e "ssh -p ${REMOTE##*:}" \
      "${REMOTE%@*}:$REMOTE_DIR/reports/" "$LOCAL_DIR/reports/remote/"
    rsync -avz --progress \
      --include 'best/' --include 'best/**' --exclude '*' \
      -e "ssh -p ${REMOTE##*:}" \
      "${REMOTE%@*}:$REMOTE_DIR/outputs/" "$LOCAL_DIR/outputs/remote/"
    ;;
  *)
    echo "用法：bash deploy/sync.sh push|pull"; exit 1 ;;
esac

echo "完成。"
