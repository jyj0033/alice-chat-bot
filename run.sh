#!/bin/bash
# 爱丽丝 - 群聊AI伙伴 启动脚本

set -e

echo "=========================================="
echo "  爱丽丝 - 群聊AI伙伴"
echo "=========================================="

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
if [[ $(echo "$PYTHON_VERSION < 3.10" | bc -l) -eq 1 ]]; then
    echo "❌ 需要 Python 3.10+，当前版本: $PYTHON_VERSION"
    exit 1
fi
echo "✓ Python版本: $PYTHON_VERSION"

# 检查依赖
echo "🔍 检查依赖..."
pip show websockets > /dev/null 2>&1 || {
    echo "📦 正在安装依赖..."
    pip install -r requirements.txt
}

# 创建数据目录
mkdir -p data logs

# 运行
echo ""
echo "🚀 启动Bot..."
echo ""
echo "💡 配置可通过Web界面完成: http://localhost:30080"
echo ""

python3 main.py --dashboard "$@"
