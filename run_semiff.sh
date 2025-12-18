#!/bin/bash
# Semiff 运行脚本

# 设置代理
export HTTP_PROXY=http://172.23.186.41:7890
export HTTPS_PROXY=http://172.23.186.41:7890

# 激活环境
source .venv/bin/activate

# 设置 Python 路径
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

echo "🚀 Starting Semiff Pipeline..."
echo "Environment: $(which python)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo ""

# 运行主程序
python main.py



