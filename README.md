# Semiff: Real-to-Sim-to-Real Pipeline

统一计算空间智能环境架构：Warp + gsplat + ProPainter + VoMP (xArm Edition)

## 简介

Semiff 是一个完整的 Real-to-Sim-to-Real 流水线框架，用于将现实世界的机器人和环境转换为物理可仿真的数字孪生体。

## 主要特性

- **多阶段流水线**：数据摄取 → 几何感知 → 语义感知 → 坐标对齐 → 资产生成
- **模块化设计**：清晰的架构，支持独立测试和扩展
- **生产级代码**：完善的错误处理、日志记录和文档

## 安装

```bash
# 激活 uv 环境
source .venv/bin/activate

# 同步依赖
uv sync

# 安装项目为可编辑包
pip install -e .
```

## 使用

```python
python main.py
```

## 架构

```
📁 semiff/
├── core/          # 基础IO和工具
├── perception/    # 视觉感知 (MASt3R, SAM2)
├── calibration/   # 坐标对齐 (ICP, Sim2Real)
├── geometry/      # 几何处理 (点云重建, 凸包分解)
└── rendering/     # 渲染准备 (Nerfstudio数据集)
```



