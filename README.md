# Semiff: Kinematic-Anchored Sim2Real Asset Construction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

**Semiff** (Semantic & Diffuser) 是一个基于单目视频的机器人场景数字化重建流水线。
其核心理念是 **"Physics as Anchor"**：利用机器人自身的精确运动学模型（URDF）作为几何真值，通过可微渲染技术消除单目视觉重建（MASt3R/3DGS）中的尺度二义性和坐标系漂移。

---

## 🏗 Pipeline 架构

本项目通过四个线性步骤将单目 RGB 视频转化为可物理仿真的 Digital Twin 资产：

### Step 1: Data Solver (视觉解算)
> *Status: ✅ Completed*
* **目标**: 从非结构化视频中提取几何与语义信息。
* **核心技术**:
    * **MASt3R**: 提取稠密 3D 点云与相机位姿 (Camera Poses)。
    * **SAM 2.1**: 分割机械臂 (Robot) 与目标物体 (Object) 的 2D 掩码。
* **输出**: 稀疏点云、相机轨迹、语义掩码序列。

### Step 2: Self-Calibration (运动学自标定)
> *Status: 🚧 In Progress*
* **目标**: 解决 Sim2Real 的 "Scale Ambiguity" 问题。
* **核心原理**:
    * 固定 URDF 几何真值（Hard Constraint）。
    * 优化视觉世界的 `Scale` 和 `Base_Transform`。
    * Loss: 渲染的 URDF 投影 vs SAM2 分割掩码 (IoU Loss)。
* **工具**: `nvdiffrast`, `pytorch_kinematics`.

### Step 3: Global Reconstruction (全局重建)
> *Status: 📅 Planned*
* **目标**: 训练高质量静态 3D Gaussian Splatting (3DGS) 场。
* **策略**: 使用 Step 2 矫正后的尺度或位姿进行训练，确保场景具有真实的物理度量单位。

### Step 4: Asset Decomposition (资产拆解)
> *Status: 📅 Planned*
* **目标**: 生成 Warp/Isaac Gym 可用的资产。
* **逻辑**: 基于对齐后的 URDF 进行空间查询（Geometric Query），将 3DGS 点云切割为 Robot、Object 和 Background，并完成骨骼绑定。

---

## 📂 项目结构

```text
semiff/
├── configs/                # Hydra/OmegaConf 配置文件
│   ├── base_config.yaml    # 基础配置
│   └── ...
├── data/                   # 输入数据目录
│   └── example_01/         # 示例数据集
│       ├── video.mp4       # 原始视频
│       ├── config/         # 机器人关节配置
│       └── robot/          # URDF 资产
├── outputs/                # 实验输出 (自动按时间戳生成)
│   └── 20260106_210039/    # 某次运行结果
│       ├── camera_poses.npy
│       ├── masks_robot/
│       └── ...
├── src/                    # 核心源码
│   └── semiff/
│       ├── core/           # 渲染、IO、配置管理
│       ├── perception/     # SAM2, MASt3R 封装
│       └── utils/          # 数学工具库
├── tools/                  # 执行脚本
│   ├── step1_preprocess.py # [Done] 数据预处理
│   └── step2_calibrate.py  # [Todo] 自标定脚本
└── main.py                 # 统一入口
```

## 🚀 快速开始

### 环境依赖
请确保安装带有 OpenGL 支持的 NVIDIA 驱动 (用于 nvdiffrast)。

```bash
pip install -r requirements.txt
# 额外安装 nvdiffrast
git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast
```

### 运行 Step 1: 预处理
```bash
python tools/step1_preprocess.py \
    data_dir=data/example_01 \
    output_dir=outputs/new
```

### 运行 Step 2: 自标定 (Coming Soon)
```bash
python tools/step2_calibrate.py \
    workspace=outputs/20260106_210039 \
    robot_urdf=data/example_01/robot/rllab_xarm/xarm6_robot.urdf
```

## 📝 贡献指南

* 视觉坐标系采用 OpenCV 标准 (Right-Down-Forward)。
* 物理坐标系采用 URDF 标准。
* 提交代码前请运行 tests/ 下的单元测试。

---

### 评估与下一步

1.  **关于 `render.py` 的修正建议**：
    * 目前的 `render.py` 中 `build_projection_matrix` 生成的是 NDC 投影，但 `render` 函数似乎没有显式处理 World-to-Camera (View Matrix) 的变换。
    * 在编写 Step 2 代码时，我们需要在传入 `vertices` 给 `render` 函数之前，先手动用 `cam_poses` (Extrinsics) 将其变换到相机坐标系。或者修改 `render.py` 让其接受 `T_view`。我建议在 Step 2 的外部逻辑中处理这个变换，保持 `render.py` 的纯粹性（只负责 Projection + Rasterization）。

2.  **关于机器人姿态**：
    * 既然提供了 `align_pose.json` 且机器人是静止的，Step 2 代码将读取这个 JSON，对所有时刻 $t$ 使用相同的关节角 $q$。这是一个很好的简化。