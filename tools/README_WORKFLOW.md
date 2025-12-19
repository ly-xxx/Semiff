# Semiff 工具链工作流程

## 概述

我们采用了分步骤的流水线模式，每个脚本只负责一个明确的任务，便于调试和维护。经过优化，我们将传统点云和 3DGS 两种方案合并为一个统一的 5 步流程。

## 统一流程

### Step 1: 场景重建 (step1_reconstruct.py)
**输入**: 视频文件 (test_bench.mp4)
**输出**:
- `outputs/mast3r_result/scene.ply`: 重建的稠密点云
- `outputs/mast3r_result/poses.npy`: 相机位姿

**功能**: 使用 MASt3R 进行 3D 重建，生成场景几何。

### Step 2: 语义分割 (step2_segment.py)
**输入**: 视频文件
**输出**:
- `outputs/masks_object/*.png`: 物体掩码
- `outputs/masks_robot/*.png`: 机器人掩码
- `outputs/final_vis_ffmpeg.mp4`: 可视化视频

**功能**: 使用 SAM2 进行视频分割，识别场景中的物体和机器人。

### Step 3: 资产生成 (step3_assets.py)
**输入**: 场景点云 + 物体掩码 + 相机位姿
**输出**:
- `outputs/assets/object_raw.ply`: 物体点云
- `outputs/assets/object.obj`: 物体网格
- `outputs/assets/object_collision.obj`: 碰撞体
- `outputs/assets/asset_info.json`: 资产信息

**功能**: 从场景中提取物体，生成网格和物理碰撞体。

### Step 4: 机器人建模 (step4_robot.py)
**输入**: 机器人掩码 + URDF
**输出**:
- `outputs/robot_gs_training/robot_gs.ply`: 训练好的 3DGS 模型
- `outputs/assets/robot_binding/`: 高斯点绑定数据

**功能**: 3DGS 训练 + 高斯点绑定到机器人 Link

### Step 5: 集成验证 (step5_integrate.py)
**输入**: 所有步骤输出
**输出**:
- 集成可视化界面
- `outputs/pipeline_report.json`: 完整验证报告

**功能**: 验证整个流水线输出，生成最终的可视化和报告。

## 快速开始

### 完整流水线执行

```bash
# Step 1: 场景重建
python tools/step1_reconstruct.py

# Step 2: 语义分割
python tools/step2_segment.py

# Step 3: 资产生成
python tools/step3_assets.py

# Step 4: 机器人建模
python tools/step4_robot.py

# Step 5: 集成验证
python tools/step5_integrate.py
```

### 配置环境
```bash
# 基础依赖
pip install -e .

# 高级功能依赖
pip install nerfstudio plyfile scikit-learn open3d trimesh yourdfpy
```

### 准备数据
- 视频文件: `test_bench.mp4`
- URDF 文件: 指定 `--urdf` 参数
- 机器人日志: 指定 `--logs` 参数 (传统模式需要)

## 输出文件结构

```
outputs/
├── mast3r_result/
│   ├── scene.ply          # 完整场景点云
│   └── poses.npy          # 相机位姿
├── masks_object/          # 物体掩码
├── masks_robot/           # 机器人掩码
├── assets/
│   ├── object_raw.ply     # 物体点云
│   ├── object.obj         # 物体网格
│   ├── object_collision.obj  # 碰撞体
│   ├── asset_info.json    # 资产信息
│   └── robot_binding/     # 3DGS模式专用
│       ├── binding_data.pkl           # 高斯点绑定信息
│       └── transformed_gaussians.pkl  # 变换后的高斯数据
├── robot_gs_training/     # 机器人 3DGS 训练
│   ├── images/            # 机器人区域图像
│   └── robot_gs.ply       # 训练好的 3DGS 模型
├── final_vis_ffmpeg.mp4   # 分割可视化视频
└── pipeline_report.json   # Step 5 生成的完整报告
```

## 在仿真中使用 3DGS 绑定

### 基本用法

```python
from semiff.simulation.warp_env import GaussianRobotRenderer

# 1. 创建渲染器
renderer = GaussianRobotRenderer(
    binding_data_path="outputs/assets/robot_binding/binding_data.pkl",
    urdf_path="path/to/robot.urdf"
)

# 2. 更新机器人姿态
joint_angles = {"joint1": 0.5, "joint2": 0.3}
renderer.update_robot_pose(joint_angles)

# 3. 获取当前高斯点位置用于渲染
current_positions = renderer.get_gaussian_positions(canonical_positions)
```

### 集成到 Diff-Phys 循环

在物理仿真循环中，根据关节角度实时更新高斯点位置，实现端到端的可微分渲染。

## 故障排除

### 常见问题

1. **Step 1 失败**: 检查 MASt3R 模型权重路径和 GPU 内存
2. **Step 2 失败**: 检查 SAM2 模型和配置，确认视频文件存在
3. **Step 3 失败**: 确保 Step 1-2 成功完成，检查掩码文件
4. **Step 4 失败**: 检查 URDF 路径，确认 plyfile/sklearn/yourdfpy 已安装
5. **Step 5 失败**: 确保前面步骤都成功，至少需要场景点云

### 调试建议

- 每个步骤都会输出详细的日志信息
- 可以单独运行任何步骤进行调试
- Step 5 会生成完整的验证报告，帮助定位问题
- 3DGS 训练需要强大的 GPU 和足够显存

### 性能优化

- 所有步骤都需要 GPU (MASt3R, SAM2, 3DGS)
- SAM2 分割可以考虑关键帧采样减少计算量
- 3DGS 训练耗时长，可减少迭代次数进行快速测试
- ICP 对齐对点云密度敏感，可适当降采样加速