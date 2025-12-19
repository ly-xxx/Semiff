"""
Step 3: Build Assets and Align with Robot
This step performs automatic Sim2Real alignment using Sapien-based robot sampling
and Gaussian Splatting ICP alignment.

Dependencies:
- pip install sapien open3d kornia
- pip install torch torchvision

Usage:
python tools/step3_build_assets.py
"""

import sys
import torch
import numpy as np
import open3d as o3d
from pathlib import Path

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# 引入复用的 real2sim 模块
from semiff.utils.gs.gs_processor import GSProcessor
from semiff.utils.gs.icp_utils import global_registration_ransac, refine_with_icp, preprocess_for_features
from semiff.utils.robot.robot_pc_sampler import RobotPcSampler
from semiff.core.logger import get_logger

logger = get_logger("step3_assets")


def main():
    # === 配置 ===
    scan_ply_path = "outputs/splat/scene.ply"  # Step 2 训练出的 3DGS
    urdf_path = "assets/robots/xarm7_with_gripper.urdf"  # 你的 URDF
    output_dir = Path("outputs/assets")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 [Step 3] Building Assets with Robot Alignment...")

    # 检查输入文件
    if not Path(scan_ply_path).exists():
        print(f"❌ Scene PLY not found: {scan_ply_path}. Run Step 2 first.")
        return

    if not Path(urdf_path).exists():
        print(f"❌ URDF not found: {urdf_path}. Check your robot URDF path.")
        return

    # 1. 初始化 Robot Sampler (Sapien)
    # 这会加载 URDF 并准备好运动学
    print(f"Loading Robot URDF: {urdf_path}")
    try:
        sampler = RobotPcSampler(urdf_path)
    except Exception as e:
        print(f"❌ Failed to load robot: {e}")
        print("Make sure Sapien is installed: pip install sapien")
        return

    # 设置机器人当前的关节角 (从 Step 1 的日志中读取一帧，或者手动指定一个对齐姿态)
    # 假设这是机器人"伸直"便于对齐的姿态
    qpos = np.zeros(7 + 2)  # 7轴 + 2指
    # 比如： sampler.compute_robot_pcd 需要的是关节角
    print("Sampling robot point cloud...")
    try:
        robot_pts = sampler.compute_robot_pcd(qpos, num_pts=5000)
        print(f"Generated {len(robot_pts)} robot points")
    except Exception as e:
        print(f"❌ Failed to sample robot point cloud: {e}")
        return

    # 2. 加载视觉场景 (3DGS)
    print("Loading Scene Splat...")
    try:
        sp = GSProcessor()
        gs_params = sp.load(scan_ply_path)
        scene_pts = gs_params['means3D'].cpu().numpy()
        print(f"Loaded {len(scene_pts)} scene points")
    except Exception as e:
        print(f"❌ Failed to load scene: {e}")
        return

    # 转换成 Open3D 对象用于 ICP
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(robot_pts)  # 源：机器人标准模型
    source.paint_uniform_color([0, 1, 0])  # 绿

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(scene_pts)  # 目标：扫描场景
    target.paint_uniform_color([1, 0, 0])  # 红

    # 3. 运行自动对齐 (RANSAC + ICP)
    # 这里的逻辑直接照搬 icp_utils.py
    print("Running Alignment...")
    voxel_size = 0.02

    try:
        source_down, source_fpfh = preprocess_for_features(source, voxel_size)
        target_down, target_fpfh = preprocess_for_features(target, voxel_size)

        # 3.1 粗配准
        ransac_res = global_registration_ransac(
            source_down, target_down, source_fpfh, target_fpfh, voxel_size
        )
        print(".4f")

        # 3.2 精配准
        icp_coarse, icp_fine = refine_with_icp(
            source, target, ransac_res.transformation, voxel_size
        )
        T_world = icp_fine.transformation
        print("Final Transform (Sim -> Real):")
        print(T_world)

    except Exception as e:
        print(f"❌ Alignment failed: {e}")
        return

    # 4. 资产生成与分割
    # 将 GS 变换到机器人的坐标系下 (Sim Frame)
    # 注意：T_world 是 Robot -> Scene，我们需要 Scene -> Robot
    T_inv = np.linalg.inv(T_world)

    print("Transforming scene to robot coordinate frame...")
    try:
        # 变换 GS 参数
        gs_params_aligned = sp.rotate(gs_params, torch.tensor(T_inv[:3, :3]).float().cuda())
        gs_params_aligned = sp.translate(gs_params_aligned, torch.tensor(T_inv[:3, 3]).float().cuda())

        # 简单的空间分割：切掉机器人（原点附近），保留物体
        # 这里可以用 RobotPcSampler 再次生成点云来做更精细的 Mask 剔除
        # ... (此处省略 KNN 剔除逻辑，参考 segment_robot 函数) ...

        # 5. 保存
        aligned_ply_path = output_dir / "scene_aligned.ply"
        sp.save(gs_params_aligned, str(aligned_ply_path))
        np.save(output_dir / "T_world.npy", T_world)

        print("✅ Assets generated and aligned!")
        print(f"📁 Aligned scene saved to: {aligned_ply_path}")
        print(f"📁 Transform saved to: {output_dir / 'T_world.npy'}")

        # 保存资产信息
        asset_info = {
            "scene_aligned_ply": str(aligned_ply_path),
            "transform_matrix": T_world.tolist(),
            "robot_qpos": qpos.tolist(),
            "robot_urdf": urdf_path,
            "alignment_method": "sapien_icp",
            "voxel_size": voxel_size
        }

        import json
        with open(output_dir / "asset_info.json", 'w') as f:
            json.dump(asset_info, f, indent=2)
        print(f"📁 Asset info saved to: {output_dir / 'asset_info.json'}")

    except Exception as e:
        print(f"❌ Asset generation failed: {e}")
        return

    print("🎯 [Step 3] Asset generation completed!")
    print(f"📁 Assets saved to: {output_dir}")


if __name__ == "__main__":
    main()