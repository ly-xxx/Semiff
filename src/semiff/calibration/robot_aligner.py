"""
Robot Aligner: Sim2Real Calibration via ICP

DEPRECATED: This module is deprecated and will be removed in future versions.
Use src/semiff/utils/robot/robot_pc_sampler.py and src/semiff/utils/gs/icp_utils.py instead,
which are based on Sapien and provide better alignment quality.

Migration guide:
- Replace RobotAligner with RobotPcSampler (Sapien-based)
- Use ICP utilities from gs.icp_utils for alignment
"""

import numpy as np
import trimesh
from trimesh.registration import icp
from pathlib import Path
from typing import Dict, Optional, Union, List

# 导入我们的自定义模块
try:
    from yourdfpy import URDF
except ImportError:
    print("Error: yourdfpy not installed. Install with `pip install yourdfpy`")

from ..core.logger import get_logger
from ..core.io import RobotLogger
from .space_trans import RigidTransform

logger = get_logger(__name__)

class RobotAligner:
    """负责计算视觉世界到物理世界的变换矩阵"""

    def __init__(self, urdf_path: Union[str, Path]):
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        logger.info(f"Loading URDF: {self.urdf_path.name}")
        self.robot = URDF.load(str(self.urdf_path))

    def generate_gt_cloud(self, joint_map: Dict[str, float], samples: int = 10000) -> np.ndarray:
        """
        根据关节角度生成机器人的真值点云
        """
        # 1. 更新机器人姿态
        self.robot.update_cfg(joint_map)

        # 2. 获取当前的 Mesh (合并所有 Link)
        # yourdfpy 的 scene.dump(concatenate=True) 返回一个合并后的 trimesh 对象
        robot_mesh = self.robot.scene.dump(concatenate=True)

        # 3. 表面采样
        points, _ = trimesh.sample.sample_surface(robot_mesh, samples)
        return points

    def align(self,
              visual_cloud: np.ndarray,
              robot_mask_path: Optional[Path] = None,
              gt_cloud: Optional[np.ndarray] = None,
              initial_guess: Optional[np.ndarray] = None) -> RigidTransform:
        """
        运行 ICP 对齐

        Args:
            visual_cloud: [N, 3] 来自 MASt3R 的完整场景点云
            robot_mask_path: (可选) 机器人分割 Mask 的路径，用于过滤 visual_cloud
            gt_cloud: [M, 3] 机器人的真值点云 (通常由 generate_gt_cloud 生成)
            initial_guess: [4, 4] 初始变换矩阵猜测

        Returns:
            RigidTransform: T_world (将视觉点云变换到 URDF 基座坐标系)
        """
        # 1. 过滤视觉点云 (只保留机器人部分)
        # 注意：这里简化了逻辑，假设 visual_cloud 已经是经过 Mask 过滤的，或者我们在此处不做像素级映射
        # 实际生产中，你需要将 3D 点投影回 2D 图像，用 Mask 检查该点是否属于机器人
        # 为简化演示，我们假设输入的 visual_cloud 主要是机器人
        if robot_mask_path:
            logger.info("Note: Assuming visual_cloud is already masked/segmented for the robot.")

        source_points = visual_cloud
        target_points = gt_cloud

        if source_points.shape[0] < 100 or target_points.shape[0] < 100:
            raise ValueError("Point clouds too sparse for ICP.")

        logger.info(f"Starting ICP... Source: {len(source_points)} pts, Target: {len(target_points)} pts")

        # 2. 预处理：去中心化 (Centering)
        # ICP 对初始位置非常敏感，先将两个点云的重心对齐
        source_mean = source_points.mean(axis=0)
        target_mean = target_points.mean(axis=0)

        source_centered = source_points - source_mean
        target_centered = target_points - target_mean

        # 3. 运行带尺度的 ICP (Scaled ICP)
        # trimesh.registration.icp 会返回变换矩阵和 cost
        # 注意: scale=True 允许求解缩放因子，这是 Sim2Real 的关键
        matrix, transformed_source, cost = icp(
            source_centered,
            target_centered,
            scale=True,
            max_iterations=100,
            threshold=1e-5
        )

        # 4. 恢复绝对变换
        # 我们计算的是 Centered 坐标系的变换 T_c
        # 最终变换 T = T_target_back @ T_c @ T_source_to_center

        # T_source_to_center: 平移 -source_mean
        T_s2c = np.eye(4)
        T_s2c[:3, 3] = -source_mean

        # T_c: ICP 算出的矩阵 (包含旋转、缩放、微太平移)
        T_c = matrix

        # T_target_back: 平移 +target_mean
        T_t2b = np.eye(4)
        T_t2b[:3, 3] = target_mean

        # 组合: T_final = T_t2b * T_c * T_s2c
        T_final = T_t2b @ T_c @ T_s2c

        scale_est = np.cbrt(np.linalg.det(T_final[:3, :3]))
        logger.info(f"ICP Converged. Cost: {cost:.4e}. Est Scale: {scale_est:.4f}")

        return RigidTransform(T_final)


def align_visual_to_robot(
    visual_cloud: np.ndarray,
    robot_mask: str,  # 实际上这里还没用到 mask 具体数据，假设外部已处理
    robot_urdf: str,
    robot_logs: str,
    timestamp: float = 0.0
) -> RigidTransform:
    """
    便捷入口函数
    """
    # 1. 准备 Robot Aligner
    aligner = RobotAligner(robot_urdf)

    # 2. 读取日志并生成 GT Cloud
    # 这里为了简单，我们取第 0 帧或指定时间戳的姿态
    # 实际应用中，应该选择机器人"伸展最开"的一帧，以获得最好的几何约束
    logger_io = RobotLogger(robot_logs)

    # 获取关节名 (需要与 URDF 匹配)
    # 这一步通常需要手动映射，这里假设日志列名包含了关节名
    # 这是一个简化假设
    try:
        # 取第一行数据作为关节角
        # 注意：这里需要 RobotLogger 能返回 dict {joint_name: angle}
        # 既然我们之前的 IO 只是返回 array，这里做一个 mock 实现
        # TODO: 增强 IO 模块以支持列名映射
        logger.warning("Using mock joint map. Ensure log columns match URDF joint names!")
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        joint_vals = logger_io.get_interpolated_joints([timestamp])[0] # [6]

        joint_map = dict(zip(joint_names, joint_vals))
    except Exception as e:
        logger.error(f"Failed to map joints: {e}. Using default pose.")
        joint_map = {} # default pose

    gt_cloud = aligner.generate_gt_cloud(joint_map)

    # 3. 运行对齐
    transform = aligner.align(visual_cloud, gt_cloud=gt_cloud)

    return transform