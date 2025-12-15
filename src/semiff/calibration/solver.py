"""
Differentiable Kinematics Solver (Production Ready)
无需机器人日志，利用可微渲染思想反推关节角度
"""

import torch
import numpy as np
import pytorch_kinematics as pk
import trimesh
from typing import Dict, Tuple, List, Optional, Union
from pathlib import Path

from ..core.logger import get_logger
from ..calibration.space_trans import RigidTransform

logger = get_logger(__name__)

class ChamferLoss(torch.nn.Module):
    """
    计算两个点云之间的倒角距离 (单向或双向)
    """
    def __init__(self):
        super().__init__()

    def forward(self, source_cloud: torch.Tensor, target_cloud: torch.Tensor,
                bidirectional: bool = True) -> torch.Tensor:
        """
        Args:
            source_cloud: [N, 3] 模型预测点云
            target_cloud: [M, 3] 真实观测点云
        """
        # 显存保护：如果在 GPU 上且点数过多，进行随机降采样
        # 经验值：对于优化任务，2048 个点通常足够提供梯度
        MAX_POINTS = 2048

        if source_cloud.shape[0] > MAX_POINTS:
            idx = torch.randperm(source_cloud.shape[0])[:MAX_POINTS]
            source_cloud = source_cloud[idx]

        if target_cloud.shape[0] > MAX_POINTS:
            idx = torch.randperm(target_cloud.shape[0])[:MAX_POINTS]
            target_cloud = target_cloud[idx]

        # 计算距离矩阵: dist_sq = x^2 + y^2 - 2xy
        # [N, 1] + [1, M] - [N, M]
        src_sq = torch.sum(source_cloud**2, dim=1, keepdim=True)
        tgt_sq = torch.sum(target_cloud**2, dim=1, keepdim=True).t()

        # 此时 matrix shape: [N, M]
        dist_matrix = src_sq + tgt_sq - 2 * torch.matmul(source_cloud, target_cloud.t())

        # 1. Source -> Target (模型点离观测点有多远)
        # 这一项惩罚模型"飘"到没有点云的地方
        min_dist_s2t, _ = torch.min(dist_matrix, dim=1)
        loss_s2t = torch.mean(min_dist_s2t)

        loss = loss_s2t

        # 2. Target -> Source (观测点离模型有多远)
        # 这一项惩罚模型没有覆盖到观测点 (即覆盖率)
        if bidirectional:
            min_dist_t2s, _ = torch.min(dist_matrix, dim=0)
            loss_t2s = torch.mean(min_dist_t2s)
            loss = loss + 0.8 * loss_t2s # 稍微降低覆盖率的权重，容忍遮挡

        return loss

class RobotOptimizer:
    def __init__(self, urdf_path: str, device: str = "cuda"):
        self.device = device
        self.urdf_path = urdf_path

        # 1. 加载运动学链 (用于计算变换矩阵)
        with open(urdf_path, "r") as f:
            self.chain = pk.build_chain_from_urdf(f.read()).to(device=device)

        self.joint_names = self.chain.get_joint_parameter_names()
        self.n_dof = len(self.joint_names)

        # 2. 真实加载几何体 (FIXED: 使用 yourdfpy/trimesh 加载真实 Mesh)
        logger.info("Pre-loading robot geometry for optimization...")
        self.link_templates = self._preload_link_geometries(urdf_path)

        self.loss_fn = ChamferLoss()

        logger.info(f"Initialized Solver for {self.n_dof} DoF robot. Loaded {len(self.link_templates)} visual links.")

    def _preload_link_geometries(self, urdf_path: str, samples_per_link: int = 1000) -> Dict[str, torch.Tensor]:
        """
        加载 URDF 中的 Visual Mesh 并采样为点云
        """
        try:
            from yourdfpy import URDF
        except ImportError:
            raise ImportError("Please install yourdfpy: pip install yourdfpy")

        robot = URDF.load(urdf_path)
        templates = {}

        # 遍历所有 Link
        for link_name, link in robot.link_map.items():
            if not link.visuals:
                continue

            # 一个 Link 可能有多个 Visual 几何体
            link_points = []

            for visual in link.visuals:
                # yourdfpy/trimesh 可能会将 mesh 加载到 scene 中
                # 这里我们直接利用 visual.geometry 的 mesh 属性 (如果是 mesh 类型)
                if visual.geometry.mesh is None:
                    # 尝试处理基本几何体 (box/sphere/cylinder)
                    # trimesh 会自动将它们转换为 mesh
                    geom_mesh = visual.geometry.to_trimesh()
                else:
                    # 加载外部 mesh 文件
                    # 注意：trimesh 处理路径可能需要一些 trick，yourdfpy 通常已经处理好了
                    geom_mesh = visual.geometry.mesh

                if geom_mesh is None:
                    continue

                # 应用 visual 的局部变换 (origin)
                if visual.origin is not None:
                    geom_mesh.apply_transform(visual.origin)

                # 采样点云
                pts, _ = trimesh.sample.sample_surface(geom_mesh, samples_per_link)
                link_points.append(pts)

            if link_points:
                # 合并该 Link 下的所有点
                full_link_pts = np.vstack(link_points)
                # 转为 Tensor 并存入 GPU
                templates[link_name] = torch.from_numpy(full_link_pts).float().to(self.device)

        return templates

    def filter_cloud_by_mask(self,
                           points_3d: np.ndarray,
                           mask: np.ndarray,
                           camera_intrinsics: Dict,
                           camera_pose: Optional[np.ndarray] = None) -> np.ndarray:
        """
        工具函数：利用 2D Mask 裁剪 3D 点云

        Args:
            points_3d: [N, 3] 世界坐标系下的点
            mask: [H, W] 二值掩码 (1=Robot, 0=Background)
            camera_intrinsics: {'fx', 'fy', 'cx', 'cy'}
            camera_pose: [4, 4] World -> Camera 的变换矩阵 (如果点云已经是相机坐标系则设为 None)
        """
        if len(points_3d) == 0:
            return points_3d

        # 1. 变换到相机坐标系
        if camera_pose is not None:
            # P_cam = T_world2cam @ P_world
            # 假设输入 pose 是 Camera -> World (c2w)，需要求逆
            # 通常 mast3r 输出的是 c2w
            c2w = camera_pose
            w2c = np.linalg.inv(c2w)

            # 齐次变换
            pts_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
            pts_cam = (pts_h @ w2c.T)[:, :3]
        else:
            pts_cam = points_3d

        # 2. 投影到 2D 平面
        # u = fx * x / z + cx
        # v = fy * y / z + cy
        fx, fy = camera_intrinsics['fl_x'], camera_intrinsics['fl_y']
        cx, cy = camera_intrinsics['cx'], camera_intrinsics['cy']

        # 避免除以 0 (z=0)
        z = pts_cam[:, 2]
        valid_z = z > 0.1 # 只保留相机前方的点

        u = (pts_cam[:, 0] * fx / z) + cx
        v = (pts_cam[:, 1] * fy / z) + cy

        u = np.round(u).astype(int)
        v = np.round(v).astype(int)

        H, W = mask.shape

        # 3. 检查边界和 Mask 值
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & valid_z

        # 筛选索引
        valid_indices = np.where(in_bounds)[0]

        # 进一步检查 Mask 值
        # 注意 mask[v, u] (Row, Col)
        in_mask = mask[v[valid_indices], u[valid_indices]] > 0

        final_indices = valid_indices[in_mask]

        return points_3d[final_indices]

    def optimize(self,
                 target_cloud: np.ndarray,
                 base_pose_init: Optional[np.ndarray] = None,
                 initial_q: Optional[np.ndarray] = None,
                 iterations: int = 150,
                 lr: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行梯度下降优化
        """
        # 准备数据
        target_tensor = torch.from_numpy(target_cloud).float().to(self.device)

        # 变量初始化
        # 关节角度: 使用 initial_q 或 0，并设为可导
        start_q = torch.zeros(self.n_dof, device=self.device)
        if initial_q is not None:
            start_q = torch.tensor(initial_q, device=self.device).float()

        q = start_q.clone().detach().requires_grad_(True)

        # 基座位姿 (目前固定，未来可优化 SE3)
        if base_pose_init is not None:
            T_base = torch.from_numpy(base_pose_init).float().to(self.device)
        else:
            T_base = torch.eye(4, device=self.device)

        optimizer = torch.optim.Adam([q], lr=lr)

        # 学习率衰减 (有助于后期稳定)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        logger.info(f"Starting kinematic optimization loop ({iterations} iters)...")

        best_loss = float('inf')
        best_q = start_q.detach().cpu().numpy()

        for i in range(iterations):
            optimizer.zero_grad()

            # --- Differentiable FK ---
            transforms = self.chain.forward_kinematics(q)

            # --- Assemble Robot Cloud ---
            reconstructed_points = []
            for link_name, link_pts in self.link_templates.items():
                if link_name in transforms:
                    trans = transforms[link_name].get_matrix() # [1, 4, 4]
                    T_world = T_base @ trans[0]

                    R = T_world[:3, :3]
                    t = T_world[:3, 3]

                    # (N, 3) @ R.T + t
                    transformed_pts = torch.matmul(link_pts, R.t()) + t
                    reconstructed_points.append(transformed_pts)

            if not reconstructed_points:
                logger.warning("No links matched! Check URDF link names vs kinematic chain.")
                break

            full_robot_cloud = torch.cat(reconstructed_points, dim=0)

            # --- Loss ---
            loss = self.loss_fn(full_robot_cloud, target_tensor)

            # 正则化：惩罚大幅度偏离 0 位 (可选，视机器人构型而定)
            # loss += 0.01 * torch.mean(q**2)

            loss.backward()
            optimizer.step()
            scheduler.step()

            # 记录最佳结果
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_q = q.detach().cpu().numpy()

            if i % 25 == 0:
                logger.info(f"Iter {i:03d}: Loss={loss.item():.5f} | lr={scheduler.get_last_lr()[0]:.4f}")

        logger.info(f"Optimization finished. Final Loss: {best_loss:.5f}")
        return best_q, T_base.cpu().numpy()
