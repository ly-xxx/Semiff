"""
可微渲染器模块 - 封装 nvdiffrast 光栅化逻辑
提供统一的渲染接口，支持梯度传播
"""

import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)

class DifferentiableRasterizer:
    """可微分光栅化渲染器，基于 nvdiffrast"""

    def __init__(self, H, W, device='cuda'):
        self.H, self.W = H, W
        self.device = device
        self.ctx = None

        try:
            import nvdiffrast.torch as dr
            self.dr = dr
            self.ctx = dr.RasterizeCudaContext(device=device)
            logger.info(f"🟢 nvdiffrast initialized on {device}")
        except ImportError:
            logger.warning("🟡 nvdiffrast not found. Falling back to CPU Mock (No Gradients!).")
            self.ctx = None

    def build_projection_matrix(self, K, near=0.1, far=100.0):
        """构建 OpenGL 投影矩阵 (NDC)"""
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        proj = torch.zeros((4, 4), device=self.device)
        proj[0, 0] = 2 * fx / self.W
        proj[1, 1] = -2 * fy / self.H  # Flip Y for OpenGL
        proj[0, 2] = (self.W - 2 * cx) / self.W
        proj[1, 2] = (self.H - 2 * cy) / self.H
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -(2 * far * near) / (far - near)
        proj[3, 2] = -1.0
        return proj

    def render(self, vertices, faces, K, cam_poses=None):
        """
        可微分渲染

        Args:
            vertices: (B, N, 3) World Space 顶点
            faces: (M, 3) Int 面索引
            K: (B, 3, 3) 内参矩阵
            cam_poses: (B, 4, 4) World-to-Camera 变换矩阵，如果为None则假设顶点已在相机坐标系

        Returns:
            (B, H, W) Alpha Mask
        """
        if self.ctx is None:
            # 这里直接抛错，确保代码走到 nvdiffrast 的逻辑里
            raise RuntimeError("❌ nvdiffrast context is None! Check installation.")

        B = vertices.shape[0]

        # 1. MVP Transform (World -> Clip)
        proj = self.build_projection_matrix(K[0]).unsqueeze(0).repeat(B, 1, 1)

        # 如果提供了相机位姿，先变换到相机坐标系
        if cam_poses is not None:
            # World -> Camera transformation
            R_cam = cam_poses[:, :3, :3]  # [B, 3, 3]
            t_cam = cam_poses[:, :3, 3].unsqueeze(1)  # [B, 1, 3]

            # Transform vertices to camera space: v_cam = R_cam @ v_world.T + t_cam
            v_cam = torch.bmm(vertices, R_cam.transpose(1, 2)) + t_cam
        else:
            v_cam = vertices

        # 齐次坐标
        ones = torch.ones((*v_cam.shape[:2], 1), device=self.device)
        v_homo = torch.cat([v_cam, ones], dim=-1)

        # 投影变换: v_clip = v_homo @ P.T
        v_clip = torch.bmm(v_homo, proj.transpose(1, 2))

        # 2. 光栅化
        rast, _ = self.dr.rasterize(self.ctx, v_clip, faces, resolution=[self.H, self.W])

        # 3. 抗锯齿插值 (提取 Alpha)
        v_colors = torch.ones_like(vertices)
        mask, _ = self.dr.interpolate(v_colors, rast, faces)
        mask = self.dr.antialias(mask, rast, v_clip, faces)

        mask_final = mask[..., 0]  # 返回单通道 alpha

        # Debug: 检查mask的统计信息
        if torch.rand(1) < 0.01:  # 只在1%的迭代中打印，避免输出太多
            logger.info(f"🎨 Render Debug - Mask stats: min={mask_final.min():.4f}, max={mask_final.max():.4f}, mean={mask_final.mean():.4f}")

        return mask_final
