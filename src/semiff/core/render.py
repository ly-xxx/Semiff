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
            self.ctx = dr.RasterizeCGLContext(device=device)
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

    def render(self, vertices, faces, K):
        """
        可微分渲染

        Args:
            vertices: (B, N, 3) World Space 顶点
            faces: (M, 3) Int 面索引
            K: (B, 3, 3) 内参矩阵

        Returns:
            (B, H, W) Alpha Mask
        """
        if self.ctx is None:
            # Fallback: 返回零张量但保持梯度
            return torch.zeros((vertices.shape[0], self.H, self.W), device=self.device, requires_grad=True)

        B = vertices.shape[0]

        # 1. MVP Transform (World -> Clip)
        proj = self.build_projection_matrix(K[0]).unsqueeze(0).repeat(B, 1, 1)

        # 齐次坐标
        ones = torch.ones((*vertices.shape[:2], 1), device=self.device)
        v_homo = torch.cat([vertices, ones], dim=-1)

        # 投影变换: v_clip = v_homo @ P.T
        v_clip = torch.bmm(v_homo, proj.transpose(1, 2))

        # 2. 光栅化
        rast, _ = self.dr.rasterize(self.ctx, v_clip, faces, resolution=[self.H, self.W])

        # 3. 抗锯齿插值 (提取 Alpha)
        v_colors = torch.ones_like(vertices)
        mask, _ = self.dr.interpolate(v_colors, rast, faces)
        mask = self.dr.antialias(mask, rast, v_clip, faces)

        return mask[..., 0]  # 返回单通道 alpha
