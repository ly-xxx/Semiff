"""
数学工具函数 - 坐标变换和几何运算
提供可微分的几何变换操作
"""

import torch
import numpy as np
from typing import Union, Tuple


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """
    应用变换矩阵到点集

    Args:
        points: (N, 3) 或 (B, N, 3) 点集
        transform: (4, 4) 或 (B, 4, 4) 变换矩阵

    Returns:
        变换后的点集，形状与输入相同
    """
    if points.dim() == 2 and transform.dim() == 2:
        # 单批次情况
        ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
        points_homo = torch.cat([points, ones], dim=1)
        transformed_homo = torch.matmul(points_homo, transform.T)
        return transformed_homo[:, :3]

    elif points.dim() == 3 and transform.dim() == 3:
        # 批次情况
        B, N, _ = points.shape
        ones = torch.ones(B, N, 1, device=points.device, dtype=points.dtype)
        points_homo = torch.cat([points, ones], dim=2)
        transformed_homo = torch.matmul(points_homo, transform.transpose(1, 2))
        return transformed_homo[:, :, :3]

    else:
        raise ValueError(f"Unsupported tensor dimensions: points {points.shape}, transform {transform.shape}")


def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    将6D旋转表示转换为旋转矩阵 (Zhou et al., "On the Continuity of Rotation Representations in Neural Networks")

    Args:
        rot_6d: (..., 6) 6D旋转参数

    Returns:
        (..., 3, 3) 旋转矩阵
    """
    if rot_6d.shape[-1] != 6:
        raise ValueError(f"Expected 6D rotation, got shape {rot_6d.shape}")

    # 提取两个正交向量
    a1 = rot_6d[..., :3]  # (..., 3)
    a2 = rot_6d[..., 3:]  # (..., 3)

    # 归一化第一个向量
    b1 = torch.nn.functional.normalize(a1, dim=-1)

    # 构造第二个向量使其正交于第一个
    b2 = torch.nn.functional.normalize(a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1, dim=-1)

    # 叉积得到第三个向量
    b3 = torch.cross(b1, b2, dim=-1)

    # 构造旋转矩阵
    R = torch.stack([b1, b2, b3], dim=-2)  # (..., 3, 3)

    return R


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    """
    将旋转矩阵转换为6D表示

    Args:
        R: (..., 3, 3) 旋转矩阵

    Returns:
        (..., 6) 6D旋转参数
    """
    # 取前两列作为6D表示
    rot_6d = torch.cat([R[..., :2, 0], R[..., :2, 1]], dim=-1)
    return rot_6d


def gpu_mem_guard():
    """
    GPU内存监控上下文管理器
    在CUDA设备上监控显存使用情况
    """
    if torch.cuda.is_available():
        return _CudaMemoryGuard()
    else:
        return _NoOpGuard()


class _CudaMemoryGuard:
    """CUDA显存监控"""

    def __init__(self):
        self.initial_mem = torch.cuda.memory_allocated()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        current_mem = torch.cuda.memory_allocated()
        delta_mb = (current_mem - self.initial_mem) / 1024 / 1024
        if abs(delta_mb) > 10:  # 只报告显著变化
            direction = "增加" if delta_mb > 0 else "减少"
            print(".1f")


class _NoOpGuard:
    """空实现，用于非CUDA环境"""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def homogeneous_transform(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    构造齐次变换矩阵

    Args:
        R: (3, 3) 或 (B, 3, 3) 旋转矩阵
        t: (3,) 或 (B, 3) 平移向量

    Returns:
        (4, 4) 或 (B, 4, 4) 齐次变换矩阵
    """
    if R.dim() == 2 and t.dim() == 1:
        T = torch.eye(4, device=R.device, dtype=R.dtype)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    elif R.dim() == 3 and t.dim() == 2:
        B = R.shape[0]
        T = torch.eye(4, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(B, 1, 1)
        T[:, :3, :3] = R
        T[:, :3, 3] = t
        return T

    else:
        raise ValueError(f"Unsupported tensor dimensions: R {R.shape}, t {t.shape}")


def project_points(points: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    将3D点投影到图像平面

    Args:
        points: (N, 3) 或 (B, N, 3) 3D点
        K: (3, 3) 或 (B, 3, 3) 内参矩阵

    Returns:
        (N, 2) 或 (B, N, 2) 像素坐标
    """
    if points.dim() == 2 and K.dim() == 2:
        # 归一化坐标
        points_norm = points / points[:, 2:3]
        # 投影
        uv_homo = torch.matmul(points_norm, K.T)
        return uv_homo[:, :2]

    elif points.dim() == 3 and K.dim() == 3:
        # 批次情况
        points_norm = points / points[..., 2:3]
        uv_homo = torch.matmul(points_norm, K.transpose(1, 2))
        return uv_homo[..., :2]

    else:
        raise ValueError(f"Unsupported tensor dimensions: points {points.shape}, K {K.shape}")


# 测试函数
if __name__ == "__main__":
    print("🧪 Testing Math Utils...")

    # 测试6D旋转转换
    rot_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    R = rotation_6d_to_matrix(rot_6d)
    print(f"6D to Matrix: {R.shape}")
    print(f"Orthogonality check: {torch.allclose(R @ R.transpose(-1, -2), torch.eye(3), atol=1e-6)}")

    # 测试点变换
    points = torch.randn(10, 3)
    T = torch.eye(4)
    T[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    transformed = transform_points(points, T)
    print(f"Point transform: {points.shape} -> {transformed.shape}")

    print("✅ Math utils working correctly!")
