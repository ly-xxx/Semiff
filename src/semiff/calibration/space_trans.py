"""
Space Transformation Toolkit
处理刚体变换 (4x4 矩阵) 和点云操作
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Union, Tuple, Optional, List

class RigidTransform:
    """刚体变换封装 (支持缩放)"""

    def __init__(self, matrix: Optional[np.ndarray] = None):
        if matrix is None:
            self.matrix = np.eye(4)
        else:
            self.matrix = np.array(matrix, dtype=np.float64)
            assert self.matrix.shape == (4, 4)

    @property
    def rotation(self) -> np.ndarray:
        """3x3 旋转矩阵"""
        return self.matrix[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        """3D 平移向量"""
        return self.matrix[:3, 3]

    @property
    def scale(self) -> float:
        """提取均匀缩放因子 (假设均匀缩放)"""
        # 利用行列式或列向量长度的平均值
        det = np.linalg.det(self.rotation)
        return np.cbrt(np.abs(det))

    def inverse(self) -> 'RigidTransform':
        """计算逆变换"""
        try:
            inv_mat = np.linalg.inv(self.matrix)
            return RigidTransform(inv_mat)
        except np.linalg.LinAlgError:
            print("Warning: Matrix is singular, cannot invert.")
            return RigidTransform(np.eye(4))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        变换点云
        Args:
            points: [N, 3] 点云
        Returns:
            transformed_points: [N, 3]
        """
        if points.shape[0] == 0:
            return points

        # 齐次坐标变换
        N = points.shape[0]
        # [N, 4] = [x, y, z, 1]
        homo_points = np.hstack([points, np.ones((N, 1))])

        # [N, 4] * [4, 4].T -> [N, 4]
        transformed = homo_points @ self.matrix.T

        return transformed[:, :3]

    @classmethod
    def from_rt(cls, rotation: np.ndarray, translation: np.ndarray, scale: float = 1.0) -> 'RigidTransform':
        """从 R, t, s 构建"""
        mat = np.eye(4)
        mat[:3, :3] = rotation * scale
        mat[:3, 3] = translation
        return cls(mat)

    def __repr__(self):
        return f"RigidTransform(\n{self.matrix}\n)"

def apply_transform_to_cameras(cameras: List[np.ndarray], transform: RigidTransform) -> List[np.ndarray]:
    """
    将变换应用到相机位姿列表
    Pose (C2W) 变换逻辑: T_new = T_align @ T_old
    """
    new_cameras = []
    T_align = transform.matrix
    for pose in cameras:
        new_pose = T_align @ pose
        new_cameras.append(new_pose)
    return new_cameras