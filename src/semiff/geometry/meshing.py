"""
Meshing Module: Point Cloud Cleaning & Reconstruction
负责将稀疏/半稠密点云转换为水密网格
"""

import numpy as np
import open3d as o3d
import trimesh
from pathlib import Path
from typing import Optional, Tuple

from ..core.logger import get_logger

logger = get_logger(__name__)

class Mesher:
    def __init__(self, density_threshold_percentile: float = 5.0):
        self.density_threshold_percentile = density_threshold_percentile

    def clean_cloud(self, points: np.ndarray, mask_indices: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
        """
        点云清洗：Mask过滤 + 统计学去噪
        """
        # 1. Mask 过滤
        if mask_indices is not None:
            # 确保索引在有效范围内
            valid_indices = mask_indices[mask_indices < len(points)]
            points = points[valid_indices]

        if len(points) == 0:
            raise ValueError("Point cloud is empty after masking!")

        # 2. 转换为 Open3D 对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 3. 统计学去噪 (Statistical Outlier Removal)
        # 移除离群点，这对泊松重建至关重要
        logger.info(f"Cleaning cloud with {len(points)} points...")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_clean = pcd.select_by_index(ind)

        logger.info(f"Removed {len(points) - len(pcd_clean.points)} outlier points.")
        return pcd_clean

    def reconstruct_mesh(self, pcd: o3d.geometry.PointCloud, depth: int = 9) -> trimesh.Trimesh:
        """
        泊松重建 (Poisson Surface Reconstruction)
        """
        logger.info("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # 调整法向量方向（假设物体中心在点云中心）
        pcd.orient_normals_towards_camera_location(pcd.get_center() + np.array([0, 0, 1.0]))

        logger.info(f"Running Poisson Reconstruction (depth={depth})...")
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=0, scale=1.1, linear_fit=False
        )

        # 裁剪低密度区域 (去除泊松重建产生的"气泡"伪影)
        densities = np.asarray(densities)
        density_threshold = np.percentile(densities, self.density_threshold_percentile)
        idxs_to_remove = densities < density_threshold
        mesh_o3d.remove_vertices_by_mask(idxs_to_remove)

        # 转换为 Trimesh 对象以便后续处理
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

        # 简单的修复
        trimesh.repair.fill_holes(mesh)

        logger.info(f"Mesh reconstructed: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces.")
        return mesh

    def run(self, cloud: np.ndarray, output_path: str) -> str:
        """
        执行完整重建流程
        """
        # 假设输入的 cloud 已经是通过 SAM2 Mask 筛选过的物体点云
        pcd = self.clean_cloud(cloud)
        mesh = self.reconstruct_mesh(pcd)

        # 保存
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_path))
        logger.info(f"Saved reconstruction to {output_path}")
        return str(output_path)