"""
Decomposition Module: Approximate Convex Decomposition (CoACD)
负责生成物理碰撞体
"""

import trimesh
import numpy as np
import coacd
from pathlib import Path
from typing import List, Optional

from ..core.logger import get_logger

logger = get_logger(__name__)

class ColliderBuilder:
    def __init__(self, threshold: float = 0.05, max_convex_hull: int = 16):
        """
        Args:
            threshold: 误差阈值 (越小细节越好，但凸包越多)
            max_convex_hull: 最大凸包数量 (物理性能权衡)
        """
        self.threshold = threshold
        self.max_convex_hull = max_convex_hull

    def decompose(self, mesh_path: str, output_path: str) -> str:
        """
        运行 CoACD 分解
        """
        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        logger.info(f"Loading mesh for decomposition: {mesh_path.name}")
        mesh = trimesh.load(mesh_path, force='mesh')

        logger.info("Running CoACD...")
        # CoACD 核心调用
        # 注意: coacd.run_coacd 返回的是 (vertices, faces) 的列表
        mesh_parts = coacd.run_coacd(
            mesh,
            threshold=self.threshold,
            max_convex_hull=self.max_convex_hull
        )

        logger.info(f"Decomposed into {len(mesh_parts)} convex hulls.")

        # 将所有凸包合并为一个 OBJ 文件 (使用 sub-objects)
        # 或者保存为单个 Trimesh Scene
        scene = trimesh.Scene()

        for i, (verts, faces) in enumerate(mesh_parts):
            part = trimesh.Trimesh(vertices=verts, faces=faces)
            # 给每个部分一个独立的名字，URDF 可以引用
            scene.add_geometry(part, node_name=f"hull_{i}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 导出为 OBJ (支持多物体)
        scene.export(str(output_path))
        logger.info(f"Saved collision mesh to {output_path}")

        return str(output_path)

    def calculate_inertial(self, mesh_path: str, mass: float = 1.0) -> dict:
        """
        计算惯性属性 (Sim 需要)
        """
        mesh = trimesh.load(mesh_path, force='mesh')
        mesh.density = mass / mesh.volume
        return {
            'mass': mesh.mass,
            'center_of_mass': mesh.center_mass,
            'inertia': mesh.moment_inertia
        }