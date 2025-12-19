"""
Robot Point Cloud Sampler using Sapien
Adapted from real2sim-eval project
"""

import numpy as np
import sapien.core as sapien
from pathlib import Path
from typing import Optional, Union
import torch


class RobotPcSampler:
    """Generate point clouds from robot URDF using Sapien"""

    def __init__(self, urdf_path: Union[str, Path], device: str = "cpu"):
        """
        Initialize robot sampler with URDF

        Args:
            urdf_path: Path to URDF file
            device: Device for computation
        """
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        # Initialize Sapien engine
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer(offscreen_only=True)
        self.engine.set_renderer(self.renderer)

        # Create scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(1 / 240.0)

        # Load robot
        self.scene.load_from_urdf(str(self.urdf_path))
        self.robot = self.scene.get_articulations()[0]  # Assume single robot

        self.device = device

    def compute_robot_pcd(self, qpos: np.ndarray, num_pts: int = 5000) -> np.ndarray:
        """
        Generate point cloud for robot at given joint configuration

        Args:
            qpos: Joint positions
            num_pts: Number of points to sample

        Returns:
            Point cloud as numpy array [N, 3]
        """
        # Set joint positions
        self.robot.set_qpos(qpos)

        # Get all links
        links = self.robot.get_links()

        all_points = []
        for link in links:
            # Skip fixed links or very small links
            if link.get_name() in ['world', 'base_link']:
                continue

            # Get collision shapes
            collision_shapes = link.get_collision_shapes()
            if not collision_shapes:
                continue

            for shape in collision_shapes:
                # Sample points from collision geometry
                # This is a simplified implementation
                # In real implementation, would use proper sampling from geometry
                try:
                    # Get geometry type and sample points accordingly
                    geometry = shape.geometry
                    if hasattr(geometry, 'half_lengths'):  # Box
                        half_lengths = geometry.half_lengths
                        # Sample from box surface
                        pts = self._sample_box_surface(half_lengths, num_pts // len(collision_shapes))
                    elif hasattr(geometry, 'radius'):  # Cylinder/Sphere
                        radius = geometry.radius
                        if hasattr(geometry, 'half_length'):  # Cylinder
                            half_length = geometry.half_length
                            pts = self._sample_cylinder_surface(radius, half_length, num_pts // len(collision_shapes))
                        else:  # Sphere
                            pts = self._sample_sphere_surface(radius, num_pts // len(collision_shapes))
                    else:
                        continue

                    # Transform points to world coordinates
                    pose = link.get_pose()
                    pts_world = pose.transform_points(pts)
                    all_points.append(pts_world)

                except Exception as e:
                    print(f"Warning: Failed to sample from {link.get_name()}: {e}")
                    continue

        if not all_points:
            raise RuntimeError("No points could be sampled from robot")

        # Combine all points
        combined_points = np.concatenate(all_points, axis=0)

        # Subsample to desired number of points
        if len(combined_points) > num_pts:
            indices = np.random.choice(len(combined_points), num_pts, replace=False)
            combined_points = combined_points[indices]

        return combined_points

    def _sample_box_surface(self, half_lengths: np.ndarray, num_pts: int) -> np.ndarray:
        """Sample points from box surface"""
        hx, hy, hz = half_lengths

        # Sample from each face
        points_per_face = num_pts // 6
        points = []

        # Front/back faces (x = ±hx)
        for x in [-hx, hx]:
            y = np.random.uniform(-hy, hy, points_per_face)
            z = np.random.uniform(-hz, hz, points_per_face)
            face_points = np.column_stack([np.full(points_per_face, x), y, z])
            points.append(face_points)

        # Left/right faces (y = ±hy)
        for y in [-hy, hy]:
            x = np.random.uniform(-hx, hx, points_per_face)
            z = np.random.uniform(-hz, hz, points_per_face)
            face_points = np.column_stack([x, np.full(points_per_face, y), z])
            points.append(face_points)

        # Top/bottom faces (z = ±hz)
        for z in [-hz, hz]:
            x = np.random.uniform(-hx, hx, points_per_face)
            y = np.random.uniform(-hy, hy, points_per_face)
            face_points = np.column_stack([x, y, np.full(points_per_face, z)])
            points.append(face_points)

        return np.concatenate(points, axis=0)

    def _sample_cylinder_surface(self, radius: float, half_length: float, num_pts: int) -> np.ndarray:
        """Sample points from cylinder surface"""
        # Sample angles
        theta = np.random.uniform(0, 2*np.pi, num_pts)

        # Sample heights
        z = np.random.uniform(-half_length, half_length, num_pts)

        # Convert to cartesian
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        return np.column_stack([x, y, z])

    def _sample_sphere_surface(self, radius: float, num_pts: int) -> np.ndarray:
        """Sample points from sphere surface"""
        # Use spherical coordinates
        phi = np.random.uniform(0, np.pi, num_pts)
        theta = np.random.uniform(0, 2*np.pi, num_pts)

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        return np.column_stack([x, y, z])

    def __del__(self):
        """Cleanup Sapien resources"""
        if hasattr(self, 'scene'):
            self.scene = None
        if hasattr(self, 'engine'):
            self.engine = None