"""
Gaussian Splatting Processor
Adapted from real2sim-eval project
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import open3d as o3d


class GSProcessor:
    """Process Gaussian Splatting models for alignment and transformation"""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

    def load(self, ply_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
        """
        Load Gaussian Splatting PLY file

        Args:
            ply_path: Path to .ply file containing Gaussians

        Returns:
            Dictionary containing GS parameters
        """
        ply_path = Path(ply_path)
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        # Load PLY file using Open3D
        pcd = o3d.io.read_point_cloud(str(ply_path))

        # Extract basic parameters
        means3D = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device=self.device)

        # For now, create dummy parameters for other GS attributes
        # In a full implementation, these would be loaded from PLY file
        num_points = len(means3D)

        # Initialize with reasonable defaults
        params = {
            'means3D': means3D,
            'scales': torch.ones((num_points, 3), dtype=torch.float32, device=self.device) * 0.01,
            'rotations': torch.zeros((num_points, 4), dtype=torch.float32, device=self.device),
            'opacities': torch.ones((num_points, 1), dtype=torch.float32, device=self.device),
            'shs': torch.zeros((num_points, 3, 16), dtype=torch.float32, device=self.device),  # 3 channels, 16 coeffs each
        }

        # Set rotations to identity quaternions
        params['rotations'][:, 0] = 1.0

        # Try to load colors if available
        if pcd.has_colors():
            colors = torch.tensor(np.asarray(pcd.colors), dtype=torch.float32, device=self.device)
            # Convert RGB to SH coefficients (simplified - just DC component)
            params['shs'][:, :, 0] = colors

        return params

    def save(self, params: Dict[str, torch.Tensor], output_path: Union[str, Path]):
        """
        Save Gaussian Splatting parameters to PLY file

        Args:
            params: GS parameters dictionary
            output_path: Output PLY file path
        """
        output_path = Path(output_path)

        # Extract means3D and colors for basic PLY
        means3D = params['means3D'].cpu().numpy()

        # Get colors from SH coefficients (simplified - just DC component)
        colors = params['shs'][:, :, 0].cpu().numpy()
        colors = np.clip(colors, 0, 1)  # Ensure valid color range

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(means3D)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save PLY
        o3d.io.write_point_cloud(str(output_path), pcd)

    def translate(self, params: Dict[str, torch.Tensor], translation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Translate Gaussian means

        Args:
            params: GS parameters
            translation: Translation vector [3]

        Returns:
            Updated parameters
        """
        new_params = params.copy()
        new_params['means3D'] = params['means3D'] + translation.to(self.device)
        return new_params

    def rotate(self, params: Dict[str, torch.Tensor], rotation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Rotate Gaussian means

        Args:
            params: GS parameters
            rotation: Rotation matrix [3, 3]

        Returns:
            Updated parameters
        """
        new_params = params.copy()
        new_params['means3D'] = torch.matmul(params['means3D'], rotation.T.to(self.device))
        return new_params