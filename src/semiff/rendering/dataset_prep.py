"""
Dataset Preparation: Nerfstudio Converter with Alpha Masking
负责生成用于背景训练的数据集 (RGB-A)
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from ..core.logger import get_logger
from ..calibration.space_trans import RigidTransform

logger = get_logger(__name__)

class NerfstudioConverter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def _opencv_to_opengl(self, pose: np.ndarray) -> np.ndarray:
        """
        坐标系转换: OpenCV (Right, Down, Forward) -> OpenGL (Right, Up, Back)
        Nerfstudio 使用 OpenGL 约定
        """
        # 翻转 Y 和 Z 轴
        # [1,  0,  0, 0]
        # [0, -1,  0, 0]
        # [0,  0, -1, 0]
        # [0,  0,  0, 1]
        flip_mat = np.diag([1, -1, -1, 1])
        return pose @ flip_mat

    def process(self,
                frames: List[np.ndarray],
                masks_dir: Path,
                poses: List[np.ndarray],
                intrinsics: Dict[str, float]) -> str:
        """
        处理帧并生成 transforms.json

        Args:
            frames: 原始 RGB 帧列表
            masks_dir: 包含 .npz 掩码的目录 (from SAM2)
            poses: 对齐后的相机位姿 (OpenCV convention)
            intrinsics: {'fl_x': ..., 'fl_y': ..., 'cx': ..., 'cy': ..., 'w': ..., 'h': ...}
        """
        logger.info("Preparing Nerfstudio dataset...")

        transforms_data = {
            "fl_x": intrinsics['fl_x'],
            "fl_y": intrinsics['fl_y'],
            "cx": intrinsics['cx'],
            "cy": intrinsics['cy'],
            "w": intrinsics['w'],
            "h": intrinsics['h'],
            "camera_model": "OPENCV",
            "frames": []
        }

        # 获取所有 mask 文件并排序
        mask_files = sorted(list(masks_dir.glob("*.npz")))

        if len(mask_files) != len(frames):
            logger.warning(f"Mismatch: {len(frames)} frames vs {len(mask_files)} masks. Using intersection.")
            # 实际生产中应处理对齐逻辑，这里简化为截断
            limit = min(len(frames), len(mask_files))
            frames = frames[:limit]
            poses = poses[:limit]
            mask_files = mask_files[:limit]

        for idx, (frame, mask_file, pose) in enumerate(tqdm(zip(frames, mask_files, poses), total=len(frames))):
            # 1. 加载 Mask
            # 假设 mask: 1=Object, 0=Background
            try:
                mask_data = np.load(mask_file)['mask']
            except Exception as e:
                logger.error(f"Failed to load mask {mask_file}: {e}")
                continue

            # 2. 生成 RGBA 图像 (Alpha Masking)
            # 背景训练逻辑：我们希望"去除"物体。
            # 所以：Mask=1 (物体) -> Alpha=0 (透明/忽略)
            #      Mask=0 (背景) -> Alpha=255 (保留)
            alpha_channel = ((1.0 - mask_data) * 255).astype(np.uint8)

            # 确保尺寸匹配
            if alpha_channel.shape[:2] != frame.shape[:2]:
                alpha_channel = cv2.resize(alpha_channel, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # 合并通道
            frame_rgba = np.dstack([frame, alpha_channel])

            # 3. 保存图像
            file_name = f"frame_{idx:05d}.png"
            save_path = self.images_dir / file_name
            # OpenCV 使用 BGR
            cv2.imwrite(str(save_path), cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGRA))

            # 4. 处理位姿 (转换到 OpenGL 约定)
            pose_opengl = self._opencv_to_opengl(pose)

            transforms_data["frames"].append({
                "file_path": f"images/{file_name}",
                "transform_matrix": pose_opengl.tolist()
            })

        # 保存 transforms.json
        json_path = self.output_dir / "transforms.json"
        with open(json_path, 'w') as f:
            json.dump(transforms_data, f, indent=4)

        logger.info(f"Dataset prepared at: {self.output_dir}")
        return str(self.output_dir)

def estimate_intrinsics(width: int, height: int, fov_deg: float = 60.0) -> Dict[str, float]:
    """
    估算相机内参 (如果 MASt3R 未提供)
    """
    fov_rad = np.deg2rad(fov_deg)
    focal = 0.5 * height / np.tan(0.5 * fov_rad)
    return {
        "fl_x": focal,
        "fl_y": focal,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "w": width,
        "h": height
    }