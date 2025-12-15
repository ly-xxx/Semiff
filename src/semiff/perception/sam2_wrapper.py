"""
SAM 2 Wrapper: Video Segmentation with Auto-Prompting
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import cv2

from ..core.logger import get_logger

logger = get_logger(__name__)

class SAM2Wrapper:
    def __init__(self, config: Dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint = config.get("checkpoint", "checkpoints/sam2_hiera_large.pt")
        self.model_cfg = config.get("model_cfg", "sam2_hiera_l.yaml")
        self.predictor = self._init_model()

    def _init_model(self):
        try:
            from sam2.build_sam import build_sam2_video_predictor
            logger.info("Initializing SAM 2 Video Predictor...")
            return build_sam2_video_predictor(self.model_cfg, self.checkpoint, device=self.device)
        except ImportError:
            logger.error("SAM 2 not found.")
            return None

    def _get_auto_prompt(self, frames: List[np.ndarray], scene_cloud: Optional[np.ndarray] = None) -> np.ndarray:
        """
        自动化提示生成 Trick
        如果提供了 MASt3R 点云，我们将尝试找到"前景物体"。
        简单的启发式：寻找离相机最近且密度较大的点簇中心，投影回第一帧。
        """
        if scene_cloud is None:
            # Fallback: 中心点提示
            h, w = frames[0].shape[:2]
            return np.array([[w // 2, h // 2]], dtype=np.float32)

        # TODO: 这里应该实现 3D -> 2D 投影逻辑
        # 既然我们还没有对齐的相机参数，我们先假设物体在图像中心区域
        # 工业级实现应该在这里使用 GroundingDINO 或 CLIP 来根据文本 "object" 找到提示点
        logger.info("Using center point as heuristic prompt.")
        h, w = frames[0].shape[:2]
        return np.array([[w // 2, h // 2]], dtype=np.float32)

    def run(self, video_path: str, output_dir: Path, scene_cloud: Optional[np.ndarray] = None) -> Dict[str, Path]:
        """
        运行视频分割

        Returns:
            Path to the saved mask directory
        """
        if self.predictor is None:
            raise RuntimeError("SAM 2 not initialized")

        # 1. 初始化推理状态
        inference_state = self.predictor.init_state(video_path=video_path)

        # 2. 获取第一帧并确定提示点
        # SAM 2 API 通常需要手动加载图像或由 init_state 处理
        # 这里假设 init_state 已经处理了视频加载

        # 临时读取第一帧用于尺寸获取
        cap = cv2.VideoCapture(video_path)
        _, first_frame = cap.read()
        cap.release()

        # 获取正向提示点 (Label=1)
        points = self._get_auto_prompt([first_frame], scene_cloud)
        labels = np.array([1], dtype=np.int32)

        # 3. 添加提示并传播
        logger.info(f"Adding prompt at {points}...")
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
        )

        # 4. 视频传播
        logger.info("Propagating masks through video...")
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            # 存储 mask, 这里简化为取第一个对象的 mask
            mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
            video_segments[out_frame_idx] = mask

        # 5. 保存结果
        save_path = output_dir / "masks"
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {len(video_segments)} masks to {save_path}...")
        for idx, mask in video_segments.items():
            # 保存为压缩 npz 以节省空间
            np.savez_compressed(save_path / f"{idx:05d}.npz", mask=mask)

            # 可选：保存为 PNG 用于调试
            # cv2.imwrite(str(save_path / f"{idx:05d}.png"), (mask * 255).astype(np.uint8))

        return {"object_masks": save_path}