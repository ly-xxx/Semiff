"""
SAM 2 Wrapper: Video Segmentation with Auto-Prompting
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import cv2
import os

from ..core.logger import get_logger

logger = get_logger(__name__)

class SAM2Wrapper:
    def __init__(self, config: Dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint = config.get("checkpoint", "checkpoints/sam2_hiera_large.pt")
        # Use the correct path to SAM 2 config file
        # SAM 2 registers itself as a hydra config module, so use relative path within the module
        self.model_cfg = config.get("model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml")

        # 获取交互模式开关，默认为 False
        pipeline_cfg = config.get("pipeline", {})
        self.interactive_mode = pipeline_cfg.get("interactive_mode", False) if isinstance(pipeline_cfg, dict) else getattr(pipeline_cfg, 'interactive_mode', False)

        self.predictor = self._init_model()

    def _init_model(self):
        try:
            # Ensure SAM 2 hydra config module is initialized
            # We need to temporarily clear GlobalHydra if it's already initialized
            import sam2
            from hydra.core.global_hydra import GlobalHydra
            from hydra import initialize_config_module

            hydra_was_initialized = GlobalHydra.instance().is_initialized()
            if hydra_was_initialized:
                GlobalHydra.instance().clear()

            # Initialize SAM 2 config module
            initialize_config_module("sam2", version_base="1.2")

            from sam2.build_sam import build_sam2_video_predictor
            logger.info("Initializing SAM 2 Video Predictor...")
            logger.info(f"Using checkpoint: {self.checkpoint}")
            logger.info(f"Using config: {self.model_cfg}")
            logger.info(f"Using device: {self.device}")

            # 检查checkpoint文件是否存在
            import os
            if not os.path.exists(self.checkpoint):
                logger.error(f"Checkpoint file not found: {self.checkpoint}")
                logger.info("Please download SAM 2 checkpoints from https://github.com/facebookresearch/sam2")
                return None

            predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint, device=self.device)
            logger.info("SAM 2 initialization successful")
            return predictor
        except ImportError as e:
            logger.error(f"SAM 2 import failed: {e}")
            logger.info("Please install SAM 2: pip install git+https://github.com/facebookresearch/sam2.git")
            return None
        except Exception as e:
            logger.error(f"SAM 2 initialization failed: {e}")
            return None

    def _get_interactive_prompt(self, frame: np.ndarray) -> np.ndarray:
        """
        [新增] 交互式获取提示点
        弹出一个窗口，用户点击物体中心，按空格或回车确认
        如果在无头环境中，自动回退到中心点提示
        """
        # 检测是否在无头环境中
        is_headless = os.environ.get('DISPLAY', '') == '' or not os.environ.get('DISPLAY')

        if is_headless:
            logger.warning(">>> HEADLESS MODE: Skipping interactive selection. Using center point.")
            h, w = frame.shape[:2]
            return np.array([[w // 2, h // 2]], dtype=np.float32)

        logger.info(">>> INTERACTIVE MODE: Please click on the object in the popup window.")
        logger.info("    Controls: [Left Click] Select Point  [Space/Enter] Confirm  [Esc] Skip")

        prompt_points = []
        window_name = "Select Object (SAM 2)"

        # 缩放图像以适应屏幕（如果图像太大）
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        scale = 1.0
        if h > 800 or w > 1200:
            scale = min(800/h, 1200/w)
            display_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        # 鼠标回调函数
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # 记录点击位置（映射回原图坐标）
                real_x, real_y = int(x / scale), int(y / scale)
                prompt_points.append([real_x, real_y])
                logger.info(f"Selected point: ({real_x}, {real_y})")

                # 在显示图上画圈
                cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow(window_name, display_frame)

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        cv2.imshow(window_name, display_frame)

        # 等待按键
        while True:
            key = cv2.waitKey(1) & 0xFF
            # 空格(32) 或 回车(13) 确认
            if key == 32 or key == 13:
                break
            # Esc(27) 退出
            if key == 27:
                logger.warning("Interactive selection skipped.")
                break

        cv2.destroyAllWindows()

        if not prompt_points:
            # 如果没点，回退到中心点
            logger.warning("No points selected. Fallback to center.")
            h, w = frame.shape[:2]
            return np.array([[w // 2, h // 2]], dtype=np.float32)

        # 目前我们只取最后一个点击点（单点提示），如果需要多点可以修改这里
        return np.array([prompt_points[-1]], dtype=np.float32)

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
        print(f"SAM2Wrapper.run called with video_path={video_path}, interactive_mode={self.interactive_mode}")

        if self.predictor is None:
            print("SAM 2 predictor is None")
            raise RuntimeError("SAM 2 not initialized")

        print("Initializing inference state...")
        # 1. 初始化推理状态
        inference_state = self.predictor.init_state(video_path=video_path)
        print("Inference state initialized")

        # 2. 获取第一帧并确定提示点
        # SAM 2 API 通常需要手动加载图像或由 init_state 处理
        # 这里假设 init_state 已经处理了视频加载

        # 读取第一帧用于交互或尺寸获取
        print("Reading first frame...")
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Cannot read first frame from video")

        print(f"First frame shape: {first_frame.shape}")

        # === 核心修改：根据配置选择 Prompt 方式 ===
        print(f"Getting prompts, interactive_mode={self.interactive_mode}")
        if self.interactive_mode:
            points = self._get_interactive_prompt(first_frame)
        else:
            points = self._get_auto_prompt([first_frame], scene_cloud)

        print(f"Selected points: {points}")

        labels = np.array([1] * len(points), dtype=np.int32)

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