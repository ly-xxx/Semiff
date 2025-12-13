"""
SAM 2 封装模块
将视频转换为物体和机器人掩码
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
from PIL import Image
import torch

from ..core.logger import get_logger, create_progress

logger = get_logger(__name__)


class SAM2Wrapper:
    """SAM 2 推理封装器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 SAM 2 推理器

        Args:
            config: 配置字典
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.predictor = None
        self.model_cfg = config.get('model_config', 'sam2_hiera_l.yaml')
        self.checkpoint = config.get('checkpoint', 'sam2_hiera_large.pt')

        # 初始化模型
        self._load_model()

    def _load_model(self):
        """加载 SAM 2 模型"""
        try:
            from sam2.build_sam import build_sam2_video_predictor

            logger.info(f"加载 SAM 2 模型: {self.model_cfg}")

            # TODO: 根据实际路径调整
            self.predictor = build_sam2_video_predictor(
                self.model_cfg,
                self.checkpoint,
                device=self.device
            )

            logger.info("SAM 2 模型加载成功")

        except ImportError as e:
            logger.error(f"无法导入 SAM 2: {e}")
            logger.error("请确保已正确安装 SAM 2")
            raise
        except Exception as e:
            logger.error(f"SAM 2 模型加载失败: {e}")
            raise

    def _find_auto_prompt(self, mast3r_cloud: np.ndarray, first_frame: np.ndarray) -> np.ndarray:
        """
        自动提示生成：找到最可能移动的物体作为正向提示

        Args:
            mast3r_cloud: MASt3R 生成的点云
            first_frame: 第一帧图像

        Returns:
            正向提示点坐标 (x, y)
        """
        # TODO: 实现自动提示逻辑
        # 根据文档，这里应该：
        # 1. 从 MASt3R 点云中找到"动得最厉害"的一坨点
        # 2. 投影回第一帧作为 SAM 2 的正向提示点

        logger.warning("自动提示生成尚未实现，使用图像中心作为默认提示")

        # 默认使用图像中心
        h, w = first_frame.shape[:2]
        center_point = np.array([[w // 2, h // 2]])
        return center_point

    def _propagate_masks(self, video_frames: List[np.ndarray],
                        prompt_points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        视频掩码传播

        Args:
            video_frames: 视频帧列表
            prompt_points: 提示点坐标

        Returns:
            掩码字典
        """
        logger.info("开始视频掩码传播")

        # 转换为 SAM 2 期望的格式
        frame_names = [f"frame_{i:06d}" for i in range(len(video_frames))]

        # 初始化推理状态
        inference_state = self.predictor.init_state(video_path=None, images=video_frames)

        # 添加正向提示点
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=prompt_points,
            labels=np.ones(len(prompt_points))  # 正向标签
        )

        # 传播到所有帧
        masks = {}
        for frame_idx, frame_name in enumerate(frame_names):
            if frame_idx == 0:
                # 第一帧已经处理
                mask = (out_mask_logits[0] > 0.0).squeeze().cpu().numpy()
            else:
                # 传播到后续帧
                out_obj_ids, out_mask_logits = self.predictor.propagate_in_video(
                    inference_state, start_frame_idx=frame_idx
                )
                mask = (out_mask_logits[0] > 0.0).squeeze().cpu().numpy()

            masks[frame_name] = mask.astype(np.uint8)

        logger.info(f"掩码传播完成，共处理 {len(masks)} 帧")
        return masks

    def _separate_robot_object_masks(self, masks: Dict[str, np.ndarray],
                                   mast3r_cloud: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        分离机器人和物体掩码

        Args:
            masks: 原始掩码字典
            mast3r_cloud: MASt3R 点云

        Returns:
            机器人掩码和物体掩码
        """
        # TODO: 实现机器人和物体掩码分离
        # 根据文档，这里应该利用机器人运动学来区分

        logger.warning("机器人/物体掩码分离尚未实现，返回相同掩码")

        # 临时实现：假设所有掩码都是物体掩码，没有机器人掩码
        robot_masks = {k: np.zeros_like(v) for k, v in masks.items()}
        object_masks = masks.copy()

        return robot_masks, object_masks

    def run(self, video_frames: List[np.ndarray],
           mast3r_cloud: Optional[np.ndarray] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        执行 SAM 2 推理

        Args:
            video_frames: 视频帧列表
            mast3r_cloud: MASt3R 生成的点云 (用于自动提示)

        Returns:
            掩码结果字典
        """
        logger.info("开始 SAM 2 推理流程")

        with create_progress("SAM 2 推理") as progress:
            # 步骤 1: 自动提示生成
            task1 = progress.add_task("生成自动提示", total=1)
            if mast3r_cloud is not None:
                prompt_points = self._find_auto_prompt(mast3r_cloud, video_frames[0])
            else:
                # 默认提示
                h, w = video_frames[0].shape[:2]
                prompt_points = np.array([[w // 2, h // 2]])
            progress.update(task1, completed=1)

            # 步骤 2: 视频传播
            task2 = progress.add_task("视频掩码传播", total=1)
            raw_masks = self._propagate_masks(video_frames, prompt_points)
            progress.update(task2, completed=1)

            # 步骤 3: 分离机器人和物体
            task3 = progress.add_task("分离掩码", total=1)
            robot_masks, object_masks = self._separate_robot_object_masks(raw_masks, mast3r_cloud)
            progress.update(task3, completed=1)

        result = {
            'robot': robot_masks,
            'object': object_masks,
            'raw': raw_masks
        }

        logger.info("SAM 2 推理完成")
        return result

    def save_masks(self, masks: Dict[str, np.ndarray], output_dir: Path,
                  format: str = 'npz') -> None:
        """
        保存掩码结果

        Args:
            masks: 掩码字典
            output_dir: 输出目录
            format: 保存格式 ('npz', 'png')
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if format == 'npz':
            # 保存为压缩的 NPZ 格式
            for name, mask in masks.items():
                mask_path = output_dir / f"{name}.npz"
                np.savez_compressed(mask_path, mask=mask)
        elif format == 'png':
            # 保存为 PNG 格式
            for name, mask in masks.items():
                mask_path = output_dir / f"{name}.png"
                # 转换为二值图像
                mask_img = (mask * 255).astype(np.uint8)
                cv2.imwrite(str(mask_path), mask_img)
        else:
            raise ValueError(f"不支持的格式: {format}")

        logger.info(f"掩码已保存到: {output_dir} (格式: {format})")


class SAM2Pipeline:
    """SAM 2 推理流水线"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.wrapper = SAM2Wrapper(config)

    def run(self, video_path: str, mast3r_cloud: Optional[np.ndarray] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        运行完整的 SAM 2 流水线

        Args:
            video_path: 视频文件路径
            mast3r_cloud: MASt3R 点云 (可选，用于自动提示)

        Returns:
            掩码结果字典
        """
        from ..core.io import load_video_frames

        logger.info(f"开始处理视频: {video_path}")

        # 加载视频帧
        frames, metadata = load_video_frames(video_path)

        # 运行 SAM 2 推理
        masks = self.wrapper.run(frames, mast3r_cloud)

        # 保存结果
        output_dir = Path(self.config.get('output_dir', 'outputs/sam2'))
        self.wrapper.save_masks(masks['object'], output_dir / 'object_masks')
        self.wrapper.save_masks(masks['robot'], output_dir / 'robot_masks')

        return masks
