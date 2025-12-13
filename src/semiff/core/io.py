"""
输入输出模块
负责视频读取、机器人日志加载等基础IO操作
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image
import pandas as pd

from .logger import get_logger

logger = get_logger(__name__)


class VideoReader:
    """视频读取器"""

    def __init__(self, video_path: Union[str, Path]):
        self.video_path = Path(video_path)
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0

        if not self.video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {self.video_path}")

        self._open_video()

    def _open_video(self):
        """打开视频文件"""
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {self.video_path}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(".1f"
                   f"{self.width}x{self.height}")

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """获取指定帧"""
        if frame_idx < 0 or frame_idx >= self.frame_count:
            logger.warning(f"帧索引超出范围: {frame_idx} (总帧数: {self.frame_count})")
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            # 转换为 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        else:
            logger.warning(f"无法读取帧: {frame_idx}")
            return None

    def get_frames(self, frame_indices: List[int]) -> List[np.ndarray]:
        """批量获取帧"""
        frames = []
        for idx in frame_indices:
            frame = self.get_frame(idx)
            if frame is not None:
                frames.append(frame)
        return frames

    def get_all_frames(self, step: int = 1) -> List[np.ndarray]:
        """获取所有帧，可选择步长采样"""
        frames = []
        for i in range(0, self.frame_count, step):
            frame = self.get_frame(i)
            if frame is not None:
                frames.append(frame)
        return frames

    def close(self):
        """关闭视频"""
        if self.cap:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_exc):
        self.close()


class RobotLogger:
    """机器人日志读取器"""

    def __init__(self, log_path: Union[str, Path]):
        self.log_path = Path(log_path)
        self.data = None
        self.columns = None

        if not self.log_path.exists():
            raise FileNotFoundError(f"机器人日志文件不存在: {self.log_path}")

        self._load_data()

    def _load_data(self):
        """加载机器人日志数据"""
        suffix = self.log_path.suffix.lower()

        try:
            if suffix == '.csv':
                self.data = pd.read_csv(self.log_path)
                self.columns = list(self.data.columns)
            elif suffix == '.json':
                with open(self.log_path, 'r', encoding='utf-8') as f:
                    data_dict = json.load(f)
                # 假设 JSON 是时间序列数据
                self.data = pd.DataFrame(data_dict)
                self.columns = list(self.data.columns)
            elif suffix in ['.pkl', '.pickle']:
                with open(self.log_path, 'rb') as f:
                    self.data = pickle.load(f)
                if isinstance(self.data, dict):
                    self.data = pd.DataFrame(self.data)
                    self.columns = list(self.data.columns)
            else:
                raise ValueError(f"不支持的文件格式: {suffix}")

            logger.info(f"加载机器人日志成功，共 {len(self.data)} 条记录，列: {self.columns}")

        except Exception as e:
            raise RuntimeError(f"加载机器人日志失败: {e}")

    def get_joint_angles(self, timestamp: Optional[float] = None) -> Optional[np.ndarray]:
        """获取指定时间戳的关节角度"""
        if 'timestamp' not in self.columns:
            logger.warning("日志中没有时间戳列")
            return None

        if timestamp is None:
            # 返回第一条记录
            return self._extract_joint_angles(self.data.iloc[0])

        # 找到最接近的时间戳
        timestamps = self.data['timestamp'].values
        idx = np.argmin(np.abs(timestamps - timestamp))
        return self._extract_joint_angles(self.data.iloc[idx])

    def get_trajectory(self, start_time: Optional[float] = None,
                      end_time: Optional[float] = None) -> pd.DataFrame:
        """获取轨迹数据"""
        if 'timestamp' not in self.columns:
            logger.warning("日志中没有时间戳列，返回全部数据")
            return self.data

        mask = np.ones(len(self.data), dtype=bool)
        if start_time is not None:
            mask &= (self.data['timestamp'] >= start_time)
        if end_time is not None:
            mask &= (self.data['timestamp'] <= end_time)

        return self.data[mask].copy()

    def _extract_joint_angles(self, row) -> Optional[np.ndarray]:
        """从数据行中提取关节角度"""
        # 查找关节角度相关的列
        joint_cols = [col for col in self.columns if 'joint' in col.lower() or 'angle' in col.lower()]

        if not joint_cols:
            logger.warning("未找到关节角度列")
            return None

        try:
            angles = row[joint_cols].values.astype(float)
            return angles
        except Exception as e:
            logger.warning(f"提取关节角度失败: {e}")
            return None

    def get_timestamps(self) -> np.ndarray:
        """获取所有时间戳"""
        if 'timestamp' not in self.columns:
            return np.arange(len(self.data))
        return self.data['timestamp'].values


def load_video_frames(video_path: Union[str, Path], frame_indices: Optional[List[int]] = None,
                     step: int = 1) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """便捷函数：加载视频帧"""
    with VideoReader(video_path) as reader:
        if frame_indices is not None:
            frames = reader.get_frames(frame_indices)
        else:
            frames = reader.get_all_frames(step)

        metadata = {
            'frame_count': reader.frame_count,
            'fps': reader.fps,
            'width': reader.width,
            'height': reader.height,
            'loaded_frames': len(frames)
        }

        return frames, metadata


def load_robot_logs(log_path: Union[str, Path]) -> RobotLogger:
    """便捷函数：加载机器人日志"""
    return RobotLogger(log_path)
