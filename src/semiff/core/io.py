"""
输入输出模块
负责视频读取、机器人日志加载与高精度插值
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2
import pandas as pd
from scipy.interpolate import interp1d

from .logger import get_logger

logger = get_logger(__name__)


class VideoReader:
    """视频读取器 (支持上下文管理)"""

    def __init__(self, video_path: Union[str, Path]):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {self.video_path}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

        logger.info(f"Video opened: {self.video_path.name} | {self.width}x{self.height} @ {self.fps:.2f}fps | {self.duration:.2f}s")

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """获取指定帧 (RGB)"""
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def get_frames(self, step: int = 1) -> Tuple[List[np.ndarray], List[float]]:
        """
        批量获取帧
        Returns:
            frames: 帧列表
            timestamps: 对应的时间戳列表 (秒)
        """
        frames = []
        timestamps = []
        for i in range(0, self.frame_count, step):
            frame = self.get_frame(i)
            if frame is not None:
                frames.append(frame)
                timestamps.append(i / self.fps)
        return frames, timestamps

    def close(self):
        if self.cap:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RobotLogger:
    """机器人日志读取器 (支持高精度插值)"""

    def __init__(self, log_path: Union[str, Path]):
        self.log_path = Path(log_path)
        if not self.log_path.exists():
            raise FileNotFoundError(f"日志文件不存在: {self.log_path}")

        self.df = self._load_data()
        self.joint_cols = [c for c in self.df.columns if 'joint' in c.lower() or 'angle' in c.lower()]
        self.timestamp_col = 'timestamp' if 'timestamp' in self.df.columns else None

        if not self.joint_cols:
            raise ValueError("未在日志中找到关节角度列")

        # 构建插值函数
        self.interpolators = {}
        if self.timestamp_col:
            t = self.df[self.timestamp_col].values
            # 处理时间戳可能的重复或乱序
            if not np.all(np.diff(t) > 0):
                logger.warning("日志时间戳非单调递增，正在排序...")
                self.df = self.df.sort_values(by=self.timestamp_col)
                t = self.df[self.timestamp_col].values

            for col in self.joint_cols:
                # 使用线性插值，fill_value="extrapolate" 允许轻微的时间越界
                self.interpolators[col] = interp1d(t, self.df[col].values, kind='linear', fill_value="extrapolate")

        logger.info(f"Robot Log loaded: {len(self.df)} records. Joints: {self.joint_cols}")

    def _load_data(self) -> pd.DataFrame:
        suffix = self.log_path.suffix.lower()
        if suffix == '.csv':
            return pd.read_csv(self.log_path)
        elif suffix == '.json':
            return pd.read_json(self.log_path)
        elif suffix in ['.pkl', '.pickle']:
            return pd.read_pickle(self.log_path)
        else:
            raise ValueError(f"不支持的格式: {suffix}")

    def get_interpolated_joints(self, query_timestamps: List[float]) -> np.ndarray:
        """
        获取查询时间戳对应的关节角度 (线性插值)

        Args:
            query_timestamps: 查询时间点列表 (秒)

        Returns:
            np.ndarray: [N, Num_Joints] 关节角度矩阵
        """
        if not self.interpolators:
            logger.warning("无时间戳列，无法插值。返回前N条记录。")
            return self.df[self.joint_cols].iloc[:len(query_timestamps)].values

        result = []
        for t in query_timestamps:
            row = [self.interpolators[col](t) for col in self.joint_cols]
            result.append(row)
        return np.array(result)


def load_video_frames(video_path: str) -> Tuple[List[np.ndarray], Dict]:
    """快捷入口"""
    with VideoReader(video_path) as reader:
        frames, _ = reader.get_frames(step=1)
        meta = {'width': reader.width, 'height': reader.height, 'fps': reader.fps}
    return frames, meta
