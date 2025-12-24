"""
智能工作区管理器
负责自动发现和管理基于时间戳的工作区，支持断点续传和自动寻路
"""

import os
import logging
from pathlib import Path
from omegaconf import OmegaConf

logger = logging.getLogger("WORKSPACE")


class WorkspaceManager:
    """智能工作区管理器"""

    def __init__(self, config_path="configs/base_config.yaml"):
        self.raw_conf = OmegaConf.load(config_path)
        # 获取 outputs/ 根目录 (workspace 字段的父目录)
        self.base_output_dir = Path(self.raw_conf.pipeline.workspace).parent

    def get_latest_workspace(self, required_files=None):
        """
        获取按时间戳排序的最新工作区。
        如果指定了 required_files，则只返回包含这些文件的工作区。

        Args:
            required_files: 需要包含的文件列表，如果为 None 返回最新目录

        Returns:
            找到的工作区 Path 对象，如果没找到返回 None
        """
        if not self.base_output_dir.exists():
            return None

        # 1. 获取所有子目录，按修改时间倒序排列
        subdirs = sorted(
            [d for d in self.base_output_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if not subdirs:
            return None

        # 2. 筛选包含必要文件的目录
        for d in subdirs:
            if not required_files:
                return d

            # 检查必要文件是否存在
            missing = [f for f in required_files if not (d / f).exists()]
            if not missing:
                return d

        return None

    def resolve(self, mode="auto", required_input_files=None):
        """
        核心解析逻辑:
        - mode="new": 强制创建新目录 (run.py 默认行为)
        - mode="resume": 强制使用最新的目录
        - mode="auto": 如果是独立运行脚本，尝试找最新的；找不到则新建。

        Args:
            mode: 解析模式
            required_input_files: 需要的输入文件列表

        Returns:
            解析后的工作区路径
        """
        from datetime import datetime

        # 情况 A: 强制新建 (通常是 run.py 的第一次启动)
        if mode == "new":
            # 生成带时间戳的目录名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ws = Path(f"outputs/{timestamp}")
            ws.mkdir(parents=True, exist_ok=True)
            logger.info(f"🆕 Created new workspace: {ws}")
            return ws

        # 情况 B: 尝试恢复/查找上下文
        latest_ws = self.get_latest_workspace(required_files=required_input_files)

        if latest_ws:
            logger.info(f"🔄 Auto-selected latest workspace: {latest_ws}")
            return latest_ws

        # 如果找不到，且模式是 resume，则报错
        if mode == "resume":
            required_str = ", ".join(required_input_files) if required_input_files else "any files"
            raise FileNotFoundError(f"❌ No valid workspace found containing: {required_str}")

        # 如果是 auto 但没找到旧的，就新建
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ws = Path(f"outputs/{timestamp}")
        ws.mkdir(parents=True, exist_ok=True)
        logger.info(f"🆕 No previous history found. Created new: {ws}")
        return ws

    def list_workspaces(self, limit=10):
        """
        列出最近的工作区，用于调试

        Args:
            limit: 最多显示的数量

        Returns:
            工作区信息列表
        """
        if not self.base_output_dir.exists():
            return []

        subdirs = sorted(
            [d for d in self.base_output_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        result = []
        for i, d in enumerate(subdirs[:limit]):
            files = [f.name for f in d.iterdir() if f.is_file()]
            result.append({
                'path': d,
                'mtime': d.stat().st_mtime,
                'files': files
            })

        return result
