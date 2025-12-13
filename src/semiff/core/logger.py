"""
统一日志管理模块
使用 Rich 提供美观的控制台输出和进度条
"""

import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn


class Logger:
    """统一日志管理器"""

    def __init__(self, name: str = "semiff", level: str = "INFO", log_file: Optional[str] = None):
        self.name = name
        self.level = getattr(logging, level.upper())
        self.log_file = Path(log_file) if log_file else None
        self.console = Console()

        # 创建 logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)

        # 移除现有的处理器
        self.logger.handlers.clear()

        # 添加 Rich 处理器到控制台
        rich_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True
        )
        rich_handler.setLevel(self.level)
        self.logger.addHandler(rich_handler)

        # 如果指定了日志文件，添加文件处理器
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(self.level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """获取配置好的 logger 实例"""
        return self.logger

    def create_progress(self, description: str = "Processing..."):
        """创建进度条"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        )


# 全局日志实例
_global_logger = None

def get_logger(name: str = "semiff", level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """获取全局日志器"""
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger(name, level, log_file)
    return _global_logger.get_logger()

def create_progress(description: str = "Processing..."):
    """创建进度条"""
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger()
    return _global_logger.create_progress(description)
