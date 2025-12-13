#!/usr/bin/env python3
"""
主程序入口 - Real-to-Sim-to-Real 流水线
最小化实现版本
"""
import hydra
from semiff.core.io import load_video_frames
from semiff.core.logger import get_logger

logger = get_logger(__name__)


@hydra.main(config_path="src/semiff/config", config_name="defaults", version_base=None)
def main(cfg):
    """
    主流水线执行函数

    Args:
        cfg: Hydra配置对象
    """
    logger.info("🚀 Starting Real-to-Sim-to-Real Pipeline...")

    # 1. 加载视频数据
    logger.info(">>> Loading video data...")
    frames, metadata = load_video_frames(cfg.data.video_path)
    logger.info(f"Loaded {len(frames)} frames from video")

    # TODO: 实现完整的流水线
    logger.warning("Pipeline implementation is minimal - core modules need to be implemented")

    logger.info("✅ Minimal pipeline structure ready!")


if __name__ == "__main__":
    main()