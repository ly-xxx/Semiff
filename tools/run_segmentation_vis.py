# tools/run_segmentation_vis.py

import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import os

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from semiff.perception.sam2_wrapper import SAM2Wrapper
from semiff.core.logger import get_logger

logger = get_logger("vis_tool")

def is_headless():
    """检测是否在无头环境中运行"""
    return os.environ.get('DISPLAY', '') == '' or not os.environ.get('DISPLAY')

@hydra.main(config_path="../src/semiff/config", config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    logger.info("🎥 Starting Rapid Visualization Pipeline...")
    logger.info(f"Video path: {cfg.data.video_path}")
    logger.info(f"Output dir: {cfg.data.output_dir}")
    logger.info(f"Interactive mode: {cfg.pipeline.interactive_mode}")

    # 检查是否在无头环境中
    headless = is_headless()
    logger.info(f"Headless environment detected: {headless}")

    if headless:
        logger.info("Running in headless mode - will use automatic center point selection")
        cfg.pipeline.interactive_mode = False
    else:
        # 强制开启交互模式
        cfg.pipeline.interactive_mode = True
        logger.info("Forced interactive mode to True")

    video_path = str(cfg.data.video_path)
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 运行 SAM 2 分割
    logger.info("Step 1: Running Segmentation...")
    print("Creating SAM2Wrapper...")
    sam2 = SAM2Wrapper(cfg)
    print("SAM2Wrapper created")

    # 这里 scene_cloud 传 None，强制触发手动点击
    print("Running SAM2...")
    result = sam2.run(video_path, output_dir, scene_cloud=None)
    print(f"SAM2 result: {result}")
    mask_dir = result['object_masks']
    print(f"Mask directory: {mask_dir}")

    # 2. 生成可视化视频
    logger.info("Step 2: Rendering Visualization Video...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 两个输出视频：一个是透明背景(绿幕)，一个是高亮叠加
    out_overlay_path = output_dir / "vis_overlay.mp4"
    out_green_path = output_dir / "vis_green_screen.mp4"

    writer_overlay = cv2.VideoWriter(str(out_overlay_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    writer_green = cv2.VideoWriter(str(out_green_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 加载对应的 mask
        mask_path = mask_dir / f"{frame_idx:05d}.npz"
        if not mask_path.exists():
            break

        mask = np.load(mask_path)['mask'] # (H, W) boolean
        mask_uint8 = (mask * 255).astype(np.uint8)

        # === 效果 1: 红色半透明叠加 (Overlay) ===
        # 创建红色遮罩
        red_mask = np.zeros_like(frame)
        red_mask[:, :, 2] = 255 # Red channel

        # 混合：原图 + 红色遮罩 (只在mask区域)
        overlay = frame.copy()
        # 仅在 mask 为 True 的地方混合
        overlay[mask] = cv2.addWeighted(frame[mask], 0.5, red_mask[mask], 0.5, 0)
        # 画轮廓
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

        writer_overlay.write(overlay)

        # === 效果 2: 绿幕分离 (Green Screen) ===
        # 背景设为绿色 (0, 255, 0)
        green_bg = np.zeros_like(frame)
        green_bg[:] = (0, 255, 0)

        # 前景扣像
        foreground = frame.copy()
        foreground[~mask] = green_bg[~mask]

        writer_green.write(foreground)

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"Rendering frame {frame_idx}...", end='\r')

    cap.release()
    writer_overlay.release()
    writer_green.release()

    logger.info("✅ Visualization saved to:")
    logger.info(f"   - {out_overlay_path} (用于展示分割准确度)")
    logger.info(f"   - {out_green_path} (用于展示物块分离效果)")

if __name__ == "__main__":
    main()
