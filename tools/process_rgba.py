import cv2
import numpy as np
import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import sys

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from semiff.core.logger import get_logger

logger = get_logger("rgba_tool")

@hydra.main(config_path="../src/semiff/config", config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    logger.info("🚀 Starting Alpha Injection (Data Solidification)...")

    # 1. 配置路径
    video_path = Path(cfg.data.video_path)
    output_dir = Path(cfg.data.output_dir)

    # 输入目录 (来自 step2_segment.py)
    mask_obj_dir = output_dir / "masks_object"
    mask_rob_dir = output_dir / "masks_robot"

    # 输出目录
    train_data_dir = output_dir / "train_data"
    images_dir = train_data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 检查输入
    if not mask_obj_dir.exists() or not mask_rob_dir.exists():
        logger.error(f"❌ Mask directories not found in {output_dir}")
        logger.error("Please run 'python tools/step2_segment.py' first.")
        return

    # 2. 读取视频信息
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 检查是否有旋转元数据 (与 segmentation 工具保持一致)
    rotate_code = None
    if hasattr(cfg.pipeline, 'input_rotate_code') and cfg.pipeline.input_rotate_code is not None:
        rotate_code = cfg.pipeline.input_rotate_code
        logger.info(f"🔄 Applying rotation code: {rotate_code}")
        # 旋转后交换宽高
        if rotate_code in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            w, h = h, w

    logger.info(f"Video: {video_path.name} | Frames: {total_frames} | Res: {w}x{h}")
    logger.info(f"Output: {images_dir}")

    # 3. 处理循环
    pbar = tqdm(total=total_frames, unit="frame")

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break

        # 应用旋转
        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)

        # 构造蒙版文件名 (假设是 00000.png 格式)
        mask_name = f"{i:05d}.png"
        path_obj = mask_obj_dir / mask_name
        path_rob = mask_rob_dir / mask_name

        # 读取 Mask (灰度模式)
        # 如果文件不存在 (例如 SAM2 没检测到)，给一个全黑的
        if path_obj.exists():
            m_obj = cv2.imread(str(path_obj), cv2.IMREAD_GRAYSCALE)
        else:
            m_obj = np.zeros((h, w), dtype=np.uint8)

        if path_rob.exists():
            m_rob = cv2.imread(str(path_rob), cv2.IMREAD_GRAYSCALE)
        else:
            m_rob = np.zeros((h, w), dtype=np.uint8)

        # 容错：确保尺寸匹配 (防止视频和mask尺寸不一致)
        if m_obj.shape != (h, w):
            m_obj = cv2.resize(m_obj, (w, h), interpolation=cv2.INTER_NEAREST)
        if m_rob.shape != (h, w):
            m_rob = cv2.resize(m_rob, (w, h), interpolation=cv2.INTER_NEAREST)

        # === The Alpha Trick ===
        # 逻辑：Background = NOT (Object OR Robot)
        # 凡是有物体(255)的地方，Alpha 设为 0 (透明)
        combined_mask = cv2.bitwise_or(m_obj, m_rob)
        alpha_channel = cv2.bitwise_not(combined_mask)

        # 合成 RGBA
        b, g, r = cv2.split(frame)
        rgba = cv2.merge([b, g, r, alpha_channel])

        # 保存
        cv2.imwrite(str(images_dir / mask_name), rgba)
        pbar.update(1)

    cap.release()
    pbar.close()
    logger.info("✅ Alpha Injection Completed.")
    logger.info(f"📂 Dataset ready at: {images_dir}")

if __name__ == "__main__":
    main()