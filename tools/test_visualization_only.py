# tools/test_visualization_only.py - 仅测试可视化部分的脚本

import cv2
import numpy as np
from pathlib import Path
import os

def create_mock_masks(video_path, output_dir, mask_dir):
    """创建模拟的mask数据用于测试"""
    print("Creating mock masks for testing...")

    # 读取视频信息
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Video info: {frame_count} frames, {fps} fps, {width}x{height}")

    # 为每个帧创建模拟mask（简单的圆形mask）
    mask_dir.mkdir(parents=True, exist_ok=True)

    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 4  # 圆形半径

    for frame_idx in range(min(frame_count, 100)):  # 最多处理100帧
        # 创建圆形mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # 添加一些随机性，让mask在帧间轻微变化
        offset_x = int(10 * np.sin(frame_idx * 0.1))
        offset_y = int(10 * np.cos(frame_idx * 0.1))

        cv2.circle(mask,
                  (center_x + offset_x, center_y + offset_y),
                  radius,
                  255,  # 白色
                  -1)   # 填充

        # 保存mask
        mask_path = mask_dir / f"{frame_idx:05d}.npz"
        np.savez_compressed(mask_path, mask=mask.astype(bool))

        if frame_idx % 20 == 0:
            print(f"Created mock mask for frame {frame_idx}")

    print(f"Created {min(frame_count, 100)} mock masks")
    return mask_dir

def generate_visualization(video_path, output_dir, mask_dir):
    """生成可视化视频"""
    print("Generating visualization videos...")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 输出视频路径
    out_overlay_path = output_dir / "vis_overlay_mock.mp4"
    out_green_path = output_dir / "vis_green_screen_mock.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_overlay = cv2.VideoWriter(str(out_overlay_path), fourcc, fps, (w, h))
    writer_green = cv2.VideoWriter(str(out_green_path), fourcc, fps, (w, h))

    frame_idx = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 加载对应的mask
        mask_path = mask_dir / f"{frame_idx:05d}.npz"
        if not mask_path.exists():
            break

        mask_data = np.load(mask_path)
        mask = mask_data['mask'].astype(bool)  # 确保是boolean类型
        mask_uint8 = (mask * 255).astype(np.uint8)

        # === 效果 1: 红色半透明叠加 (Overlay) ===
        red_mask = np.zeros_like(frame)
        red_mask[:, :, 2] = 255  # Red channel

        overlay = frame.copy()
        overlay[mask] = cv2.addWeighted(frame[mask], 0.5, red_mask[mask], 0.5, 0)

        # 画轮廓
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

        writer_overlay.write(overlay)

        # === 效果 2: 绿幕分离 (Green Screen) ===
        green_bg = np.zeros_like(frame)
        green_bg[:] = (0, 255, 0)  # 绿色背景

        foreground = frame.copy()
        foreground[~mask] = green_bg[~mask]  # 非mask区域设为绿色

        writer_green.write(foreground)

        frame_idx += 1
        processed_frames += 1

        if processed_frames % 20 == 0:
            print(f"Processed {processed_frames} frames...")

    cap.release()
    writer_overlay.release()
    writer_green.release()

    print("✅ Visualization videos generated:")
    print(f"   - {out_overlay_path} (红色叠加效果)")
    print(f"   - {out_green_path} (绿幕分离效果)")

def main():
    print("🎥 Mock Segmentation Visualization Test")
    print("=" * 50)

    video_path = "test_bench.mp4"
    output_dir = Path("outputs")

    # 检查视频文件
    if not Path(video_path).exists():
        print(f"ERROR: Video file not found: {video_path}")
        return

    print(f"Video: {video_path}")
    print(f"Output dir: {output_dir}")

    # 创建输出目录
    output_dir.mkdir(exist_ok=True)

    # 创建模拟mask
    mask_dir = output_dir / "masks_mock"
    create_mock_masks(video_path, output_dir, mask_dir)

    # 生成可视化
    generate_visualization(video_path, output_dir, mask_dir)

    print("\n🎉 Test completed successfully!")
    print("You can now view the generated videos to see the segmentation visualization.")

if __name__ == "__main__":
    main()



