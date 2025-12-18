# tools/test_simple.py - 简化的测试脚本

import cv2
import numpy as np
import sys
from pathlib import Path

def test_video_and_opencv():
    """测试视频读取和OpenCV交互"""
    video_path = "test_bench.mp4"

    print(f"Testing video: {video_path}")

    # 检查视频文件是否存在
    if not Path(video_path).exists():
        print("ERROR: Video file not found!")
        return

    # 尝试打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open video file!")
        return

    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame!")
        return

    print(f"Video opened successfully. Frame shape: {frame.shape}")

    # 测试交互式选点
    print("Testing interactive selection...")

    # 缩放图像以适应屏幕
    h, w = frame.shape[:2]
    scale = 1.0
    if h > 800 or w > 1200:
        scale = min(800/h, 1200/w)
        display_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    else:
        display_frame = frame.copy()

    print("Please click on the object in the popup window...")
    print("Controls: [Left Click] Select Point  [Space/Enter] Confirm  [Esc] Skip")

    prompt_points = []
    window_name = "Test Selection"

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            real_x, real_y = int(x / scale), int(y / scale)
            prompt_points.append([real_x, real_y])
            print(f"Selected point: ({real_x}, {real_y})")
            cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, display_frame)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, display_frame)

    print("Window opened. Waiting for user input...")

    # 等待按键或超时
    timeout_counter = 0
    confirmed = False

    while timeout_counter < 300:  # 30秒超时
        key = cv2.waitKey(100) & 0xFF
        if key == 32 or key == 13:  # Space or Enter
            confirmed = True
            break
        elif key == 27:  # Esc
            break
        timeout_counter += 1

    cv2.destroyAllWindows()

    if confirmed and prompt_points:
        print(f"Selection confirmed: {prompt_points[-1]}")
        print("✅ Interactive selection test PASSED!")
    else:
        print("Selection not confirmed or no points selected")
        print("⚠️  Interactive selection test completed (no selection made)")

    cap.release()

if __name__ == "__main__":
    test_video_and_opencv()



