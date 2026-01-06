import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from pathlib import Path

# ================= 配置 =================
# ⚠️ 修改为你的实际路径
WORKSPACE_DIR = Path("/home/lyx/semiff/outputs/20260106_180940") 
IMAGE_PATH = WORKSPACE_DIR / "images" / "00056.png"
CLOUD_PATH = WORKSPACE_DIR / "sparse_cloud.npy"
POSES_PATH = WORKSPACE_DIR / "camera_poses.npy"

# 模型推理时使用的长边尺寸 (你的 wrapper 里写的是 512)
MODEL_LONG_EDGE = 512.0 
# =======================================

def project_points():
    # 1. 加载数据
    if not CLOUD_PATH.exists():
        print("❌ 找不到点云文件")
        return

    cloud = np.load(CLOUD_PATH) # shape (N, 6)
    img = cv2.imread(str(IMAGE_PATH))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    raw_h, raw_w = img.shape[:2]

    print(f"🖼️ 原始图像尺寸: {raw_w} x {raw_h}")
    print(f"☁️ 点云数量: {len(cloud)}")

    # 1.1 加载相机元数据 JSON (自动修正可视化覆盖范围)
    json_path = WORKSPACE_DIR / "cameras_metadata.json"
    if not json_path.exists():
        print("⚠️ 找不到 cameras_metadata.json，使用估算参数")
        meta = None
    else:
        with open(json_path, 'r') as f:
            meta = json.load(f)
        print(f"📄 已加载相机元数据: {json_path}")

        # 获取第0帧的真实内参 (MASt3R 推理出的参数)
        if "0" in meta["frames"]:
            frame_meta = meta["frames"]["0"]
            focal_model = frame_meta["intrinsics"]["focal_length_px"]
            cx_model, cy_model = frame_meta["intrinsics"]["principal_point_px"]
            print(".2f")
            print(".2f")
        else:
            print("⚠️ JSON 中找不到第0帧数据，使用估算参数")
            meta = None

    # 2. 计算模型推理时的实际尺寸 (用于坐标变换)
    scale_factor = MODEL_LONG_EDGE / max(raw_h, raw_w)
    target_h = int(raw_h * scale_factor) // 16 * 16
    target_w = int(raw_w * scale_factor) // 16 * 16

    print(f"📐 模型输入尺寸: {target_w} x {target_h}")

    # 2.1 如果有真实参数，使用真实内参；否则使用估算参数
    if meta is None:
        # 💡 估算内参 (Fallback)
        # 经验值：Dust3R 输出的 focal length 通常约为 W/2 (对应FOV ~90度)
        focal_model = max(target_h, target_w) / 2.0
        cx_model = target_w / 2.0
        cy_model = target_h / 2.0
        print("📏 使用估算内参: focal=%.1f, cx=%.1f, cy=%.1f" % (focal_model, cx_model, cy_model))
    else:
        # ✅ 使用 MASt3R 推理出的真实内参
        # 注意：这里已经是模型尺度下的参数，无需额外计算
        print("🎯 使用真实内参进行精确投影")
    
    # 3. 投影计算
    # 分离 XYZ 和 RGB
    xyz = cloud[:, :3]
    rgb = cloud[:, 3:] / 255.0  # matplotlib 需要 0-1 的 float 颜色

    valid_mask = xyz[:, 2] > 0.1
    pts = xyz[valid_mask]
    colors = rgb[valid_mask]    # 🆕 筛选对应的颜色

    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # 针孔相机投影 (在模型尺度下)
    u_model = (X / Z) * focal_model + cx_model
    v_model = (Y / Z) * focal_model + cy_model
    
    # 4. 映射回原始大图尺寸
    # 坐标从 model 尺寸 -> raw 尺寸
    u_raw = u_model * (raw_w / target_w)
    v_raw = v_model * (raw_h / target_h)
    
    # 过滤画幅外的点
    in_view = (u_raw >= 0) & (u_raw < raw_w) & (v_raw >= 0) & (v_raw < raw_h)
    u_final = u_raw[in_view]
    v_final = v_raw[in_view]
    c_final = colors[in_view]   # 🆕 最终可视化的颜色

    print(f"🎯 投影后在视野内的点数: {len(u_final)}")

    # 5. 绘图
    plt.figure(figsize=(16, 9))
    plt.imshow(img) # 背景原图

    # 🆕 使用 c=c_final (真彩色)
    # 注意：s (点的大小) 可以稍微设大一点 (如 2.0) 以便看清颜色
    plt.scatter(u_final, v_final, c=c_final, s=1.5, alpha=0.8)

    plt.axis('off')
    plt.title(f"Reprojection (True Color, Focal: {focal_model:.1f})")

    save_path = WORKSPACE_DIR / "reprojection_rgb.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 结果已保存: {save_path}")

if __name__ == "__main__":
    project_points()