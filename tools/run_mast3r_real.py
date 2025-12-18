import os
import json
import torch
import numpy as np
import trimesh
from PIL import Image, ImageOps
from tqdm import tqdm

# === MASt3R / Dust3R 依赖 ===
# 确保你的环境里可以通过 'from dust3r import ...' 导入
# 如果 mast3r 代码在 third_party 下，可能需要 sys.path.append
import sys
sys.path.append("third_party/mast3r")

# 设置 dust3r 路径 (必须在导入 dust3r 之前)
import mast3r.utils.path_to_dust3r  # noqa

from dust3r.inference import inference
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# === 配置 ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MASt3R 权重路径 (请修改为你下载的实际路径)
MODEL_PATH = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
INPUT_DIR = "outputs/train_data/images"  # 你的 RGBA 或 RGB 图片目录
OUTPUT_DIR = "outputs/mast3r_result"
IMG_SIZE = 288  # 降低分辨率以节省显存，288 对 12G 显存更友好
SCHEDULE = "linear" # 学习率调度

def get_resized_image(pil_img, target_size=512):
    """
    强制缩放：所有图片缩放到相同尺寸，且确保长宽均为 16 的倍数。

    Args:
        pil_img: PIL Image 对象
        target_size: 目标尺寸 (正方形，必须是 16 的倍数)
    """
    # 强制缩放到正方形，确保模型兼容性
    resized_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)

    return resized_img

def load_and_process_images(folder_path, size=512):
    """
    读取图片，强制 Exif 旋转，并转换为 Tensor 列表
    """
    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')])

    # 关键帧采样：如果图片太多(>100)，每隔 5 帧取一张，防止显存爆炸
    # 163帧 -> 约32帧，足以重建场景且速度快
    if len(image_files) > 50:
        print(f"⚠️ Image count {len(image_files)} is large. Sampling every 5th frame.")
        image_files = image_files[::5]

    imgs = []
    print("📸 Loading and preprocessing images...")
    for img_path in tqdm(image_files):
        # 1. 打开图片
        pil_img = Image.open(img_path).convert('RGB')

        # 2. 【关键】Exif 旋转修正 (对齐 SAM2 坐标系)
        pil_img = ImageOps.exif_transpose(pil_img)

        # 3. 强制缩放 (所有图片缩放到相同尺寸)
        pil_img = get_resized_image(pil_img, target_size=size)
        new_w, new_h = pil_img.size

        # 4. 转 Tensor: (1, 3, H, W)
        img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        # 存储为一个 dict 列表，符合 Dust3R 接口
        # true_shape 应该是 [[H, W]] 格式的二维数组
        imgs.append({'img': img_tensor, 'true_shape': np.array([[size, size]]), 'idx': len(imgs), 'instance': str(len(imgs))})

    return imgs, image_files

def run_mast3r_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载模型
    print(f"🤖 Loading MASt3R model from {MODEL_PATH}...")
    try:
        model = AsymmetricMASt3R.from_pretrained(MODEL_PATH).to(DEVICE)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Tip: Ensure 'dust3r' and 'mast3r' are in your python path.")
        return

    # 2. 准备数据
    imgs, file_paths = load_and_process_images(INPUT_DIR, size=IMG_SIZE)
    if len(imgs) < 2:
        print("❌ Need at least 2 images!")
        return

    # 3. 生成配对 (Pair Generation) - 瘦身版策略
    # 对于 12G 显存，使用 swin-3 且不通过对称加倍，大幅减少配对数
    print("🔗 Generating image pairs (Lean Mode)...")
    pairs = make_pairs(imgs, scene_graph='swin-3', prefilter=None, symmetrize=False)

    # 4. 模型推理 (Inference)
    print("🧠 Running inference (this may take a while)...")
    output = inference(pairs, model, DEVICE, batch_size=1, verbose=True)

    # 5. 全局优化 (Global Alignment)
    # 这是从两两匹配恢复全局相机位姿和点云的关键步骤
    print("🌍 Running Global Alignment...")
    scene = global_aligner(output, device=DEVICE, mode=GlobalAlignerMode.PointCloudOptimizer)

    # 显存回收
    torch.cuda.empty_cache()

    # 运行优化 (pnp, 焦距优化等)
    loss = scene.compute_global_alignment(init="mst", niter=300, schedule=SCHEDULE, lr=0.01)
    print(f"✅ Optimization done. Final Loss: {loss}")

    # 6. 后处理与保存
    save_results(scene, file_paths)

def save_results(scene, file_paths):
    """
    导出位姿 JSON 和点云 PLY (增强鲁棒性版)
    兼容 Tensor/Numpy，自动处理 CHW/HWC 格式差异
    """
    print("💾 Saving results...")

    # --- 1. 保存位姿 (Poses) ---
    try:
        # get_im_poses 返回的是 Tensor，需要转 numpy
    poses = scene.get_im_poses().detach().cpu().numpy()
    cameras_out = {}
    for idx, pose in enumerate(poses):
            # 防止索引越界（防御性编程）
            if idx < len(file_paths):
                file_name = os.path.basename(file_paths[idx])
            else:
                file_name = f"unknown_{idx}.jpg"

        cameras_out[f"frame_{idx}"] = {
                "file": file_name,
            "transform_matrix": pose.tolist()
        }

    json_path = os.path.join(OUTPUT_DIR, "cameras.json")
    with open(json_path, 'w') as f:
        json.dump(cameras_out, f, indent=4)
    np.save(os.path.join(OUTPUT_DIR, "poses.npy"), poses)
    print(f"   -> Saved poses to {json_path}")
    except Exception as e:
        print(f"⚠️ Error saving poses: {e}")

    # --- 2. 保存点云 (Point Cloud) ---
    print("   -> Merging point clouds...")

    # get_pts3d() 返回通常是 Tensor 列表
    pts3d = scene.get_pts3d()

    all_pts = []
    all_colors = []

    for i in range(len(pts3d)):
        try:
            # --- A. 处理几何信息 (XYZ) ---
            pts_data = pts3d[i]
            if isinstance(pts_data, torch.Tensor):
                pts = pts_data.detach().cpu().numpy().reshape(-1, 3)
            else:
                pts = pts_data.reshape(-1, 3)

            # --- B. 处理颜色信息 (RGB) - 核心修复区 ---
            # 1. 获取原始数据对象
            img_entry = scene.imgs[i]
            # scene.imgs 可能是字典列表，也可能是直接的图像列表
            if isinstance(img_entry, dict):
                raw_img = img_entry['img']
            else:
                raw_img = img_entry

            # 2. 统一转为 Numpy
            if isinstance(raw_img, torch.Tensor):
                img_np = raw_img.detach().cpu().numpy()
            elif isinstance(raw_img, np.ndarray):
                img_np = raw_img
        else:
                print(f"⚠️ Frame {i}: Unknown image type {type(raw_img)}, skipping.")
                continue

            # 3. 维度标准化 -> 目标格式 (H, W, 3)
            # 此时 img_np 可能是 (1, 3, H, W), (3, H, W), 或 (H, W, 3)

            # 情况1: 4维 (Batch, C, H, W) -> 去掉 Batch
            if img_np.ndim == 4:
                img_np = img_np.squeeze(0)

            # 情况2: 3维 (C, H, W) 通常 C=3 -> 转置为 (H, W, C)
            if img_np.ndim == 3 and img_np.shape[0] == 3 and img_np.shape[2] != 3:
                img_np = np.transpose(img_np, (1, 2, 0))

            # 4. 展平
            color = img_np.reshape(-1, 3)

            # 5. 颜色值范围归一化检测 (0-1 还是 0-255)
            # 如果最大值很小(<=1.05)，认为是 float 0-1，需要乘 255
            if color.max() <= 1.05:
        color = (color * 255).astype(np.uint8)
            else:
                color = color.astype(np.uint8)

            # --- C. 过滤与合并 ---
            # 过滤掉原点 (0,0,0) 或无效深度点
            # 计算每个点的模长，太小的视为无效
            norms = np.linalg.norm(pts, axis=1)
            valid_mask = norms > 1e-6

            # 再次检查形状匹配
            if len(pts) == len(color):
                all_pts.append(pts[valid_mask])
                all_colors.append(color[valid_mask])
            else:
                # 如果形状不匹配，尝试 resize color (极其罕见但在 resizing 逻辑不严谨时会发生)
                print(f"⚠️ Frame {i} mismatch: pts {pts.shape} vs color {color.shape}. Skipping.")

        except Exception as step_e:
            print(f"⚠️ Error processing frame {i}: {step_e}")
            import traceback
            traceback.print_exc()
            continue

    # --- D. 导出最终文件 ---
    if all_pts:
        final_pts = np.concatenate(all_pts, axis=0)
        final_colors = np.concatenate(all_colors, axis=0)

        ply_path = os.path.join(OUTPUT_DIR, "scene.ply")
        try:
            # 优先使用 Trimesh
            pcd = trimesh.PointCloud(vertices=final_pts, colors=final_colors)
        pcd.export(ply_path)
            print(f"✅ Success! Saved colored point cloud to {ply_path} ({len(final_pts)} points)")
        except Exception as e:
            print(f"❌ Trimesh export failed: {e}. Falling back to text format.")
            # 备用方案：纯文本写入
            header = "ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header".format(len(final_pts))
            np.savetxt(ply_path,
                       np.hstack((final_pts, final_colors)),
                       fmt='%.6f %.6f %.6f %d %d %d',
                       header=header, comments='')
            print(f"   (Fallback) Saved raw PLY to {ply_path}")
    else:
        print("❌ Error: No valid points generated from any frame.")

if __name__ == "__main__":
    # 显存清理
    torch.cuda.empty_cache()
    run_mast3r_pipeline()