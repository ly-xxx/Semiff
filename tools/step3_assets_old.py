import sys
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from semiff.geometry.meshing import Mesher
from semiff.geometry.decomposition import ColliderBuilder
from semiff.core.logger import get_logger

logger = get_logger("step3_assets")

def project_points_to_uv(points, K, T_cw):
    """将3D点投影到2D像素坐标"""
    # points: (N, 3), T_cw: (4, 4) world -> camera, K: (3, 3)
    # 1. 转到相机坐标系
    R = T_cw[:3, :3]
    t = T_cw[:3, 3]
    pts_cam = points @ R.T + t

    # 2. 投影
    pts_2d = pts_cam @ K.T
    uv = pts_2d[:, :2] / pts_2d[:, 2:3]
    return uv

def main():
    # 配置路径 (可以后续改用 Hydra)
    base_dir = Path("outputs")
    ply_path = base_dir / "mast3r_result" / "scene.ply"
    poses_path = base_dir / "mast3r_result" / "poses.npy"
    mask_dir = base_dir / "masks_object"  # Step 2 的物体掩码目录
    output_asset_dir = base_dir / "assets"
    output_asset_dir.mkdir(exist_ok=True)

    print("🚀 [Step 3] Asset Generation...")

    # 检查依赖文件
    if not ply_path.exists():
        print(f"❌ Scene PLY not found: {ply_path}. Run Step 1 first.")
        return
    if not poses_path.exists():
        print(f"❌ Poses file not found: {poses_path}. Run Step 1 first.")
        return
    if not mask_dir.exists():
        print(f"❌ Object masks not found: {mask_dir}. Run Step 2 first.")
        return

    # 1. 加载点云和位姿
    print(f"Loading point cloud: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    poses = np.load(poses_path)  # (N_frames, 4, 4) camera-to-world
    print(f"Loaded {len(points)} points, {len(poses)} camera poses")

    # 2. 简化的内参估计 (MASt3R 输出归一化到 512x512)
    H, W = 512, 512
    fx = fy = W * 0.8  # 粗略估计
    cx, cy = W / 2, H / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    print(f"Using camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    # 3. 过滤点云 (Mask Filtering)
    # 简单的投票机制：如果一个点在多个视角的 Mask 里，则保留
    point_votes = np.zeros(len(points), dtype=int)
    valid_frames = 0

    mask_files = sorted(list(mask_dir.glob("*.png")))
    print(f"Found {len(mask_files)} mask files")

    # 采样一些帧进行过滤 (避免太慢)
    sample_indices = np.linspace(0, len(mask_files)-1, min(10, len(mask_files))).astype(int)
    print(f"Sampling {len(sample_indices)} frames for filtering")

    for idx in tqdm(sample_indices, desc="Filtering points"):
        # 读取 Mask
        mask_path = mask_files[idx]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        mask = mask > 127  # 二值化
        if mask.shape != (H, W):
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        # 获取该帧位姿 (World -> Camera)
        if idx >= len(poses):
            continue

        T_wc = poses[idx]
        T_cw = np.linalg.inv(T_wc)

        # 投影所有点
        uvs = project_points_to_uv(points, K, T_cw)

        # 检查是否在 Mask 内
        u, v = uvs[:, 0].astype(int), uvs[:, 1].astype(int)
        valid_idx = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        # 累加投票 (仅对在视锥内的点)
        in_mask = np.zeros(len(points), dtype=bool)
        in_mask[valid_idx] = mask[v[valid_idx], u[valid_idx]]

        point_votes[in_mask] += 1
        point_votes[~in_mask & valid_idx] -= 1  # 如果在视锥内但不在mask里，扣分
        valid_frames += 1

    if valid_frames == 0:
        print("❌ No valid frames found for filtering")
        return

    # 保留分数 > 0 的点
    object_indices = point_votes > 0
    obj_points = points[object_indices]
    obj_colors = colors[object_indices]

    print(f"Extracted {len(obj_points)} points for object (from {valid_frames} frames)")

    if len(obj_points) < 1000:
        print(f"⚠️ Warning: Only {len(obj_points)} points extracted. Object may be too small or poorly segmented.")
        return

    # 4. 保存物体点云
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
    obj_pcd.colors = o3d.utility.Vector3dVector(obj_colors)
    obj_ply_path = output_asset_dir / "object_raw.ply"
    o3d.io.write_point_cloud(str(obj_ply_path), obj_pcd)
    print(f"✅ Saved object point cloud to {obj_ply_path}")

    # 5. 生成 Mesh 和 Collision (调用现有的类)
    try:
        print("Running meshing...")
        mesher = Mesher()
        mesh_path = mesher.run(obj_points, output_path=str(output_asset_dir / "object.obj"))
        print(f"✅ Mesh saved to {mesh_path}")

        print("Running collision decomposition...")
        collider = ColliderBuilder()
        collision_path = collider.decompose(mesh_path, output_path=str(output_asset_dir / "object_collision.obj"))
        print(f"✅ Collision mesh saved to {collision_path}")

        # 保存资产信息
        asset_info = {
            "object_mesh": str(mesh_path),
            "object_collision": str(collision_path),
            "object_pointcloud": str(obj_ply_path),
            "point_count": len(obj_points),
            "extraction_frames": valid_frames
        }

        import json
        with open(output_asset_dir / "asset_info.json", 'w') as f:
            json.dump(asset_info, f, indent=2)
        print(f"✅ Asset info saved to {output_asset_dir / 'asset_info.json'}")

    except Exception as e:
        print(f"⚠️ Mesh/collision generation failed: {e}")
        print("Object point cloud is still available for manual processing")

    print("🎯 [Step 3] Asset generation completed!")
    print(f"📁 Assets saved to: {output_asset_dir}")

if __name__ == "__main__":
    main()