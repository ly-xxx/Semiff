import sys
import cv2
import numpy as np
import json
import subprocess
import pickle
from pathlib import Path
from tqdm import tqdm

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors
from yourdfpy import URDF
import open3d as o3d
from semiff.core.logger import get_logger

logger = get_logger("step4_robot")

def get_video_rotation(video_path):
    """检测视频旋转元数据"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None, False

        data = json.loads(result.stdout)
        video_stream = next((s for s in data.get('streams', []) if s['codec_type'] == 'video'), None)
        if not video_stream:
            return None, False

        tags = video_stream.get('tags', {})
        rotate = int(tags.get('rotate', 0))

        if rotate == 90:
            return cv2.ROTATE_90_CLOCKWISE, True
        elif rotate == 180:
            return cv2.ROTATE_180, False
        elif rotate == 270:
            return cv2.ROTATE_90_COUNTERCLOCKWISE, True
        else:
            return None, False
    except Exception:
        return None, False

def extract_robot_images(video_path, mask_dir, output_dir, rotate_code=None):
    """
    从视频中提取机器人相关的图像，用于 3DGS 训练
    """
    logger.info("Extracting robot images for 3DGS training...")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 应用旋转
    if rotate_code is not None:
        if rotate_code in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            w, h = h, w

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Video: {w}x{h} @ {fps}fps, {total_frames} frames")
    logger.info(f"Output: {images_dir}")

    extracted_count = 0
    pbar = tqdm(total=total_frames, desc="Extracting frames")

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # 应用旋转
        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)

        # 构造掩码文件名
        mask_name = f"{frame_idx:05d}.png"
        mask_path = mask_dir / mask_name

        # 读取机器人掩码
        if mask_path.exists():
            robot_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if robot_mask is not None:
                # 确保尺寸匹配
                if robot_mask.shape != frame.shape[:2]:
                    robot_mask = cv2.resize(robot_mask, (frame.shape[1], frame.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)

                # 创建 RGBA 图像，机器人区域为不透明
                robot_mask = (robot_mask > 127).astype(np.uint8) * 255
                b, g, r = cv2.split(frame)
                rgba = cv2.merge([b, g, r, robot_mask])

                # 保存
                output_path = images_dir / mask_name
                cv2.imwrite(str(output_path), rgba)
                extracted_count += 1

        pbar.update(1)

    cap.release()
    pbar.close()

    logger.info(f"✅ Extracted {extracted_count} robot images")
    return extracted_count

def create_nerfstudio_config(images_dir, output_dir):
    """创建 NeRFStudio 数据配置文件"""
    config = {
        "method_name": "gaussian-splatting",
        "data": str(images_dir),
        "output_dir": str(output_dir / "nerfstudio")
    }

    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return config_path

def run_nerfstudio_training(config_path, output_dir, max_steps=30000):
    """
    调用 NeRFStudio 训练 3DGS
    """
    logger.info("Starting NeRFStudio 3DGS training...")

    cmd = [
        "ns-train",
        "gaussian-splatting",
        "--data", str(config_path),
        "--output-dir", str(output_dir),
        "--max-num-iterations", str(max_steps),
        "--save-only-latest-checkpoint", "False",
        "--viewer.quit-on-train-completion", "True"
    ]

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(output_dir))

        if result.returncode != 0:
            logger.error(f"NeRFStudio training failed: {result.stderr}")
            return False
        else:
            logger.info("✅ NeRFStudio training completed successfully")
            logger.info(result.stdout)
            return True

    except Exception as e:
        logger.error(f"Error running NeRFStudio: {e}")
        return False

def extract_final_model(output_dir):
    """
    从 NeRFStudio 输出中提取最终的 PLY 模型
    """
    nerfstudio_dir = output_dir / "nerfstudio"

    if not nerfstudio_dir.exists():
        logger.error(f"NeRFStudio output directory not found: {nerfstudio_dir}")
        return None

    subdirs = [d for d in nerfstudio_dir.iterdir() if d.is_dir()]
    if not subdirs:
        logger.error("No training directories found")
        return None

    latest_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
    logger.info(f"Using latest training directory: {latest_dir}")

    ply_files = list(latest_dir.glob("*.ply"))
    if not ply_files:
        logger.error("No PLY files found in training directory")
        return None

    ply_file = ply_files[0]
    logger.info(f"Found PLY file: {ply_file}")

    final_ply = output_dir / "robot_gs.ply"
    import shutil
    shutil.copy2(ply_file, final_ply)

    logger.info(f"✅ Copied final model to: {final_ply}")
    return final_ply

def load_gs_ply(path):
    """
    加载 3DGS PLY 文件
    """
    logger.info(f"Loading 3DGS PLY: {path}")
    plydata = PlyData.read(path)

    xyz = np.stack((
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ), axis=1)

    try:
        f_dc_0 = plydata['vertex']['f_dc_0']
        f_dc_1 = plydata['vertex']['f_dc_1']
        f_dc_2 = plydata['vertex']['f_dc_2']
        colors = 0.5 + 0.28209479177387814 * np.stack([f_dc_0, f_dc_1, f_dc_2], axis=1)
        colors = np.clip(colors, 0, 1)
    except:
        logger.warning("Could not load colors, using default")
        colors = np.ones((len(xyz), 3)) * 0.5

    vertex_data = {}
    for prop in plydata['vertex'].properties:
        if prop.name not in ['x', 'y', 'z']:
            try:
                vertex_data[prop.name] = plydata['vertex'][prop.name]
            except:
                pass

    logger.info(f"Loaded {len(xyz)} Gaussians")
    return xyz, colors, vertex_data

def sample_urdf_points(urdf_path, num_samples=50000, canonical_pose=None):
    """
    从 URDF 采样点云
    """
    logger.info(f"Loading URDF: {urdf_path}")
    robot = URDF.load(urdf_path)

    if canonical_pose is None:
        canonical_pose = {joint.name: 0.0 for joint in robot.joints}

    robot.update_cfg(canonical_pose)
    logger.info(f"Set robot to canonical pose: {canonical_pose}")

    import trimesh
    scene = robot.scene
    combined_mesh = scene.dump(concatenate=True)
    points, _ = trimesh.sample.sample_surface(combined_mesh, num_samples)

    logger.info(f"Sampled {len(points)} points from URDF")
    return points, robot

def align_gs_to_urdf(gs_xyz, urdf_points, threshold=0.02):
    """
    ICP 对齐 3DGS 到 URDF
    """
    logger.info("Creating point clouds for ICP...")

    gs_pcd = o3d.geometry.PointCloud()
    gs_pcd.points = o3d.utility.Vector3dVector(gs_xyz)

    urdf_pcd = o3d.geometry.PointCloud()
    urdf_pcd.points = o3d.utility.Vector3dVector(urdf_points)

    gs_center = gs_pcd.get_center()
    urdf_center = urdf_pcd.get_center()
    trans_init = np.eye(4)
    trans_init[:3, 3] = urdf_center - gs_center

    logger.info("Running ICP alignment...")

    gs_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    urdf_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))

    reg_p2l = o3d.pipelines.registration.registration_icp(
        gs_pcd, urdf_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )

    logger.info(".4f")
    logger.info(".4f")

    return reg_p2l.transformation

def bind_gaussians_to_links(gs_points, robot, transform_matrix):
    """
    将 3DGS 高斯点绑定到机器人 Link
    """
    logger.info("Binding Gaussians to robot links...")

    ones = np.ones((len(gs_points), 1))
    points_homo = np.hstack([gs_points, ones])
    points_aligned = (transform_matrix @ points_homo.T).T[:, :3]

    link_names = []
    link_meshes = []
    link_transforms = []

    for link_name in robot.link_map.keys():
        link = robot.link_map[link_name]

        if len(link.visuals) == 0:
            continue

        import trimesh
        meshes = []
        for visual in link.visuals:
            if visual.geometry.mesh:
                try:
                    m = visual.geometry.mesh.meshes[0]
                    m.apply_transform(visual.origin)
                    meshes.append(m)
                except:
                    continue

        if meshes:
            combined_mesh = trimesh.util.concatenate(meshes)
            link_transform = robot.get_transform(link_name)
            combined_mesh.apply_transform(link_transform)

            link_names.append(link_name)
            link_meshes.append(combined_mesh)
            link_transforms.append(link_transform)

    logger.info(f"Found {len(link_names)} links with geometry: {link_names}")

    if len(link_names) == 0:
        logger.error("No links with geometry found!")
        return [], np.full(len(gs_points), -1, dtype=np.int32), points_aligned

    link_samples = []
    for mesh in link_meshes:
        samples, _ = trimesh.sample.sample_surface(mesh, 2000)
        link_samples.append(samples)

    logger.info(f"Computing distances for {len(points_aligned)} Gaussians...")

    num_points = len(points_aligned)
    link_indices = np.full(num_points, -1, dtype=np.int32)
    min_dists = np.full(num_points, np.inf)

    for i, samples in enumerate(tqdm(link_samples, desc="Processing links")):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(samples)
        dists, _ = nbrs.kneighbors(points_aligned)
        dists = dists.flatten()

        mask = dists < min_dists
        min_dists[mask] = dists[mask]
        link_indices[mask] = i

    unique, counts = np.unique(link_indices, return_counts=True)
    for link_idx, count in zip(unique, counts):
        if link_idx >= 0:
            logger.info(f"  {link_names[link_idx]}: {count} Gaussians")
        else:
            logger.info(f"  Unassigned: {count} Gaussians")

    return link_names, link_indices, points_aligned

def save_binding_data(output_dir, link_names, link_indices, transform_matrix,
                     gs_xyz, gs_colors, vertex_data):
    """
    保存绑定数据
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    binding_data = {
        'link_names': link_names,
        'link_indices': link_indices,
        'transform_matrix': transform_matrix,
        'num_gaussians': len(gs_xyz)
    }

    binding_path = output_dir / "binding_data.pkl"
    with open(binding_path, 'wb') as f:
        pickle.dump(binding_data, f)

    logger.info(f"✅ Saved binding data to {binding_path}")

    transformed_data = {
        'xyz': gs_xyz,
        'colors': gs_colors,
        'vertex_data': vertex_data
    }

    transformed_path = output_dir / "transformed_gaussians.pkl"
    with open(transformed_path, 'wb') as f:
        pickle.dump(transformed_data, f)

    logger.info(f"✅ Saved transformed Gaussians to {transformed_path}")


def run_3dgs_pipeline(urdf_path):
    """
    执行 3DGS 训练和绑定流水线
    """
    logger.info("🎨 Running 3DGS training and binding pipeline...")

    video_path = Path("test_bench.mp4")
    mask_dir = Path("outputs/masks_robot")
    output_dir = Path("outputs/robot_gs_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查依赖
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return False

    if not mask_dir.exists():
        logger.error(f"Robot masks not found: {mask_dir}. Run Step 2 first.")
        return False

    # 1. 提取机器人图像
    rotate_code, _ = get_video_rotation(video_path)
    if rotate_code is not None:
        logger.info(f"Detected video rotation: {rotate_code}")

    try:
        extracted_count = extract_robot_images(video_path, mask_dir, output_dir, rotate_code)
        if extracted_count == 0:
            logger.error("No robot images extracted!")
            return False
    except Exception as e:
        logger.error(f"Failed to extract images: {e}")
        return False

    # 2. 训练 3DGS
    images_dir = output_dir / "images"
    config_path = create_nerfstudio_config(images_dir, output_dir)

    success = run_nerfstudio_training(config_path, output_dir)
    if not success:
        logger.error("NeRFStudio training failed")
        return False

    # 3. 提取模型
    gs_ply_path = extract_final_model(output_dir)
    if not gs_ply_path:
        logger.error("Failed to extract final model")
        return False

    # 4. 对齐和绑定
    logger.info("Starting alignment and binding...")

    try:
        # 加载 3DGS
        gs_xyz, gs_colors, vertex_data = load_gs_ply(gs_ply_path)

        # 从 URDF 采样参考点云
        urdf_points, robot = sample_urdf_points(str(urdf_path))

        # ICP 对齐
        transform_matrix = align_gs_to_urdf(gs_xyz, urdf_points)

        # 绑定
        link_names, link_indices, points_aligned = bind_gaussians_to_links(
            gs_xyz, robot, transform_matrix)

        # 保存
        binding_output_dir = Path("outputs/assets/robot_binding")
        save_binding_data(binding_output_dir, link_names, link_indices, transform_matrix,
                         gs_xyz, gs_colors, vertex_data)

        logger.info("✅ 3DGS alignment and binding completed!")
        return True

    except Exception as e:
        logger.error(f"Alignment/binding failed: {e}")
        return False

def main():
    """
    Step 4: Robot 3DGS Modeling

    这个步骤执行以下操作：
    1. 从机器人掩码中提取图像
    2. 使用 NeRFStudio 训练 3DGS 模型
    3. ICP 对齐 3DGS 到 URDF 坐标系
    4. 将高斯点绑定到机器人 Link

    使用前请确保：
    - 已运行 step1_reconstruct.py 和 step2_segment.py
    - 已安装 plyfile, scikit-learn, yourdfpy
    - URDF 文件路径正确
    """
    # 硬编码 URDF 路径 (可以后续改用配置文件)
    urdf_path = "path/to/your/robot.urdf"  # 请修改为实际路径

    logger.info("🚀 [Step 4] Robot 3DGS Modeling...")
    logger.info(f"URDF: {urdf_path}")

    success = run_3dgs_pipeline(urdf_path)

    if success:
        logger.info("🎯 [Step 4] Robot 3DGS modeling completed!")
        logger.info("Next: Run step5_integrate.py for final validation")
    else:
        logger.error("❌ [Step 4] Robot 3DGS modeling failed")

if __name__ == "__main__":
    main()