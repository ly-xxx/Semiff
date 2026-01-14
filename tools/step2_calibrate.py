"""
tools/step2_calibrate.py
Semiff Pipeline V2 Step 2: 运动学自标定 (Kinematic Self-Calibration) + 内参修正 (Intrinsics Correction)
Usage:
    auto mode:   python tools/step2_calibrate.py pipeline.mode=auto
    manual mode: python tools/step2_calibrate.py pipeline.mode=manual pipeline.parent_workspace=outputs/xxxx_step1
"""
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.spatial  # 用于KDTree
from scipy.spatial import cKDTree

import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import trimesh
import pytorch_kinematics as pk
import cv2
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

# 导入统一路径管理工具
_current_file = Path(__file__).resolve()
_src_dir = _current_file.parents[1] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from semiff.core.workspace import WorkspaceManager

# 🔧 使用统一方法获取项目根目录
PROJECT_ROOT = WorkspaceManager.find_project_root(start_path=_current_file.parent)
from semiff.solvers.aligner import HybridAligner
from semiff.engine.math_utils import project_points, rotation_6d_to_matrix
from semiff.engine.render import DifferentiableRasterizer

# Try to import open3d
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Step2")

if not HAS_OPEN3D:
    logger.warning("Open3D not available. Some features will be disabled.")


def save_debug_ply(pcd, filename, folder):
    """辅助函数：安全保存调试点云"""
    try:
        if not HAS_OPEN3D: return
        out_path = folder / filename
        o3d.io.write_point_cloud(str(out_path), pcd)
    except Exception as e:
        logger.warning(f"Failed to save debug PLY {filename}: {e}")


def segment_robot_from_scene(ply_path, mask_dir, poses_path, intrinsics_path, device='cuda'):
    """
    [重构版 V7] 机械臂点云分割 - 密度优先版
    新增逻辑：
    1. 【密度滤波】计算 KNN 距离，只保留密度最高的前 90% 点，打断粘连。
    2. 【保守去墙】收紧墙壁 RANSAC 阈值，防止切穿机械臂。
    """
    # 0. 准备调试路径
    ply_path = Path(ply_path)
    debug_dir = ply_path.parent / "debug_segmentation"
    debug_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"💾 [Debug] Saving intermediate clouds to: {debug_dir}")

    if not HAS_OPEN3D:
        logger.warning("❌ Open3D missing. Returning empty.")
        return np.zeros((0, 3), dtype=np.float32)

    # 1. 加载与基础预处理
    pcd = o3d.io.read_point_cloud(str(ply_path))
    save_debug_ply(pcd, "00_raw_input.ply", debug_dir)

    logger.info(f"   Input points: {len(pcd.points)}")
    
    # [Step 1] 统计滤波去噪 (去除明显离群点)
    # nb_neighbors: 邻域点数, std_ratio: 越小杀得越狠
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.0)
    pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    
    if len(pcd.points) == 0:
        logger.error("❌ Point cloud empty after statistical denoising.")
        return np.zeros((0, 3), dtype=np.float32)
        
    save_debug_ply(pcd, "01_statistical_denoised.ply", debug_dir)

    # === 🔥 [Step 1.5] 新增：基于密度的滤波 (Density Filtering) ===
    # 目的：打断机械臂与背景之间的稀疏“粘连”
    logger.info("📉 Filtering by density (Top 90%)...")
    try:
        # 计算每个点到最近 16 个点的平均距离
        dists = pcd.compute_nearest_neighbor_distance()
        dists = np.asarray(dists)
        
        # 找到分位数阈值 (保留距离最小的前 90% -> 密度最大的点)
        # 如果粘连依然严重，可以尝试降低这个值 (例如 0.85)
        density_threshold = np.percentile(dists, 90) 
        
        # 筛选
        mask_density = dists < density_threshold
        idx_density = np.where(mask_density)[0]
        
        pcd = pcd.select_by_index(idx_density)
        logger.info(f"   Density Filter: Kept {len(idx_density)} points (Threshold: {density_threshold:.4f}m)")
        save_debug_ply(pcd, "02_density_filtered.ply", debug_dir)
        
    except Exception as e:
        logger.warning(f"Density filtering failed: {e}")

    # 2. 准备投影数据 (后续步骤需要 Tensor)
    points_np = np.asarray(pcd.points)
    points_t = torch.tensor(points_np, dtype=torch.float32, device=device)
    
    c2w_all = np.load(poses_path)
    mask_files = sorted(list(Path(mask_dir).glob("*.png")))
    
    if os.path.exists(intrinsics_path):
        K_all = np.load(intrinsics_path)
    else:
        H, W = 720, 1280
        focal = 0.5 * W / np.tan(0.5 * 60 * np.pi / 180)
        K_dummy = np.array([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]])
        K_all = np.tile(K_dummy[None], (len(c2w_all), 1, 1))

    # 3. 投票 (Space Carving)
    n_views = min(15, len(mask_files), len(c2w_all))
    indices = np.linspace(0, min(len(mask_files), len(c2w_all))-1, n_views, dtype=int)
    vote_count = torch.zeros(points_t.shape[0], device=device, dtype=torch.int32)

    for idx in indices:
        mask_cv = cv2.imread(str(mask_files[idx]), cv2.IMREAD_GRAYSCALE)
        if mask_cv is None: continue
        
        mask_t = torch.tensor(mask_cv, device=device) > 128
        H, W = mask_t.shape
        K = torch.tensor(K_all[idx], dtype=torch.float32, device=device)
        pose_w2c = torch.inverse(torch.tensor(c2w_all[idx], dtype=torch.float32, device=device))
        
        pts_cam = torch.matmul(points_t, pose_w2c[:3, :3].T) + pose_w2c[:3, 3]
        z = pts_cam[:, 2]
        valid_z = z > 0.1
        
        uv = torch.matmul(pts_cam[:, :3], K.T)
        u = (uv[:, 0] / (z + 1e-6)).long()
        v = (uv[:, 1] / (z + 1e-6)).long()
        
        in_view = valid_z & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        valid_idx = torch.nonzero(in_view).squeeze()
        
        if len(valid_idx.shape) > 0 and len(valid_idx) > 0:
            is_robot = mask_t[v[valid_idx], u[valid_idx]]
            vote_count[valid_idx[is_robot]] += 1

    # 4. 生成种子与候选点
    thresh_seed = max(2, int(n_views * 0.4)) 
    mask_seed = (vote_count >= thresh_seed).cpu().numpy()
    mask_candidate = (vote_count >= 2).cpu().numpy()
    
    idx_seed = np.where(mask_seed)[0]
    idx_candidate = np.where(mask_candidate)[0]

    if len(idx_seed) == 0:
        logger.error("❌ No seed points found. Check poses/masks.")
        return np.zeros((0, 3), dtype=np.float32)

    pcd_seed = pcd.select_by_index(idx_seed)
    save_debug_ply(pcd_seed, "03_seeds.ply", debug_dir)

    # [Step 2] ROI 裁切 (物理隔离)
    seed_points = points_np[idx_seed]
    seed_center = np.mean(seed_points, axis=0)
    
    candidate_points = points_np[idx_candidate]
    dists = np.linalg.norm(candidate_points - seed_center, axis=1)
    
    keep_roi = dists < 1.0 
    idx_roi = idx_candidate[keep_roi]
    
    if len(idx_roi) == 0:
        return points_np[idx_seed]
        
    pcd_roi = pcd.select_by_index(idx_roi)
    save_debug_ply(pcd_roi, "04_roi_cropped.ply", debug_dir)

    # [Step 3] RANSAC 去桌面
    pcd_objects = pcd_roi
    try:
        plane, inliers = pcd_roi.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
        [a, b, c, d] = plane
        
        check_val = np.mean(a * seed_points[:, 0] + b * seed_points[:, 1] + c * seed_points[:, 2] + d)
        if check_val < 0:
            a, b, c, d = -a, -b, -c, -d
            
        pts_roi_np = np.asarray(pcd_roi.points)
        dists_plane = (a * pts_roi_np[:, 0] + b * pts_roi_np[:, 1] + c * pts_roi_np[:, 2] + d)
        
        mask_above = dists_plane > 0.02
        idx_final = np.where(mask_above)[0]
        
        pcd_objects = pcd_roi.select_by_index(idx_final)
        save_debug_ply(pcd_objects, "05_table_removed.ply", debug_dir)
        
    except Exception as e:
        logger.warning(f"Table removal failed: {e}")

    # === [Step 4] RANSAC 去墙壁 (保守修正版) ===
    # 之前的问题：06 把机械臂挖穿了。
    # 修正：
    # 1. 如果刚才的密度滤波成功了，机械臂应该已经和墙壁断开了，可以依赖聚类。
    # 2. 我们将 RANSAC 的阈值收紧 (0.03 -> 0.015)，只切除非常平整的表面。
    # 3. 只有当拟合出的平面点数占比非常大 (>60%) 时才执行切除。
    if len(pcd_objects.points) > 2000:
        try:
            # 更加严格的阈值 (1.5cm)，防止切掉曲面的机械臂
            plane2, inliers2 = pcd_objects.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)
            
            ratio = len(inliers2) / len(pcd_objects.points)
            if ratio > 0.6: # 只有当超过 60% 的点都共面时，才认为是背景墙
                logger.info(f"🧱 Large planar wall detected ({ratio:.1%} points), removing safely.")
                pcd_objects = pcd_objects.select_by_index(inliers2, invert=True)
                save_debug_ply(pcd_objects, "06_wall_removed.ply", debug_dir)
            else:
                logger.info(f"🛡️ RANSAC detected a plane ({ratio:.1%}), but likely robot parts. Skipping wall removal.")
        except:
            pass

    # [Step 5] 聚类筛选 (Clustering)
    if len(pcd_objects.points) < 10:
        return points_np[idx_seed]

    # DBSCAN
    # 如果密度滤波有效，这里 eps 可以保持 0.04 或 0.03
    labels = np.array(pcd_objects.cluster_dbscan(eps=0.04, min_points=15, print_progress=False))
    max_label = labels.max()
    
    if max_label < 0:
        return points_np[idx_seed]

    # Visualization
    colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label >=0 else 1))
    colors[labels < 0] = 0 
    pcd_objects.colors = o3d.utility.Vector3dVector(colors[:, :3])
    save_debug_ply(pcd_objects, "07_clusters.ply", debug_dir)

    # 评分逻辑
    pts_objects_np = np.asarray(pcd_objects.points)
    seed_tree = scipy.spatial.cKDTree(seed_points)
    
    best_cluster_idx = -1
    max_score = -1
    
    for i in range(max_label + 1):
        c_mask = (labels == i)
        c_pts = pts_objects_np[c_mask]
        
        sample_pts = c_pts[::5] 
        dists, _ = seed_tree.query(sample_pts, k=1, distance_upper_bound=0.02)
        score = np.sum(dists != float('inf'))
        
        if score > max_score:
            max_score = score
            best_cluster_idx = i
            
    if best_cluster_idx != -1 and max_score > 0:
        logger.info(f"✅ Selected Cluster {best_cluster_idx} (Score: {max_score})")
        final_pcd = pcd_objects.select_by_index(np.where(labels == best_cluster_idx)[0])
    else:
        logger.warning("⚠️ No cluster matched seeds. Fallback to largest cluster.")
        counts = np.bincount(labels[labels>=0])
        best_cluster_idx = np.argmax(counts)
        final_pcd = pcd_objects.select_by_index(np.where(labels == best_cluster_idx)[0])

    # [Step 6] 最后清洗
    final_pcd, _ = final_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.2)
    save_debug_ply(final_pcd, "08_final.ply", debug_dir)
    
    return np.asarray(final_pcd.points)
    

def filter_point_cloud_by_masks_simple(ply_path, mask_dir, poses_path, intrinsics_path, device='cuda'):
    """简单版本的点云过滤（当 Open3D 不可用时）"""
    pcd = trimesh.load(str(ply_path))
    verts = np.asarray(pcd.vertices)
    
    cam_poses = np.load(poses_path)
    if os.path.exists(intrinsics_path):
        intrinsics = np.load(intrinsics_path)
        K = intrinsics[0]
    else:
        K = np.eye(3) 
    
    mask_files = sorted(list(Path(mask_dir).glob("*.png")))
    masks = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in mask_files[:10]]
    H, W = masks[0].shape if masks else (720, 1280)
    
    w2c_poses = [np.linalg.inv(pose) for pose in cam_poses[:10]]
    
    return filter_point_cloud_by_masks(verts, w2c_poses, masks, K, H, W, device)


def filter_point_cloud_by_masks(verts, cam_poses, masks, K, H, W, device='cuda'):
    logger.info("🧹 Filtering Point Cloud using Masks...")
    verts_t = torch.tensor(verts, dtype=torch.float32, device=device)
    num_points = verts_t.shape[0]
    
    keep_counts = torch.zeros(num_points, device=device)
    indices = np.linspace(0, len(cam_poses)-1, min(10, len(cam_poses)), dtype=int)
    
    for idx in indices:
        pose = torch.tensor(cam_poses[idx], dtype=torch.float32, device=device)
        mask = masks[idx]
        if mask.max() == 0: continue
        
        R = pose[:3, :3]
        t = pose[:3, 3]
        p_cam = torch.matmul(verts_t, R.T) + t
        p_cam_z = p_cam[:, 2] + 1e-6
        u = (K[0,0] * p_cam[:, 0] / p_cam_z) + K[0,2]
        v = (K[1,1] * p_cam[:, 1] / p_cam_z) + K[1,2]
        
        valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (p_cam_z > 0)
        u_int = u.long().clamp(0, W-1)
        v_int = v.long().clamp(0, H-1)
        
        mask_t = torch.tensor(mask, device=device)
        in_mask = mask_t[v_int, u_int] > 0
        keep_counts[valid_uv & in_mask] += 1
        
    valid_indices = keep_counts >= 1
    filtered_verts = verts_t[valid_indices]
    
    logger.info(f"Points filtered: {num_points} -> {filtered_verts.shape[0]}")
    return filtered_verts


def load_trimesh_robot_robust(chain, urdf_root_dir):
    """根据 Kinematics Chain 解析所有 Link 的 Visual Mesh"""
    robot_meshes = {}
    links = chain.get_links()

    for link in links:
        link_name = link.name
        if not link.visuals: continue

        combined_mesh = None
        for visual in link.visuals:
            geom_param = visual.geom_param
            if visual.geom_type == 'mesh':
                if isinstance(geom_param, tuple) and len(geom_param) >= 1:
                    filename = geom_param[0]
                    scale = geom_param[1] if len(geom_param) > 1 and geom_param[1] is not None else None
                else:
                    continue

                if not filename: continue
                mesh_path = (Path(urdf_root_dir).parent / filename).resolve()
                
                if str(filename).startswith("package://"):
                    pass 

                if not mesh_path.exists():
                    logger.warning(f"Mesh not found: {mesh_path}")
                    continue

                try:
                    sub_mesh = trimesh.load(str(mesh_path), force='mesh')
                    if visual.offset is not None:
                        origin_transform = visual.offset.get_matrix()
                        if origin_transform is not None:
                            transform_matrix = origin_transform.squeeze(0).cpu().numpy()
                            sub_mesh.apply_transform(transform_matrix)

                    if scale:
                        sub_mesh.apply_scale(scale)

                    if combined_mesh is None:
                        combined_mesh = sub_mesh
                    else:
                        combined_mesh = trimesh.util.concatenate(combined_mesh, sub_mesh)
                except Exception as e:
                    logger.warning(f"Failed to load mesh {mesh_path}: {e}")

        if combined_mesh:
            robot_meshes[link_name] = combined_mesh
            logger.info(f"Loaded mesh for link: {link_name} with {len(combined_mesh.vertices)} verts")

    return robot_meshes


# ==============================================================================
# 链式生长优化器 (Chain Growth Solver) - 升级版 (带内参修正)
# ==============================================================================
class ChainGrowSolver(torch.nn.Module):
    def __init__(self, urdf_path, device='cuda'):
        super().__init__()
        self.device = device
        
        with open(urdf_path, "rb") as f:
            self.chain = pk.build_chain_from_urdf(f.read()).to(dtype=torch.float32, device=device)
        
        self.joint_names = self.chain.get_joint_parameter_names()
        self.link_names = [l.name for l in self.chain.get_links()]
        self.n_dof = len(self.joint_names)
        
        self._load_meshes_indexed(urdf_path)
        
        # === 核心优化变量 ===
        # 1. 全局尺度 s (Global Scale)
        self.log_scale = torch.nn.Parameter(torch.zeros(1, device=device)) 
        
        # 2. 焦距修正因子 alpha (Z-Stretch Correction)
        # 用于修复因内参估计错误导致的 Z 轴透视畸变
        # 初始值为 0 (即 exp(0) = 1.0)
        self.log_z_stretch = torch.nn.Parameter(torch.zeros(1, device=device)) 
        
        # 3. 基座姿态 (Base Pose)
        self.base_rot_6d = torch.nn.Parameter(torch.tensor([1.,0,0,0,1,0], device=device))
        self.base_trans = torch.nn.Parameter(torch.zeros(3, device=device))
        
        # 4. 关节角度 (Joint Angles)
        self.joint_angles = torch.nn.Parameter(torch.zeros(1, self.n_dof, device=device))

    def _load_meshes_indexed(self, urdf_path):
        all_pts, all_ids = [], []
        urdf_root = Path(urdf_path).parent
        self.link_name_to_idx = {name: i for i, name in enumerate(self.link_names)}
        
        for link in self.chain.get_links():
            if not link.visuals or link.visuals[0].geom_type != 'mesh': continue
            visual = link.visuals[0]
            fname = str(visual.geom_param[0]).replace("package://", "")
            
            candidates = [urdf_root/fname, urdf_root.parent/fname, urdf_root.parent.parent/fname]
            mpath = next((p for p in candidates if p.exists()), None)
            
            if mpath:
                try:
                    m = trimesh.load(str(mpath), force='mesh')
                    if visual.offset: 
                        m.apply_transform(visual.offset.get_matrix().squeeze(0).cpu().numpy())
                    
                    pts, _ = trimesh.sample.sample_surface(m, 1500)
                    all_pts.append(torch.tensor(pts, dtype=torch.float32, device=self.device))
                    all_ids.append(torch.full((len(pts),), self.link_name_to_idx[link.name], device=self.device, dtype=torch.long))
                except Exception as e:
                    logger.warning(f"Failed to load mesh for {link.name}: {e}")
        
        self.raw_mesh_pts = all_pts
        self.raw_mesh_ids = all_ids
        logger.info(f"✅ Loaded {len(all_pts)} link meshes for chain growth optimization")

    def get_articulated_cloud(self):
        ret = self.chain.forward_kinematics(self.joint_angles)
        out_pts, out_ids = [], []
        
        current_mesh_idx = 0
        for link in self.chain.get_links():
            if not link.visuals or link.visuals[0].geom_type != 'mesh': continue
            
            if link.name in ret and current_mesh_idx < len(self.raw_mesh_pts):
                trans = ret[link.name].get_matrix()
                pts_local = self.raw_mesh_pts[current_mesh_idx]
                ids_local = self.raw_mesh_ids[current_mesh_idx]
                
                pts_homo = torch.cat([pts_local, torch.ones(len(pts_local), 1, device=self.device)], dim=1).unsqueeze(0)
                pts_world = torch.bmm(trans, pts_homo.transpose(1, 2)).transpose(1, 2)[0, :, :3]
                
                out_pts.append(pts_world)
                out_ids.append(ids_local)
                current_mesh_idx += 1
            
        if not out_pts: return None, None
        return torch.cat(out_pts, dim=0), torch.cat(out_ids, dim=0)


def masked_chamfer_distance(scan_pts, robot_pts, robot_ids, active_link_ids, device='cuda'):
    if robot_pts is None or robot_ids is None:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    mask = torch.isin(robot_ids, torch.tensor(active_link_ids, device=device, dtype=torch.long))
    active_robot = robot_pts[mask]
    
    if active_robot.shape[0] == 0: 
        return torch.tensor(0.0, device=device, requires_grad=True)

    N_scan = min(len(scan_pts), 3000)
    N_robot = min(len(active_robot), 3000)
    
    sample_scan = scan_pts[torch.randperm(len(scan_pts), device=device)[:N_scan]]
    sample_robot = active_robot[torch.randperm(len(active_robot), device=device)[:N_robot]]
    
    d_mat = torch.cdist(sample_robot.unsqueeze(0), sample_scan.unsqueeze(0))
    
    # 1. Robot -> Scan
    d_r2s = d_mat.min(dim=2)[0].mean()
    
    # 2. Scan -> Robot
    d_s2r_vals = d_mat.min(dim=1)[0]
    valid_mask = d_s2r_vals < 0.1
    d_s2r = d_s2r_vals[valid_mask].mean() if valid_mask.sum() > 0 else d_s2r_vals.mean()
    
    return d_r2s + d_s2r


def coarse_align_robust(scan_pts, solver, device='cuda'):
    """
    粗配准 - 将机械臂移动到点云重心，并强制 scale=1.0
    """
    # 🚨 核心修复：空值检查，防止 AttributeError
    if scan_pts is None or len(scan_pts) == 0:
        logger.error("❌ coarse_align_robust received EMPTY points. Skipping alignment.")
        return np.eye(4), 1.0

    logger.info(f"🔍 [Coarse Align] Aligning centers. Scan points: {len(scan_pts)}")

    # 获取归零姿态的机械臂模型点云
    with torch.no_grad():
        solver.joint_angles.data.fill_(0.0)
        robot_pts, _ = solver.get_articulated_cloud()

    if robot_pts is None:
        logger.error("❌ Failed to generate robot model points.")
        return np.eye(4), 1.0

    tgt_pts = robot_pts.cpu().numpy()

    # 计算重心
    src_center = scan_pts.mean(axis=0) # 此处已安全
    tgt_center = tgt_pts.mean(axis=0)

    # 强制 Scale=1.0 (反向标定策略：先不对齐尺度，靠 solver 优化)
    scale_fixed = 1.0
    translation = src_center - tgt_center

    T_coarse = np.eye(4)
    T_coarse[:3, 3] = translation

    # [Debug] 保存粗配准结果，验证位置是否重合
    debug_dir = Path("outputs/debug_coarse")
    debug_dir.mkdir(exist_ok=True, parents=True)

    try:
        # Green = Scan
        pcd_scan = trimesh.PointCloud(scan_pts, colors=[0, 255, 0, 255])
        pcd_scan.export(str(debug_dir / "debug_coarse_scan.ply"))

        # Red = Robot Model
        tgt_pts_aligned = tgt_pts + translation
        pcd_robot = trimesh.PointCloud(tgt_pts_aligned, colors=[255, 0, 0, 255])
        pcd_robot.export(str(debug_dir / "debug_coarse_robot.ply"))
        logger.info(f"💾 Coarse alignment visualization saved to {debug_dir}")
    except Exception as e:
        logger.warning(f"Failed to export coarse align debug: {e}")

    return T_coarse, scale_fixed


# ==============================================================================
# 全局帮助函数：应用各向异性拉伸、旋转和平移
# ==============================================================================
def transform_scan(scan_pts, s_log, alpha_log, R_6d, trans, device='cuda'):
    """
    scan_pts: (N, 3) tensor
    s_log: log scale
    alpha_log: log z-stretch (focal correction)
    R_6d: (6,) rotation representation
    trans: (3,) translation
    """
    s = torch.exp(torch.clamp(s_log, -10, 10))  # 防止指数爆炸
    alpha = torch.exp(torch.clamp(alpha_log, -2, 2))  # 限制 alpha 在合理范围内
    R = rotation_6d_to_matrix(R_6d.unsqueeze(0))[0]

    # 数值稳定性检查
    if torch.isnan(s) or torch.isinf(s) or s <= 0:
        s = torch.tensor(1.0, device=device)
    if torch.isnan(alpha) or torch.isinf(alpha) or alpha <= 0:
        alpha = torch.tensor(1.0, device=device)

    # 1. 反向标定核心：各向异性拉伸 (Anisotropic Z-Stretch)
    # alpha 用于修复因内参估计错误导致的 Z 轴透视畸变
    # 修复之前的维度报错：直接 cat，不要 unsqueeze
    stretch_vec = torch.cat([torch.ones(2, device=device), alpha])

    scan_stretched = scan_pts * stretch_vec

    # 2. 刚性对齐 + 全局缩放
    scan_final = (torch.matmul(scan_stretched, R.T) + trans.unsqueeze(0)) * s
    return scan_final


def optimize_pipeline_chain_growth(solver, scan_tensor, device='cuda'):
    """
    链式生长优化管道 (带内参自标定)
    """
    ordered_links = [l.name for l in solver.chain.get_links()]
    link_map = solver.link_name_to_idx
    link_indices = [link_map[n] for n in ordered_links if n in link_map]
    
    if not link_indices: return

    # --- Phase 1: Base Lock (Fixed Intrinsics) ---
    logger.info("🔒 [Phase 1] Anchoring Base (Alpha Locked)...")
    # 注意：此时不优化 log_z_stretch (alpha)，保持为 1.0
    opt_base = torch.optim.Adam([solver.log_scale, solver.base_rot_6d, solver.base_trans], lr=2e-3)
    active_ids = link_indices[:min(2, len(link_indices))]
    
    for _ in tqdm(range(150), desc="Base Lock", leave=False):
        opt_base.zero_grad()
        
        # 使用 detach 的 alpha (即 1.0)，不优化它
        scan_trans = transform_scan(scan_tensor, solver.log_scale, solver.log_z_stretch.detach(),
                                  solver.base_rot_6d, solver.base_trans, device=device)
        
        robot_pts, robot_ids = solver.get_articulated_cloud()
        if robot_pts is None: continue
        loss = masked_chamfer_distance(scan_trans, robot_pts, robot_ids, active_ids, device)
        loss.backward()
        opt_base.step()
        
    # --- Phase 2: Chain Growth (Fixed Intrinsics) ---
    logger.info("🌱 [Phase 2] Chain Growth...")
    for j_idx in range(solver.n_dof):
        logger.info(f"   Fitting Joint {j_idx+1}/{solver.n_dof}")
        
        solver.log_scale.requires_grad = False
        solver.base_rot_6d.requires_grad = False
        solver.base_trans.requires_grad = False
        
        optimizer = torch.optim.Adam([solver.joint_angles], lr=3e-2)
        target_ids = link_indices[:min(j_idx+3, len(link_indices))]
        
        best_loss = float('inf')
        best_cfg = None
        
        for _ in range(80):
            optimizer.zero_grad()
            
            # 依然不优化 alpha
            scan_trans = transform_scan(scan_tensor, solver.log_scale, solver.log_z_stretch.detach(),
                                      solver.base_rot_6d, solver.base_trans, device=device)
            
            robot_pts, robot_ids = solver.get_articulated_cloud()
            if robot_pts is None: continue
            loss = masked_chamfer_distance(scan_trans, robot_pts, robot_ids, target_ids, device)
            loss.backward()
            
            mask = torch.zeros_like(solver.joint_angles.grad)
            mask[0, j_idx] = 1.0
            if j_idx > 0: mask[0, j_idx-1] = 0.2
            solver.joint_angles.grad *= mask
            
            optimizer.step()
            solver.joint_angles.data.clamp_(-3.14, 3.14)
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_cfg = solver.joint_angles.data.clone()
                
        if best_cfg is not None:
            solver.joint_angles.data = best_cfg
        
    # --- Phase 3: Global Fine-tune with Self-Calibration (Unlock Alpha!) ---
    logger.info("🚀 [Phase 3] Global Fine-tune & Intrinsics Correction...")

    # 🔓 解锁所有参数，包括内参修正因子 alpha
    solver.log_scale.requires_grad = True
    solver.log_z_stretch.requires_grad = True # <--- Unlock!
    solver.base_rot_6d.requires_grad = True
    solver.base_trans.requires_grad = True

    # 降低学习率，加入 alpha 约束
    opt_global = torch.optim.Adam(solver.parameters(), lr=1e-4)

    # 收敛检测
    prev_loss = float('inf')
    patience = 0
    max_patience = 30
    min_improvement = 1e-6

    for iter_idx in tqdm(range(150), desc="Self-Calibration", leave=False):
        opt_global.zero_grad()

        # 此时优化 alpha
        scan_trans = transform_scan(scan_tensor, solver.log_scale, solver.log_z_stretch,
                                  solver.base_rot_6d, solver.base_trans, device=device)

        robot_pts, robot_ids = solver.get_articulated_cloud()
        if robot_pts is None: continue

        geom_loss = masked_chamfer_distance(scan_trans, robot_pts, robot_ids, link_indices, device)

        # 正则化：Alpha 应该接近 1.0 (log_alpha 接近 0)
        # 使用 Huber loss 减少对大偏差的惩罚，更鲁棒
        alpha_error = solver.log_z_stretch.abs()
        alpha_reg = torch.where(alpha_error < 0.5, alpha_error.pow(2), alpha_error).mean() * 0.05

        loss = geom_loss + alpha_reg
        loss.backward()
        opt_global.step()

        # 限制 alpha 范围 [0.5, 2.0]
        solver.log_z_stretch.data.clamp_(np.log(0.5), np.log(2.0))

        # 收敛检测
        current_loss = loss.item()
        if prev_loss - current_loss > min_improvement:
            patience = 0
        else:
            patience += 1

        if patience >= max_patience:
            logger.info(f"   Early stopping at iteration {iter_idx+1} (loss converged)")
            break

        prev_loss = current_loss

    # 更新收敛状态
    if hasattr(solver, 'calibration_stats'):
        solver.calibration_stats['converged'] = patience < max_patience


def main():
    # 1. Config & Workspace Resolution
    base_config_path = PROJECT_ROOT / "configs" / "base_config.yaml"
    base_cfg = OmegaConf.load(base_config_path)

    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base_cfg, cli_cfg)

    mode = cfg.pipeline.get("mode", "auto")
    manual_parent = cfg.pipeline.get("parent_workspace", None)

    ws_mgr = WorkspaceManager(str(base_config_path))
    required_step1_files = ["camera_poses.npy", "sparse_cloud.ply", "masks_robot"]

    try:
        workspace, parent_ws = ws_mgr.resolve_child(
            parent_requirements=required_step1_files,
            step_name="step2_calibrate",
            mode=mode,
            manual_parent_path=manual_parent
        )
    except Exception as e:
        logger.error(f"{e}")
        return

    logger.info(f"📂 Current Workspace: {workspace}")
    logger.info(f"📂 Parent Workspace (Step 1 Data): {parent_ws}")

    # 2. Load Data from Parent Workspace
    poses_path = parent_ws / "camera_poses.npy"
    ply_path = parent_ws / "sparse_cloud.ply"
    mask_dir = parent_ws / "masks_robot"

    cam_poses = np.load(poses_path)
    mask_files = sorted(list(mask_dir.glob("*.png")))
    max_frames = min(len(mask_files), len(cam_poses))
    train_indices = np.linspace(0, max_frames-1, min(50, max_frames), dtype=int)

    gt_masks = []
    train_poses = []

    logger.info(f"📸 Loading masks... (using {len(train_indices)} frames out of {max_frames} available)")
    for idx in train_indices:
        m = cv2.imread(str(mask_files[idx]), cv2.IMREAD_GRAYSCALE)
        gt_masks.append(m)
        train_poses.append(cam_poses[idx])

    gt_masks = np.stack(gt_masks) > 128

    logger.info("🔄 Inverting Camera Poses (C2W -> W2C)...")
    c2w_poses = np.stack(train_poses)
    w2c_poses = []
    for pose in c2w_poses:
        w2c_poses.append(np.linalg.inv(pose))
    train_poses = np.stack(w2c_poses)

    H, W = gt_masks.shape[1], gt_masks.shape[2]
    intrinsics_path = parent_ws / "intrinsics.npy"
    if intrinsics_path.exists():
        logger.info("Using MASt3R estimated intrinsics.")
        intrinsics = np.load(intrinsics_path)
        K = intrinsics[0].copy()
    else:
        logger.warning("Intrinsics file not found! Fallback to dummy K.")
        focal = 0.5 * W / np.tan(0.5 * 60 * np.pi / 180)
        K = np.array([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]])

    # 3. Load Robot Config
    step1_config = OmegaConf.load(parent_ws / "runtime_config.yaml")
    data_root = PROJECT_ROOT / step1_config.data.root_dir
    urdf_rel = cfg.robot.get("urdf_rel_path", step1_config.robot.get("urdf_rel_path", "robot/rllab_xarm_urdf_combo/xarm6_with_gripper_v1.urdf"))
    urdf_path = data_root / urdf_rel

    align_pose_path = data_root / "config" / "align_pose.json"
    with open(align_pose_path) as f:
        joint_cfg = json.load(f)

    # 4. Initialize Aligner
    device = "cuda" if torch.cuda.is_available() else "cpu"
    aligner = HybridAligner(str(urdf_path), align_cfg=cfg.calibration, device=device)
    mesh_dict = load_trimesh_robot_robust(aligner.chain, urdf_path)
    aligner.inject_mesh_data(mesh_dict, num_samples_per_link=512)

    # --------------------------------------------------
    # Phase 2/3: Fine Optimization with Self-Calibration
    # --------------------------------------------------
    if cfg.calibration.fine.enable:
        use_chain_growth = cfg.calibration.fine.get("use_chain_growth", False)
        
        if use_chain_growth:
            logger.info("🌱 [Chain Growth Mode] Starting Hierarchical Optimization...")
            
            # 1. 空间雕刻
            logger.info("🧹 [Step 1] Segmentation...")
            scan_pts = segment_robot_from_scene(
                str(ply_path), str(mask_dir), str(poses_path),
                str(intrinsics_path) if intrinsics_path.exists() else None,
                device=device
            )

            # 🚨 检查是否分割成功
            if scan_pts is None or len(scan_pts) < 50:
                logger.error("❌ Segmentation failed (too few points). Aborting pipeline.")
                logger.error(f"👉 Check debug files in: {str(ply_path.parent / 'debug_segmentation')}")
                return # 直接退出，避免后面报错

            # 2. 初始化 Solver
            logger.info("🔧 [Step 2] Initializing Chain Growth Solver...")
            solver = ChainGrowSolver(str(urdf_path), device=device)
            
            # 3. 粗配准 (强制 Scale=1.0)
            T_ransac, s_init = coarse_align_robust(scan_pts, solver, device=device)
            
            # 预处理点云：应用粗配准
            scan_pre = (scan_pts - scan_pts.mean(axis=0)) * s_init + scan_pts.mean(axis=0)
            scan_pre = (np.hstack((scan_pre, np.ones((len(scan_pre),1)))) @ T_ransac.T)[:, :3]
            scan_tensor = torch.tensor(scan_pre, dtype=torch.float32, device=device)
            
            # 4. 链式生长优化 (带内参自标定)
            optimize_pipeline_chain_growth(solver, scan_tensor, device=device)

            # === [修复代码开始] ===
            # 在 main 函数中重新定义 link_indices，因为它是局部变量
            ordered_links = [l.name for l in solver.chain.get_links()]
            link_map = solver.link_name_to_idx
            link_indices = [link_map[n] for n in ordered_links if n in link_map]
            # === [修复代码结束] ===

            # 5. 获取结果
            s_fine = torch.exp(solver.log_scale).item()
            alpha_fine = torch.exp(solver.log_z_stretch).item() # 获取焦距修正系数
            
            T_fine_R = rotation_6d_to_matrix(solver.base_rot_6d.unsqueeze(0))[0].detach().cpu().numpy()
            T_fine_t = solver.base_trans.detach().cpu().numpy()
            q_final = solver.joint_angles.detach().cpu().numpy()[0]
            
            total_scale = s_init * s_fine

            # 计算标定质量指标
            with torch.no_grad():
                final_scan = transform_scan(scan_tensor, solver.log_scale, solver.log_z_stretch,
                                          solver.base_rot_6d, solver.base_trans, device=device)
                final_robot_pts, final_robot_ids = solver.get_articulated_cloud()
                if final_robot_pts is not None:
                    final_loss = masked_chamfer_distance(final_scan, final_robot_pts,
                                                       final_robot_ids, link_indices, device).item()
                    logger.info(f"✅ Final Alignment Loss: {final_loss:.6f}")
                else:
                    final_loss = float('nan')

            logger.info(f"✅ Final Scale (s): {total_scale:.4f}")
            logger.info(f"✅ Intrinsics Z-Stretch (alpha): {alpha_fine:.4f} (Ideal=1.0, deviation: {abs(alpha_fine-1.0):.4f})")
            logger.info(f"✅ Final Joints: {q_final}")

            # 保存标定统计信息 (注意：这里的变量需要在 Phase 3 中定义，这里先占位)
            solver.calibration_stats = {
                'final_loss': final_loss,
                'alpha_deviation': abs(alpha_fine - 1.0),
                'scale_confidence': 1.0 / (1.0 + abs(np.log(total_scale))),  # 尺度置信度
                'converged': True  # 会在 Phase 3 中更新
            }
            
            # 保存到 aligner
            joint_names = aligner.chain.get_joint_parameter_names()
            q_list = [joint_cfg.get(n, 0.0) for n in joint_names]
            q_init_tensor = torch.tensor([q_list], device=device)
            
            aligner.log_scale.data = torch.tensor(np.log(total_scale), device=device)
            aligner.base_rot_6d.data = solver.base_rot_6d.data
            aligner.base_trans.data = solver.base_trans.data
            aligner.delta_q.data = torch.tensor(q_final, device=device).unsqueeze(0) - q_init_tensor
            
            # 将 alpha 存入 aligner (Monkey Patching) 以便后续使用
            aligner.z_stretch_val = alpha_fine
            logger.info("✅ Chain Growth optimization completed!")
        
        else:
            logger.info("🚀 [Rendering Mode] Code path exists but is not active in this fixed script.")

    # ==============================================================================
    # 🐞 终极调试：全场景分割可视化
    # ==============================================================================
    logger.info("🐞 Starting Debug Export (Full Scene Visualization)...")

    s_final, R_final, t_final = aligner.get_transform_params()
    alpha_final = getattr(aligner, 'z_stretch_val', 1.0) # 获取 alpha
    
    joint_names = aligner.chain.get_joint_parameter_names()
    q_list = [joint_cfg.get(n, 0.0) for n in joint_names]
    q_init_tensor = torch.tensor([q_list], device=device)

    use_chain_growth = cfg.calibration.fine.get("use_chain_growth", False)
    if use_chain_growth:
        q_final_tensor = q_init_tensor + aligner.delta_q
    else:
        q_final_tensor = q_init_tensor

    with torch.no_grad():
        all_verts = []
        all_colors = []

        # ------------------------------------------------------------------
        # Part A: 处理全场景点云 (应用 Alpha 修正!)
        # ------------------------------------------------------------------
        try:
            logger.info("🎨 Processing Full Scene for visualization...")
            full_pcd = trimesh.load(str(ply_path))
            full_verts = np.asarray(full_pcd.vertices)

            if len(full_verts) > 100000:
                choice = np.random.choice(len(full_verts), 100000, replace=False)
                full_verts = full_verts[choice]

            # 颜色设置 (简单全灰)
            colors = np.full((len(full_verts), 4), [200, 200, 200, 255], dtype=np.uint8)

            # 变换: 1. Stretch (修正内参) -> 2. Scale -> 3. Rigid Align
            full_verts_t = torch.tensor(full_verts, dtype=torch.float32, device=device)
            
            # 应用反向标定得到的修正因子!
            stretch_vec = torch.tensor([1.0, 1.0, alpha_final], device=device)
            full_verts_corrected = full_verts_t * stretch_vec
            
            full_verts_scaled = full_verts_corrected * s_final
            full_verts_aligned = torch.matmul(full_verts_scaled, R_final.T) + t_final.unsqueeze(0)
            
            all_verts.append(full_verts_aligned.cpu().numpy())
            all_colors.append(colors)

        except Exception as e:
            logger.error(f"Error processing scene cloud: {e}")

        # ------------------------------------------------------------------
        # Part B: 处理 CAD 模型 (蓝色)
        # ------------------------------------------------------------------
        cloud_robot_local = aligner.get_robot_point_cloud(q_final_tensor)
        if cloud_robot_local is not None:
            robot_verts = cloud_robot_local[0].detach().cpu().numpy()
            robot_colors = np.tile([0, 0, 255, 255], (len(robot_verts), 1)).astype(np.uint8)
            
            all_verts.append(robot_verts)
            all_colors.append(robot_colors)

        # ------------------------------------------------------------------
        # Part C: 手动合并并保存
        # ------------------------------------------------------------------
        if all_verts:
            final_verts = np.vstack(all_verts)
            final_colors = np.vstack(all_colors)
            
            combined_pcd = trimesh.PointCloud(vertices=final_verts, colors=final_colors)
            save_path = workspace / "debug_full_scene_corrected.ply"
            combined_pcd.export(str(save_path))
            logger.info(f"💾 Full Scene Debug saved to: {save_path}")

    # 5. Save Results
    s, R, t = aligner.get_transform_params()
    T_final = np.eye(4)
    T_final[:3, :3] = R.detach().cpu().numpy()
    T_final[:3, 3] = t.detach().cpu().numpy()

    if use_chain_growth:
        q_list = [joint_cfg.get(n, 0.0) for n in joint_names]
        q_init_tensor = torch.tensor([q_list], device=device)
        final_q = (q_init_tensor + aligner.delta_q).detach().cpu().numpy().tolist()
        optimized_joints = dict(zip(joint_names, final_q[0]))
    else:
        optimized_joints = joint_cfg

    res = {
        "scale": s.item(),
        "z_stretch_correction": getattr(aligner, 'z_stretch_val', 1.0), # 输出修正系数
        "base_transform": T_final.tolist(),
        "optimized_joints": optimized_joints,
        "parent_workspace": str(parent_ws)
    }

    # 添加标定统计信息（如果有的话）
    if hasattr(aligner, 'calibration_stats'):
        res["calibration_stats"] = aligner.calibration_stats
        logger.info(f"📊 Calibration Stats: Loss={res['calibration_stats']['final_loss']:.6f}, "
                   f"Alpha Dev={res['calibration_stats']['alpha_deviation']:.4f}, "
                   f"Converged={res['calibration_stats']['converged']}")

    with open(workspace / "calibration_meta.json", "w") as f:
        json.dump(res, f, indent=2)

    logger.info(f"✅ Calibration saved to {workspace}/calibration_meta.json")

if __name__ == "__main__":
    main()