"""
src/semiff/solvers/aligner.py
混合式自标定核心模块：结合 3D RANSAC 粗配准与 2D Differentiable Rendering 精配准
"""

import torch
import torch.nn as nn
import numpy as np
import pytorch_kinematics as pk
import trimesh
import logging
import os
from omegaconf import DictConfig

from semiff.engine.render import DifferentiableRasterizer
from semiff.engine.math_utils import rotation_6d_to_matrix, transform_points, chamfer_distance

# Try to import open3d, fallback if not available
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    logging.warning("Open3D not available. RANSAC coarse alignment will be disabled.")

logger = logging.getLogger("ALIGNER")

class HybridAligner(nn.Module):
    def __init__(self, urdf_path, align_cfg: DictConfig, device='cuda'):
        """
        Args:
            urdf_path: URDF 文件路径
            align_cfg: 对应 config.yaml 中的 'calibration' 部分
            device: 计算设备
        """
        super().__init__()
        self.device = device
        self.cfg = align_cfg
        self.urdf_path = urdf_path

        # 1. Kinematics Chain
        with open(urdf_path, "rb") as f:
            self.chain = pk.build_chain_from_urdf(f.read()).to(dtype=torch.float32, device=device)
            
        self.joint_names = self.chain.get_joint_parameter_names()
        self.n_dof = len(self.joint_names)

        # 2. Mesh Storage (Loaded later)
        self.link_meshes = {} # {link_name: verts_tensor} - 存储原始顶点
        self.link_faces = {} # {link_name: faces_tensor} - 存储面片（用于渲染）
        self.sampled_indices = {} # {link_name: indices} - 存储预先随机采样的索引

        # 3. Parameters (初始化)
        # 如果有 Coarse 阶段，这些会被覆盖；否则使用 Config 中的 Initial Guess
        init_trans = self.cfg.initial_guess.get("translate", [0,0,1])
        init_scale = self.cfg.initial_guess.get("scale", 1.0)

        self.log_scale = nn.Parameter(torch.tensor(float(np.log(init_scale)), device=device))
        self.base_trans = nn.Parameter(torch.tensor(init_trans, dtype=torch.float32, device=device))
        self.base_rot_6d = nn.Parameter(torch.tensor([1., 0, 0, 0, 1, 0], device=device))

        # C. 关节角度修正 (Joint Optimization)
        # 这是为了解决 "关节角度不一致" 的关键
        self.delta_q = nn.Parameter(torch.zeros(1, self.n_dof, device=device))

        self.renderer = None 

    def load_meshes(self, urdf_root_dir):
        """
        Parse URDF meshes using trimesh and load to GPU.
        Simplified loader: assumes URDF file names match link names or standard structure.
        """
        # 注意：这里需要根据具体的 URDF 结构编写 Mesh 加载逻辑
        # 下面是一个通用的启发式加载器
        # 实际使用中，pytorch_kinematics 的 chain.get_link_names() 可以帮助我们
        logger.info(f"🔩 Loading meshes from {urdf_root_dir}...")
        
        # 解析 URDF 获取 mesh 路径 (借助 urdfpy 或 trimesh.load)
        # 这里为了不引入新依赖，假设我们手动扫描 mesh 文件夹
        # 假设 mesh 都在 robot/visual/ 下，且文件名包含 link 名
        # *这是一个需要根据你实际文件结构调整的地方*
        
        # 备选方案: 使用 trimesh 加载整个 URDF scene
        try:
            scene = trimesh.load(self.urdf_path)
            # trimesh load urdf returns a Scene with geometry attached to nodes
            # We need to map geometry to link names
            if isinstance(scene, trimesh.Scene):
                 for name, geom in scene.geometry.items():
                    # 简化：假设所有 geometry 都合并处理，或者根据 name 匹配 link
                    # 在 Step 2 脚本中我们会处理具体的 Mesh 加载
                    pass
        except Exception as e:
            logger.warning(f"Trimesh direct load failed: {e}")

    def inject_mesh_data(self, mesh_dict, num_samples_per_link=512):
        """
        注入 Mesh 数据，并预先生成采样索引以节省显存
        mesh_dict: {link_name: trimesh.Trimesh}
        num_samples_per_link: 每个 link 采样的点数
        """
        for name, mesh in mesh_dict.items():
            if len(mesh.vertices) == 0: continue
            v = torch.from_numpy(mesh.vertices).float().to(self.device)
            f = torch.from_numpy(mesh.faces).int().to(self.device) if len(mesh.faces) > 0 else None
            
            # 随机下采样索引，保留梯度传播能力
            # 我们不存储采样后的点，而是存储索引，因为我们要变换的是 v
            n_v = v.shape[0]
            if n_v > num_samples_per_link:
                # 随机选点
                idx = torch.randperm(n_v, device=self.device)[:num_samples_per_link]
            else:
                idx = torch.arange(n_v, device=self.device)
                
            self.link_meshes[name] = v  # 原始顶点
            if f is not None:
                self.link_faces[name] = f  # 面片（用于渲染）
            self.sampled_indices[name] = idx  # 采样索引
            
        logger.info(f"✅ [Aligner] Injected meshes. Optimization will use dynamic sampling.")

    # ================= PHASE 1: 3D Coarse (CPU/Open3D) =================
    
    def run_coarse_alignment(self, visual_ply_path, joint_cfg, filter_mask_fn=None):
        if not self.cfg.coarse.enable:
            logger.info("⏭️ [Phase 1] Coarse alignment disabled by config.")
            return None

        logger.info("🏗️ [Phase 1] Starting 3D RANSAC...")
        num_samples = self.cfg.coarse.num_samples_physical
        thresh = self.cfg.coarse.ransac_threshold

        # A. Sample Physical Robot (Target)
        phy_pts = self._sample_physical_robot(joint_cfg, num_samples=num_samples)

        # B. Load Visual Cloud (Source)
        vis_pcd = o3d.io.read_point_cloud(str(visual_ply_path))
        vis_pts = np.asarray(vis_pcd.points)

        # C. Filter Background (Optional but recommended)
        if self.cfg.coarse.visual_filter.enable and filter_mask_fn:
            mask_indices = filter_mask_fn(vis_pts)
            vis_pts = vis_pts[mask_indices]
            logger.info(f"Filtered points: {len(vis_pcd.points)} -> {len(vis_pts)}")

        if len(vis_pts) < 100:
            logger.warning("⚠️ Too few visual points for RANSAC. Using raw cloud.")
            vis_pts = np.asarray(vis_pcd.points)

        # D. RANSAC
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(phy_pts) # Physical
        tgt = o3d.geometry.PointCloud()
        tgt.points = o3d.utility.Vector3dVector(vis_pts) # Visual (MASt3R)
        
        # Note: We want T_sim2real: Physical -> Visual
        # So Physical is Source, Visual is Target
        
        src.estimate_normals()
        tgt.estimate_normals()
        
        # FPFH Features
        voxel_size = thresh * 0.5  # 自适应特征半径
        src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(src, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
        tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(tgt, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src, tgt, src_fpfh, tgt_fpfh,
            mutual_filter=True,
            max_correspondence_distance=thresh,  # 使用配置
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(thresh)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )

        logger.info(f"✅ RANSAC Fitness: {result.fitness:.4f}")
        self._set_params_from_matrix(result.transformation)
        return result.transformation

    def _sample_physical_robot(self, joint_cfg, num_samples=10000):
        """Use FK to place meshes and sample points"""
        # joint_cfg: dict {name: val}
        # Build tensor batch for FK
        # Note: We need to match joint names. 
        # For simplicity, we assume joint_cfg keys match URDF joint names or we iterate links.
        
        # 简单处理：我们不通过 FK 采样，而是假设 t=0 时刻 Pose 已知
        # 正确做法：利用 self.chain.forward_kinematics 算出每个 link 的 transform
        # 然后把 self.link_meshes 的点变换过去
        
        th_q = torch.tensor([list(joint_cfg.values())], device=self.device)
        # joint names matching is crucial here, assuming ordered values for now or robust mapping
        # Ideally: chain.forward_kinematics(th_q) where th_q matches chain.get_joint_parameter_names()
        
        # 临时 Hack: 如果 joint_cfg 是 dict, 转换为 list
        joint_names = self.chain.get_joint_parameter_names()
        q_list = [joint_cfg.get(n, 0.0) for n in joint_names]
        th_q = torch.tensor([q_list], device=self.device)

        ret = self.chain.forward_kinematics(th_q)
        
        all_pts = []
        total_verts = sum([len(m[0]) for m in self.link_meshes.values()])
        
        for name, (v_gpu, _) in self.link_meshes.items():
            if name not in ret: continue
            
            # Count proportional samples
            n_samp = int(num_samples * (len(v_gpu) / total_verts))
            if n_samp < 10: n_samp = 10
            
            # Random sample indices
            idx = torch.randperm(len(v_gpu))[:n_samp]
            v_sample = v_gpu[idx] # [N, 3]
            
            # Transform to Base
            trans = ret[name].get_matrix()[0] # [4, 4]
            v_homo = torch.cat([v_sample, torch.ones(len(v_sample), 1, device=self.device)], dim=1)
            v_base = (trans @ v_homo.T).T[:, :3]
            
            all_pts.append(v_base.cpu().numpy())
            
        return np.vstack(all_pts)

    def _set_params_from_matrix(self, T):
        T_ten = torch.tensor(T, dtype=torch.float32, device=self.device)
        scale = torch.norm(T_ten[:3, 0]) # 简单估算
        self.log_scale.data = torch.log(scale)
        
        R_mat = T_ten[:3, :3] / scale
        # Gram-Schmidt to ensure ortho (RANSAC with scaling might distort)
        u, s, v = torch.svd(R_mat)
        R_ortho = u @ v.T
        
        r6d = R_ortho[:, :2].T.flatten() # 6D representation
        self.base_rot_6d.data = r6d
        self.base_trans.data = T_ten[:3, 3]

    # ================= PHASE 2: 2D Fine (DiffRender) =================

    def get_transform(self):
        """获取变换参数（保持向后兼容）"""
        return self.get_transform_params()
    
    def get_transform_params(self):
        """获取变换参数"""
        s = torch.exp(self.log_scale)
        R = rotation_6d_to_matrix(self.base_rot_6d.unsqueeze(0))[0]
        t = self.base_trans
        return s, R, t

    def get_robot_point_cloud(self, q_current):
        """
        生成当前参数下的机器人 3D 点云 (Differentiable)
        
        Args:
            q_current: [1, n_dof] 关节角度
        
        Returns:
            cloud_urdf: [1, M, 3] 机器人点云（在 Robot Base Frame 下）
        """
        # 1. 正向运动学 FK
        ret = self.chain.forward_kinematics(q_current)
        
        sampled_points = []
        
        for name, v_all in self.link_meshes.items():
            if name not in ret: continue
            
            # 取出预选的采样点
            idx = self.sampled_indices[name]
            v_batch = v_all[idx]  # [M, 3]
            
            # FK 变换: Local -> Robot Base Frame
            trans = ret[name].get_matrix()  # [1, 4, 4]
            
            ones = torch.ones(v_batch.shape[0], 1, device=self.device)
            v_homo = torch.cat([v_batch, ones], dim=1).unsqueeze(0)  # [1, M, 4]
            
            # [1, 4, 4] @ [1, M, 4].T -> [1, 4, M] -> [1, M, 4] -> [1, M, 3]
            v_base = torch.bmm(trans, v_homo.transpose(1, 2)).transpose(1, 2)[:, :, :3]
            
            sampled_points.append(v_base)
            
        if not sampled_points:
            return None
            
        return torch.cat(sampled_points, dim=1)  # [1, Total_Points, 3]

    def forward(self, q_init, obs_cloud=None, cam_poses=None, K=None, H=None, W=None):
        """
        前向传播：支持两种模式
        1. Chamfer Distance 模式：q_init, obs_cloud 不为 None
        2. 渲染模式：cam_poses, K, H, W 不为 None（保持向后兼容）
        
        Args:
            q_init: [1, n_dof] 初始关节角度
            obs_cloud: [1, N, 3] MASt3R 观测点云（已过滤背景），用于 Chamfer Loss
            cam_poses: [B, 4, 4] World-to-Camera，用于渲染
            K: [B, 3, 3] 内参矩阵
            H, W: 图像尺寸
        """
        # 模式1: Chamfer Distance 优化
        if obs_cloud is not None:
            return self._forward_chamfer(q_init, obs_cloud)
        
        # 模式2: 渲染模式（保持向后兼容）
        elif cam_poses is not None and K is not None and H is not None and W is not None:
            return self._forward_render(q_init, cam_poses, K, H, W)
        
        else:
            raise ValueError("Either obs_cloud or (cam_poses, K, H, W) must be provided")
    
    def _forward_chamfer(self, q_init, obs_cloud):
        """
        计算 3D Chamfer Loss
        Args:
            q_init: 初始关节角度 [1, n_dof]
            obs_cloud: MASt3R 观测点云 [1, N, 3] (已过滤背景)
        Returns:
            loss: scalar tensor
        """
        # 1. 准备参数
        s, R_base, t_base = self.get_transform_params()
        
        # 2. 关节角度自适应: q = q_init + delta_q
        q_current = q_init + self.delta_q
        
        # 3. 生成虚拟机器人点云 (Ground Truth Geometry, in Robot Base Frame)
        # P_urdf(q)
        cloud_urdf = self.get_robot_point_cloud(q_current)  # [1, M, 3]
        
        if cloud_urdf is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 4. 变换观测点云 (MASt3R -> Robot Base Frame)
        # 我们寻找 T_base 和 s 使得： T_base * s * P_mast3r ~= P_urdf
        # 公式: P_obs_aligned = (P_mast3r * s) @ R_base.T + t_base
        # 注意：R_base, t_base 是 World(MASt3R) -> Robot 的变换
        
        obs_cloud_scaled = obs_cloud * s
        
        # 使用 R_base, t_base 作为 World(MASt3R) -> Robot 的变换参数
        # R_base 是 [3, 3]，obs_cloud_scaled 是 [1, N, 3]
        # P_obs_aligned = (P_mast3r * s) @ R_base.T + t_base
        obs_cloud_aligned = torch.matmul(obs_cloud_scaled, R_base.T) + t_base.unsqueeze(0)
        
        # 5. 计算 Loss
        loss = chamfer_distance(obs_cloud_aligned, cloud_urdf)
        
        return loss
    
    def _forward_render(self, joint_cfg_tensor, cam_poses, K, H, W):
        """
        Render the robot at current estimate pose (保持向后兼容)
        cam_poses: [B, 4, 4] World-to-Camera
        """
        if self.renderer is None:
            self.renderer = DifferentiableRasterizer(H, W, device=self.device)
            
        s, R_sim2real, t_sim2real = self.get_transform_params()
        
        # 1. FK (Batch=1 for static robot)
        ret = self.chain.forward_kinematics(joint_cfg_tensor)
        
        all_verts = []
        all_faces = []
        offset = 0
        
        for name, v_loc in self.link_meshes.items():
            if name not in ret: continue
            tg = ret[name].get_matrix()  # [1, 4, 4]
            
            # Link -> Physical Base (使用全部顶点，不采样，以保证渲染质量)
            ones = torch.ones(len(v_loc), 1, device=self.device)
            v_homo = torch.cat([v_loc, ones], dim=1)
            v_phys = (tg @ v_homo.T).transpose(1, 2)[:, :, :3]  # [1, N, 3]
            
            # Physical Base -> Visual World
            v_vis = (v_phys @ R_sim2real.T + t_sim2real) * s
            
            # Repeat for batch size of cameras
            B = cam_poses.shape[0]
            v_vis_batch = v_vis.repeat(B, 1, 1)  # [B, N, 3]
            
            all_verts.append(v_vis_batch)
            
            # 添加 faces（如果存在）
            if name in self.link_faces:
                f_loc = self.link_faces[name]
                all_faces.append(f_loc + offset)
            
            offset += len(v_loc)
            
        if not all_verts: 
            return None
        
        mesh_verts = torch.cat(all_verts, dim=1)  # [B, Total_N, 3] World Space
        
        if not all_faces:
            logger.warning("⚠️ No faces available for rendering. Returning None.")
            return None
        
        mesh_faces = torch.cat(all_faces, dim=0)

        # 2. Render (传递相机位姿，让render函数内部处理World->Camera->Clip变换)
        masks = self.renderer.render(mesh_verts, mesh_faces, K, cam_poses)

        # Debug: 检查渲染结果
        if masks is not None and torch.rand(1) < 0.01:
            logger.info(f"🤖 Aligner Debug - Rendered masks shape: {masks.shape}, stats: min={masks.min():.4f}, max={masks.max():.4f}, mean={masks.mean():.4f}")

        return masks