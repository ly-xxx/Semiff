#!/usr/bin/env python3
"""
Semiff Main Entry: Real-to-Sim-to-Real Pipeline
"""
import hydra
from pathlib import Path
import numpy as np
from omegaconf import DictConfig

# 引入我们定义的模块
from semiff.core.io import VideoReader, RobotLogger
from semiff.perception.mast3r_wrapper import MASt3RWrapper
from semiff.perception.sam2_wrapper import SAM2Wrapper
from semiff.calibration.robot_aligner import align_visual_to_robot
from semiff.calibration.space_trans import RigidTransform, apply_transform_to_cameras
from semiff.geometry.meshing import Mesher
from semiff.geometry.decomposition import ColliderBuilder
from semiff.rendering.dataset_prep import NerfstudioConverter, estimate_intrinsics
from semiff.core.logger import get_logger

logger = get_logger("semiff_main")

@hydra.main(config_path="src/semiff/config", config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    logger.info("🚀 Starting Semiff Pipeline...")
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Stage 1: Data Ingestion ===
    logger.info(">>> [Stage 1] Loading Data")
    # 加载视频
    with VideoReader(cfg.data.video_path) as video:
        frames, timestamps = video.get_frames(step=cfg.pipeline.keyframe_interval)
        # 保存视频元数据以供后续使用
        video_meta = {'width': video.width, 'height': video.height, 'fps': video.fps}

    # 加载机器人日志 (可选，用于对齐)
    if cfg.data.robot_logs:
        robot_log = RobotLogger(cfg.data.robot_logs)
        # 获取与视频帧对应的关节角度
        joints = robot_log.get_interpolated_joints(timestamps)
        logger.info(f"Aligned {len(joints)} joint states to video frames.")

    # === Stage 2: Perception (Geometry) ===
    logger.info(">>> [Stage 2] Running MASt3R for Sparse Reconstruction")
    mast3r = MASt3RWrapper(device=cfg.device)
    # 运行 MASt3R 得到位姿和点云
    poses, scene_cloud = mast3r.run(frames)

    # 保存中间结果
    np.save(output_dir / "poses.npy", poses)
    # TODO: 保存 scene_cloud 为 .ply (需要 open3d)

    # === Stage 3: Perception (Semantics) ===
    logger.info(">>> [Stage 3] Running SAM 2 for Segmentation")
    sam2 = SAM2Wrapper(cfg)
    mask_paths = sam2.run(
        video_path=cfg.data.video_path,
        output_dir=output_dir,
        scene_cloud=scene_cloud # 传入点云用于辅助 Prompting
    )

    logger.info("✅ Perception stages completed.")

    # === Stage 4: Calibration (Sim2Real Alignment) ===
    logger.info(">>> [Stage 4] Aligning Coordinate Systems...")

    # 假设我们选择视频的第 10 帧作为对齐参考帧 (此时机器人姿态较好)
    ref_frame_idx = min(10, len(timestamps) - 1)  # 确保索引不越界
    ref_time = timestamps[ref_frame_idx]

    # 分支逻辑：Mode 1 (有日志) vs Mode 2 (无日志)
    if cfg.data.robot_logs:
        logger.info("Mode 1: Log-based Hard Alignment")
        # 注意：这里我们需要仅仅提取属于机器人的点云
        # 在生产环境中，你需要利用 SAM2 的 mask (mask_paths) 来索引 scene_cloud
        # 这里暂时传入完整的 cloud (假设场景主要是机器人，或者依靠 ICP 的鲁棒性)
        T_align = align_visual_to_robot(
            visual_cloud=scene_cloud, # TODO: Filter this with robot mask!
            robot_mask=None,
            robot_urdf=cfg.data.robot_urdf,
            robot_logs=cfg.data.robot_logs,
            timestamp=ref_time
        )
    else:
        logger.info("Mode 2: Visual-only Alignment (No Logs)")
        from semiff.calibration.solver import RobotOptimizer

        # 初始化优化器
        solver = RobotOptimizer(urdf_path=cfg.data.robot_urdf, device=cfg.device)

        # 运行优化 (反算关节角 + 基座变换)
        # 注意：这里需要传入这一帧的视觉点云
        # 且需要处理 mask 过滤 (目前 solver.optimize 接收的是 target_cloud)
        best_q, T_base = solver.optimize(target_cloud=scene_cloud)

        # 将 T_base 转为 RigidTransform
        T_align = RigidTransform(T_base)

    # 保存变换矩阵
    np.save(output_dir / "T_world.npy", T_align.matrix)
    logger.info(f"Alignment Transform:\n{T_align}")

    # 应用变换到所有相机位姿 (转到 URDF 坐标系)
    aligned_poses = apply_transform_to_cameras(poses, T_align)
    np.save(output_dir / "poses_aligned.npy", aligned_poses)

    logger.info("✅ Calibration completed.")

    # === Stage 5: Geometry Asset Generation ===
    logger.info(">>> [Stage 5] Generating Physics Assets...")

    # 1. 过滤点云：只保留物体
    # 在生产级代码中，这里应该使用 3D Mask 或 投影 2D Mask 来过滤
    # 这里做个简单的 Mock，假设 aligned_cloud 已经被裁剪或者是物体的
    # 实际：mesher.clean_cloud(scene_cloud, mask_indices=...)

    mesher = Mesher()
    mesh_path = mesher.run(scene_cloud, output_path=output_dir / "assets" / "object_raw.obj")

    collider = ColliderBuilder()
    collision_path = collider.decompose(mesh_path, output_path=output_dir / "assets" / "object_collision.obj")

    # === Stage 6: Rendering Dataset Prep ===
    logger.info(">>> [Stage 6] Preparing Background Training Data...")

    # 估算内参 (因为 MASt3R Wrapper 目前没有返回精确内参)
    intrinsics = estimate_intrinsics(video_meta['width'], video_meta['height']) # 从 VideoReader 获取

    ns_converter = NerfstudioConverter(output_dir=output_dir / "nerfstudio")
    ns_converter.process(
        frames=frames,
        masks_dir=mask_paths['object_masks'], # 来自 SAM2
        poses=aligned_poses,                  # 来自 Calibration
        intrinsics=intrinsics
    )

    logger.info("✅ All assets prepared. Ready for Warp Simulation & Splatfacto training.")
    logger.info(f"Check outputs in {output_dir}")

if __name__ == "__main__":
    main()