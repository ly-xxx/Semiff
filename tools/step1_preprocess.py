import argparse
import json
import numpy as np
import logging
import cv2
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

# ==================== 1. 核心路径配置 ====================
# 动态获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

# >>> 数据集名称 <<<
DATASET_NAME = "example_01" 
# =======================================================

from semiff.core.workspace import WorkspaceManager
try:
    from semiff.perception.sam2_wrapper import SAM2Wrapper
except ImportError:
    pass 
try:
    from semiff.perception.mast3r_wrapper import MASt3RWrapper
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Step1")

# ==================== 辅助类 ====================
class FFmpegWriter:
    def __init__(self, filename, width, height, fps):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        self.cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', f'{fps}',
            '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'fast', '-crf', '23', str(filename)
        ]
        self.process = subprocess.Popen(
            self.cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )

    def write(self, frame):
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass

    def release(self):
        if self.process:
            self.process.stdin.close()
            self.process.wait()

# ==================== 主流程 ====================
def run_step1():
    # -----------------------------------------------------
    # 1. 路径组装
    # -----------------------------------------------------
    base_config_path = PROJECT_ROOT / "configs" / "base_config.yaml"
    dataset_dir = PROJECT_ROOT / "data" / DATASET_NAME
    video_path = dataset_dir / "video.mp4"
    robot_config_path = dataset_dir / "config" / "align_pose.json"

    if not base_config_path.exists():
        logger.error(f"❌ Base config not found: {base_config_path}")
        return
    if not video_path.exists():
        logger.error(f"❌ Video file not found: {video_path}")
        return

    # -----------------------------------------------------
    # 2. 启动工作区
    # -----------------------------------------------------
    ws_mgr = WorkspaceManager(str(base_config_path))
    workspace = ws_mgr.resolve(mode="new")
    logger.info(f"🚀 [Step 1] Dataset: {DATASET_NAME}")
    logger.info(f"📂 Workspace Created: {workspace}")

    # -----------------------------------------------------
    # 3. 配置管理
    # -----------------------------------------------------
    cfg = OmegaConf.load(base_config_path)
    
    # 修正相对路径
    cfg.checkpoint = str(PROJECT_ROOT / cfg.get("checkpoint", "checkpoints/sam2_hiera_large.pt"))
    cfg.model_cfg = str(PROJECT_ROOT / cfg.get("model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml"))
    
    # 覆盖数据路径
    cfg.data.video_path = str(video_path)
    cfg.data.robot_config = str(robot_config_path)
    cfg.dataset_name = DATASET_NAME
    
    OmegaConf.save(cfg, workspace / "runtime_config.yaml")

    # -----------------------------------------------------
    # 4. 准备输出目录
    # -----------------------------------------------------
    mask_obj_dir = workspace / "masks_object"
    mask_robot_dir = workspace / "masks_robot"
    images_dir = workspace / "images"  # 新增：用于存储 RGB 帧给 Step 2 使用
    
    mask_obj_dir.mkdir(exist_ok=True)
    mask_robot_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    # -----------------------------------------------------
    # 5. 初始化 SAM 2 & 视频属性
    # -----------------------------------------------------
    sam2 = SAM2Wrapper(cfg)
    
    # 检测旋转
    detected_rotate = sam2._detect_video_rotation(video_path)
    rotate_code = cfg.pipeline.get("input_rotate_code", None)
    if rotate_code is None: rotate_code = detected_rotate
    if rotate_code is not None: logger.info(f"🔄 Rotation Applied: {rotate_code}")

    # 视频参数
    cap = cv2.VideoCapture(str(video_path))
    w_raw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_raw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 输出尺寸
    swap_dims = rotate_code in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]
    w_out, h_out = (h_raw, w_raw) if swap_dims else (w_raw, h_raw)
    if w_out % 2 != 0: w_out -= 1
    if h_out % 2 != 0: h_out -= 1

    vis_writer = FFmpegWriter(workspace / "segmentation_vis.mp4", w_out, h_out, fps)

    # -----------------------------------------------------
    # 6. 执行感知循环 (SAM 2 + 图像提取)
    # -----------------------------------------------------
    logger.info("🎨 Starting Segmentation & Frame Extraction...")
    generator = sam2.run_generator(str(video_path))
    
    pbar = tqdm(total=total_frames, unit="frame", desc="Processing")
    cap_read = cv2.VideoCapture(str(video_path))
    current_idx = 0

    # 收集每一帧（旋转后）用于 MASt3R
    # 注意：如果显存有限，这里可以只存路径，但 MASt3R Wrapper 需要 np.ndarray
    rgb_frames_buffer = [] 

    try:
        for result in generator:
            if result.get("status") == "cancelled":
                logger.warning("🚫 Segmentation Cancelled.")
                break
            
            frame_idx = result["frame_idx"]
            all_masks = result["masks"]

            # 同步读取原视频
            while current_idx <= frame_idx:
                ret, frame = cap_read.read()
                current_idx += 1
                if not ret: break
            if not ret: break

            # 1. 旋转与缩放 (统一标准)
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)
            if frame.shape[1] != w_out or frame.shape[0] != h_out:
                frame = cv2.resize(frame, (w_out, h_out))

            # 2. 保存 RGB 帧 (给 3DGS 使用)
            # MASt3R 和 3DGS 都需要这些处理过的图片
            frame_filename = f"{frame_idx:05d}.png"
            cv2.imwrite(str(images_dir / frame_filename), frame)
            
            # 3. 存入 Buffer (给 MASt3R 使用)
            # 为了节省内存，如果视频极长，可以考虑只存关键帧，
            # 但 MASt3R Wrapper 会自己处理间隔。
            # 这里存 BGR -> RGB (MASt3R 需要 RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frames_buffer.append(rgb_frame)

            # 4. 处理 Mask & 可视化
            vis = frame.copy()
            for obj_id, mask in all_masks.items():
                mask_uint8 = mask.astype(np.uint8) * 255
                if rotate_code is not None:
                    mask_uint8 = cv2.rotate(mask_uint8, rotate_code)
                if mask_uint8.shape[:2] != (h_out, w_out):
                    mask_uint8 = cv2.resize(mask_uint8, (w_out, h_out), interpolation=cv2.INTER_NEAREST)

                if obj_id == 1: # Object
                    cv2.imwrite(str(mask_obj_dir / frame_filename), mask_uint8)
                    vis[mask_uint8 > 127] = cv2.addWeighted(vis[mask_uint8 > 127], 0.5, np.array([0,0,255]), 0.5, 0)
                elif obj_id == 2: # Robot
                    cv2.imwrite(str(mask_robot_dir / frame_filename), mask_uint8)
                    vis[mask_uint8 > 127] = cv2.addWeighted(vis[mask_uint8 > 127], 0.5, np.array([255,0,0]), 0.5, 0)

            vis_writer.write(vis)
            pbar.update(1)

    except KeyboardInterrupt:
        logger.warning("Interrupted.")
    finally:
        pbar.close()
        vis_writer.release()
        cap_read.release()

    # -----------------------------------------------------
    # 7. 3D 姿态估计 (MASt3R)
    # -----------------------------------------------------
    logger.info("📷 Running MASt3R for 3D Pose Estimation...")
    
    mast3r = MASt3RWrapper(device="cuda")
    
    # 关键帧间隔：根据显存和视频长度调整，默认 15
    keyframe_interval = 15
    # 注意：我们已经手动旋转过图片了，这里传 None 防止重复旋转
    poses, cloud = mast3r.run(rgb_frames_buffer, keyframe_interval=keyframe_interval, rotate_code=None)
    
    # 保存结果
    if len(poses) > 0:
        # A. 生成 transforms.json (3DGS 格式)
        transforms = {
            "camera_model": "PINHOLE",
            "frames": []
        }
        
        # 估算内参 (如果 MASt3R 未返回，使用视场角估算)
        # 假设 HFOV ~ 60度 -> fl ~ w * 0.8
        fl_x = w_out * 1.0 
        fl_y = w_out * 1.0
        cx = w_out / 2.0
        cy = h_out / 2.0

        for i, pose in enumerate(poses):
            # 找到对应的真实帧 ID
            real_idx = i * keyframe_interval
            
            # Nerfstudio / 3DGS 通常期望 OpenGL 坐标系
            # MASt3R 输出通常是对齐的，这里直接保存，后续视情况在 Step 2 调整
            frame_entry = {
                "file_path": f"images/{real_idx:05d}.png",
                "transform_matrix": pose.tolist(),
                "w": w_out,
                "h": h_out,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "cx": cx,
                "cy": cy
            }
            transforms["frames"].append(frame_entry)

        with open(workspace / "transforms.json", "w") as f:
            json.dump(transforms, f, indent=4)
        logger.info(f"✅ Saved transforms.json with {len(poses)} frames.")

        # B. 保存稀疏点云 (加速 Step 2 初始化)
        if cloud is not None:
            # 需要转为 open3d 或 plyfile 保存
            # 这里简单复用 MASt3RWrapper 的 save_results 逻辑部分
            mast3r.save_results(workspace / "sparse_recon", poses, cloud)
            # 也可以手动保存一份根目录的 ply
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud)
            o3d.io.write_point_cloud(str(workspace / "sparse_pc.ply"), pcd)
            logger.info("✅ Saved sparse_pc.ply")

    else:
        logger.error("❌ MASt3R failed to reconstruct poses.")

    # -----------------------------------------------------
    # 8. 机器人配置复制
    # -----------------------------------------------------
    if robot_config_path.exists():
        with open(robot_config_path) as f:
            data = json.load(f)
        with open(workspace / "align_pose.json", 'w') as f:
            json.dump(data, f)
        logger.info("✅ Robot config copied.")

    logger.info(f"🎉 Step 1 Pipeline Finished. All assets in: {workspace}")

if __name__ == "__main__":
    run_step1()