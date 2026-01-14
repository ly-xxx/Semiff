import argparse
import json
import numpy as np
import logging
import cv2
import subprocess
import sys
import pandas as pd
import warnings
import shutil
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict

warnings.filterwarnings("ignore")

# 导入统一路径管理工具
_current_file = Path(__file__).resolve()
_src_dir = _current_file.parents[1] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from semiff.core.workspace import WorkspaceManager

# 🔧 使用统一方法获取项目根目录
PROJECT_ROOT = WorkspaceManager.find_project_root(start_path=_current_file.parent)

try:
    from semiff.solvers.sam2_wrapper import SAM2Wrapper
except ImportError:
    SAM2Wrapper = None

# 🆕 尝试导入 SAM 3 Wrapper
try:
    from semiff.solvers.sam3_wrapper import SAM3Wrapper
except ImportError:
    SAM3Wrapper = None

try:
    from semiff.solvers.mast3r_wrapper import MASt3RWrapper
except ImportError:
    MASt3RWrapper = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Step1")

# ... (辅助函数 get_video_rotation 和 FFmpegWriter 保持不变，此处省略以节省篇幅) ...

def get_video_rotation(video_path):
    # (保持原样)
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0: return None, False
        data = json.loads(result.stdout)
        video_stream = next((s for s in data.get('streams', []) if s['codec_type'] == 'video'), None)
        if not video_stream: return None, False
        rotate = int(video_stream.get('tags', {}).get('rotate', 0))
        if rotate != 0: logger.info(f"🕵️ Metadata Rotation Tag: {rotate}°")
        if rotate == 90: return cv2.ROTATE_90_CLOCKWISE, True
        elif rotate == 180: return cv2.ROTATE_180, False
        elif rotate == 270: return cv2.ROTATE_90_COUNTERCLOCKWISE, True
        elif rotate == -90: return cv2.ROTATE_90_COUNTERCLOCKWISE, True
        else: return None, False
    except: return None, False

def run_step1():
    base_config_path = PROJECT_ROOT / "configs" / "base_config.yaml"
    base_cfg = OmegaConf.load(base_config_path)

    # 1. 基础参数读取 (✅ 检查通过)
    workspace_mode = base_cfg.pipeline.get("mode", "auto")
    root_dir = base_cfg.data.get("root_dir", "data/example_01")
    video_path_rel = base_cfg.data.get("video_path", "video.mp4")

    dataset_dir = Path(root_dir) if Path(root_dir).is_absolute() else PROJECT_ROOT / root_dir
    video_path = dataset_dir / video_path_rel

    ws_mgr = WorkspaceManager(str(base_config_path))
    workspace = ws_mgr.resolve(mode=workspace_mode)
    logger.info(f"📂 Workspace: {workspace}")

    runtime_cfg_path = workspace / "runtime_config.yaml"
    cfg = OmegaConf.merge(OmegaConf.load(runtime_cfg_path), base_cfg) if runtime_cfg_path.exists() else base_cfg

    # 2. 步骤开关读取 (✅ 检查通过)
    ENABLE_SAM2 = cfg.pipeline.get("steps", {}).get("step1", {}).get("enable_sam2", False)
    ENABLE_SAM3 = cfg.pipeline.get("steps", {}).get("step1", {}).get("enable_sam3", True)
    ENABLE_MAST3R = cfg.pipeline.get("steps", {}).get("step1", {}).get("enable_mast3r", True)

    mask_obj_dir = workspace / "masks_object"
    mask_robot_dir = workspace / "masks_robot"
    images_dir = workspace / "images"
    for d in [mask_obj_dir, mask_robot_dir, images_dir]: d.mkdir(exist_ok=True)

    # 3. 旋转检测逻辑 (✅ 检查通过)
    rotate_code, is_vertical_meta = get_video_rotation(video_path)
    temp_cap = cv2.VideoCapture(str(video_path))
    ret, temp_frame = temp_cap.read()
    temp_cap.release()
    if not ret: raise RuntimeError(f"Cannot read video: {video_path}")
    h_raw, w_raw = temp_frame.shape[:2]

    need_manual_rotate = False
    if is_vertical_meta and w_raw > h_raw:
        need_manual_rotate = True
        logger.info(f"🔄 Manual Rotation Required: Metadata says Vertical, but Raw is {w_raw}x{h_raw}")
    else:
        logger.info(f"✅ No Manual Rotation Needed. Raw frame is naturally {w_raw}x{h_raw}")
        if w_raw < h_raw: rotate_code = None

    if need_manual_rotate: w_out, h_out = h_raw, w_raw
    else: w_out, h_out = w_raw, h_raw
    logger.info(f"📐 Target Dims: {w_out}x{h_out}")

    # 🔥🔥【修复 1】: 将计算出的 rotate_code 注入到 cfg 中 🔥🔥
    # 必须使用 open_dict 上下文才能修改 OmegaConf 对象
    effective_rotate_code = int(rotate_code) if (need_manual_rotate and rotate_code is not None) else None

    with open_dict(cfg):
        if 'pipeline' not in cfg: cfg.pipeline = {}
        # 显式写入，确保 Wrapper 能读到 'input_rotate_code'
        cfg.pipeline.input_rotate_code = effective_rotate_code
        logger.info(f"🔧 Config Injection: pipeline.input_rotate_code = {effective_rotate_code}")

    # 保存最终使用的配置（包含注入的旋转参数）
    OmegaConf.save(cfg, runtime_cfg_path)

    rgb_frames_buffer = []
    masks_buffer = []

    # === Phase A: Segmentation ===
    active_segmenter = None
    if ENABLE_SAM3 and SAM3Wrapper:
        logger.info("🚀 [SAM3] Initializing Text-Driven Segmentation...")
        # SAM3Wrapper 内部通常会读取 cfg.sam3 和 cfg.pipeline，这里传入完整 cfg 是对的
        active_segmenter = SAM3Wrapper(cfg)

    elif ENABLE_SAM2 and SAM2Wrapper:
        logger.info("🎨 [SAM2] Initializing Segmentation...")
        # SAM2Wrapper 现在可以直接从 cfg.sam2 读取配置
        active_segmenter = SAM2Wrapper(cfg)

    if active_segmenter:
        generator = active_segmenter.run_generator(str(video_path), output_dir=workspace)
        cap_read = cv2.VideoCapture(str(video_path))
        
        # 🎬 创建可视化帧临时目录
        vis_frames_dir = workspace / "vis_frames_temp"
        vis_frames_dir.mkdir(exist_ok=True)
        
        # 获取原视频帧率
        fps = cap_read.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps > 120:  # 防止异常值
            fps = 30.0
        logger.info(f"📹 Video FPS: {fps}")

        current_idx = 0
        try:
            total_frames = int(cap_read.get(cv2.CAP_PROP_FRAME_COUNT))
            pbar = tqdm(total=total_frames, desc="Segmenting")

            for result in generator:
                if result.get("status") == "cancelled": break
                frame_idx = result["frame_idx"]
                all_masks = result["masks"]

                while current_idx <= frame_idx:
                    ret, raw_frame = cap_read.read()
                    current_idx += 1
                if not ret: break

                if need_manual_rotate and rotate_code is not None:
                    frame_upright = cv2.rotate(raw_frame, rotate_code)
                else:
                    frame_upright = raw_frame

                cv2.imwrite(str(images_dir / f"{frame_idx:05d}.png"), frame_upright)
                rgb_frames_buffer.append(cv2.cvtColor(frame_upright, cv2.COLOR_BGR2RGB))

                vis_frame = frame_upright.copy()

                # Robot Mask (ID 2)
                robot_mask = np.zeros((h_out, w_out), dtype=np.uint8)
                if 2 in all_masks:
                    m = (all_masks[2] * 255).astype(np.uint8)
                    if m.shape[:2] != (h_out, w_out):
                        m = cv2.resize(m, (w_out, h_out), interpolation=cv2.INTER_NEAREST)
                    robot_mask = m
                    bool_mask = robot_mask > 0

                    if bool_mask.any():
                        color = np.array([255, 0, 0], dtype=np.uint8)  # 蓝色
                        roi = vis_frame[bool_mask]
                        blended = (roi * 0.5 + color * 0.5).astype(np.uint8)
                        vis_frame[bool_mask] = blended

                # Object Mask (ID 1)
                object_mask = np.zeros((h_out, w_out), dtype=np.uint8)
                if 1 in all_masks:
                    m = (all_masks[1] * 255).astype(np.uint8)
                    if m.shape[:2] != (h_out, w_out):
                        m = cv2.resize(m, (w_out, h_out), interpolation=cv2.INTER_NEAREST)
                    object_mask = m
                    bool_mask = object_mask > 0

                    if bool_mask.any():
                        color = np.array([0, 255, 255], dtype=np.uint8)  # 青色
                        roi = vis_frame[bool_mask]
                        blended = (roi * 0.5 + color * 0.5).astype(np.uint8)
                        vis_frame[bool_mask] = blended

                cv2.imwrite(str(mask_robot_dir / f"{frame_idx:05d}.png"), robot_mask)
                cv2.imwrite(str(mask_obj_dir / f"{frame_idx:05d}.png"), object_mask)
                
                # 🆕 保存可视化帧到临时目录
                cv2.imwrite(str(vis_frames_dir / f"{frame_idx:05d}.png"), vis_frame)

                masks_buffer.append(robot_mask)
                pbar.update(1)
            pbar.close()
        finally:
            cap_read.release()
            logger.info("✅ Segmentation Done.")
        
        # 🎬 使用 ffmpeg 合成视频
        video_save_path = workspace / "vis_segmentation.mp4"
        logger.info(f"🎥 Encoding video with ffmpeg: {video_save_path}")
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # 覆盖已存在的文件
            '-framerate', str(fps),
            '-i', str(vis_frames_dir / '%05d.png'),
            '-c:v', 'libx264',  # H.264 编码
            '-preset', 'medium',  # 编码速度 (faster/medium/slow)
            '-crf', '23',  # 质量 (18-28, 越小质量越好)
            '-pix_fmt', 'yuv420p',  # 兼容性最好
            str(video_save_path)
        ]
        
        try:
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            logger.info(f"✅ Video saved: {video_save_path}")
            
            # 清理临时帧
            shutil.rmtree(vis_frames_dir)
            logger.info("🧹 Cleaned up temporary frames")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ ffmpeg failed: {e.stderr}")
            logger.warning(f"⚠️  Temporary frames kept at: {vis_frames_dir}")

    # ... (Phase B: MASt3R 逻辑保持不变) ...
    # 为了完整性，请保留原有的 MASt3R 代码块
    if ENABLE_MAST3R and MASt3RWrapper is not None and len(rgb_frames_buffer) > 0:
        # (原样保留 Phase B 代码)
        logger.info(f"🧠 [MASt3R] Reconstruction with {len(rgb_frames_buffer)} frames...")
        mast3r = MASt3RWrapper(device="cuda")
        debug_dir = workspace / "debug_mast3r"
        debug_dir.mkdir(exist_ok=True)

        poses, cloud, intrinsics = mast3r.run(
            frames=rgb_frames_buffer,
            masks=masks_buffer,
            keyframe_interval=2,
            debug_dir=debug_dir
        )

        np.save(workspace / "camera_poses.npy", poses)
        np.save(workspace / "sparse_cloud.npy", cloud)
        np.save(workspace / "intrinsics.npy", intrinsics)

        # PLY 保存逻辑
        if cloud.shape[0] > 0:
            ply_path = workspace / "sparse_cloud.ply"

            xyz = cloud[:, :3]
            rgb = cloud[:, 3:6].astype(np.uint8)
            lbl = cloud[:, 6].astype(np.uint8)

            header = (
                "ply\n"
                "format ascii 1.0\n"
                f"element vertex {len(cloud)}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "property uchar label\n"
                "end_header\n"
            )

            with open(ply_path, "w") as f:
                f.write(header)
                for p, c, l in zip(xyz, rgb, lbl):
                    f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])} {int(l)}\n")

            logger.info(f"✅ Saved LABELED point cloud to {ply_path}")
            logger.info("ℹ️  Use 'label' scalar field in CloudCompare/MeshLab to isolate robot.")
    else:
        if len(rgb_frames_buffer) == 0: logger.error("❌ No frames loaded!")

if __name__ == "__main__":
    run_step1()