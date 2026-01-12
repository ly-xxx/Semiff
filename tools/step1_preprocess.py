import argparse
import json
import numpy as np
import logging
import cv2
import subprocess
import sys
import pandas as pd
import warnings
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
from semiff.core.workspace import WorkspaceManager

try:
    from semiff.solvers.sam2_wrapper import SAM2Wrapper
except ImportError:
    SAM2Wrapper = None
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
    OmegaConf.save(cfg, runtime_cfg_path)

    ENABLE_SAM2 = cfg.pipeline.get("steps", {}).get("step1", {}).get("enable_sam2", True)
    ENABLE_MAST3R = cfg.pipeline.get("steps", {}).get("step1", {}).get("enable_mast3r", True)

    mask_obj_dir = workspace / "masks_object"
    mask_robot_dir = workspace / "masks_robot"
    images_dir = workspace / "images"
    for d in [mask_obj_dir, mask_robot_dir, images_dir]: d.mkdir(exist_ok=True)

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

    rgb_frames_buffer = []
    masks_buffer = [] # 存储对应帧的 Robot Mask

    # === Phase A: SAM 2 ===
    if ENABLE_SAM2:
        logger.info("🎨 [SAM2] Starting Segmentation...")
        eff_rotate_code = rotate_code if need_manual_rotate else None
        
        with open_dict(cfg):
            if 'pipeline' not in cfg: cfg.pipeline = {}
            cfg.pipeline.input_rotate_code = int(eff_rotate_code) if eff_rotate_code is not None else None

        sam2 = SAM2Wrapper(cfg)

        if sam2.predictor:
            # 这里 vis_writer 相关的代码我略微精简，重点在 mask 提取
            generator = sam2.run_generator(str(video_path), output_dir=workspace)
            cap_read = cv2.VideoCapture(str(video_path))
            current_idx = 0

            try:
                total_frames = int(cap_read.get(cv2.CAP_PROP_FRAME_COUNT))
                pbar = tqdm(total=total_frames, desc="Processing")

                for result in generator:
                    if result.get("status") == "cancelled": break
                    frame_idx = result["frame_idx"]
                    all_masks = result["masks"] # Dict {obj_id: mask}

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

                    # 🔥 Mask 处理逻辑：只提取 Robot (ID=2)，或者合并所有前景
                    # 假设 ID 2 是机械臂
                    robot_mask = np.zeros((h_out, w_out), dtype=np.uint8)
                    
                    if 2 in all_masks:
                        m = (all_masks[2] * 255).astype(np.uint8)
                        # Resize 保护
                        if m.shape[:2] != (h_out, w_out):
                            m = cv2.resize(m, (w_out, h_out), interpolation=cv2.INTER_NEAREST)
                        robot_mask = m
                        # 保存一份到磁盘
                        cv2.imwrite(str(mask_robot_dir / f"{frame_idx:05d}.png"), robot_mask)
                    
                    masks_buffer.append(robot_mask) # 存入 Buffer
                    pbar.update(1)
                pbar.close()
            finally:
                cap_read.release()
    else:
        # Manual Mode
        # ... (简略，同前) ...
        pass

    # === Phase B: MASt3R ===
    if ENABLE_MAST3R and MASt3RWrapper is not None and len(rgb_frames_buffer) > 0:
        logger.info(f"🧠 [MASt3R] Reconstruction with {len(rgb_frames_buffer)} frames...")
        mast3r = MASt3RWrapper(device="cuda")
        debug_dir = workspace / "debug_mast3r"
        debug_dir.mkdir(exist_ok=True)

        # 运行 MASt3R，传入 masks_buffer 用于标记
        # keyframe_interval 设为 2，尽可能多地喂数据，由 wrapper 内部控制 120 帧上限
        poses, cloud, intrinsics = mast3r.run(
            frames=rgb_frames_buffer,
            masks=masks_buffer, 
            keyframe_interval=2, 
            debug_dir=debug_dir
        )

        np.save(workspace / "camera_poses.npy", poses)
        np.save(workspace / "sparse_cloud.npy", cloud) # 注意现在 cloud 是 Nx7
        np.save(workspace / "intrinsics.npy", intrinsics)
        
        # 🆕 保存带 Label 的 PLY
        if cloud.shape[0] > 0:
            ply_path = workspace / "sparse_cloud.ply"
            
            # cloud: [X, Y, Z, R, G, B, Label]
            xyz = cloud[:, :3]
            rgb = cloud[:, 3:6].astype(np.uint8)
            lbl = cloud[:, 6].astype(np.uint8)

            # 自定义 Header，增加 'label' 属性
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
                "property uchar label\n"  # 🔥 新增属性
                "end_header\n"
            )

            with open(ply_path, "w") as f:
                f.write(header)
                # 逐行写入
                for p, c, l in zip(xyz, rgb, lbl):
                    f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])} {int(l)}\n")

            logger.info(f"✅ Saved LABELED point cloud to {ply_path}")
            logger.info("ℹ️  Use 'label' scalar field in CloudCompare/MeshLab to isolate robot.")
    else:
        if len(rgb_frames_buffer) == 0: logger.error("❌ No frames loaded!")

if __name__ == "__main__":
    run_step1()