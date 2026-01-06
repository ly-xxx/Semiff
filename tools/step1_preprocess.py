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

# ==================== 1. 路径配置 ====================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
from semiff.core.workspace import WorkspaceManager

try:
    from semiff.perception.sam2_wrapper import SAM2Wrapper
except ImportError:
    SAM2Wrapper = None
try:
    from semiff.perception.mast3r_wrapper import MASt3RWrapper
except ImportError:
    MASt3RWrapper = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Step1")

# ==================== 辅助函数 ====================
def get_video_rotation(video_path):
    """
    获取视频的旋转 Metadata。
    注意：这只是 Metadata，OpenCV 读取时可能会自动应用这个旋转，也可能不会。
    """
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0: return None, False
        data = json.loads(result.stdout)
        video_stream = next((s for s in data.get('streams', []) if s['codec_type'] == 'video'), None)
        if not video_stream: return None, False
        rotate = int(video_stream.get('tags', {}).get('rotate', 0))

        if rotate != 0:
            logger.info(f"🕵️ Metadata Rotation Tag: {rotate}°")

        if rotate == 90: return cv2.ROTATE_90_CLOCKWISE, True
        elif rotate == 180: return cv2.ROTATE_180, False
        elif rotate == 270: return cv2.ROTATE_90_COUNTERCLOCKWISE, True
        elif rotate == -90: return cv2.ROTATE_90_COUNTERCLOCKWISE, True
        else: return None, False
    except: return None, False

class FFmpegWriter:
    def __init__(self, filename, width, height, fps):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        if width % 2 != 0: width -= 1
        if height % 2 != 0: height -= 1
        self.cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', f'{fps}',
            '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'fast', '-crf', '23', str(filename)
        ]
        self.process = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def write(self, frame):
        try:
            h, w = frame.shape[:2]
            target_str = self.cmd[7]
            target_w, target_h = int(target_str.split('x')[0]), int(target_str.split('x')[1])
            if w != target_w or h != target_h: frame = cv2.resize(frame, (target_w, target_h))
            self.process.stdin.write(frame.tobytes())
        except: pass

    def release(self):
        if self.process:
            self.process.stdin.close()
            self.process.wait()

# ==================== 主流程 ====================
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

    # 1. 预读视频信息
    rotate_code, is_vertical_meta = get_video_rotation(video_path)

    # 2. 读取第一帧来决定是否需要手动旋转
    # (这是针对你的环境做的 Robust Check)
    temp_cap = cv2.VideoCapture(str(video_path))
    ret, temp_frame = temp_cap.read()
    temp_cap.release()
    if not ret: raise RuntimeError(f"Cannot read video: {video_path}")

    h_raw, w_raw = temp_frame.shape[:2]

    # === 智能旋转决策 ===
    need_manual_rotate = False

    # 如果 Metadata 说它是竖屏 (90/270度)，但读出来的宽 > 高 (横屏)
    # 说明 OpenCV 没有自动旋转，我们需要手动转
    if is_vertical_meta and w_raw > h_raw:
        need_manual_rotate = True
        logger.info(f"🔄 Manual Rotation Required: Metadata says Vertical, but Raw is {w_raw}x{h_raw}")
    else:
        # 否则 (Metadata没说要转，或者 Metadata说了要转且OpenCV已经转成了竖屏)
        need_manual_rotate = False
        logger.info(f"✅ No Manual Rotation Needed. Raw frame is naturally {w_raw}x{h_raw}")
        # 如果已经自动转正了，清除 rotate_code 防止后续逻辑误判
        if w_raw < h_raw:
            rotate_code = None

    # 最终输出尺寸
    if need_manual_rotate:
        # 交换宽高
        w_out, h_out = h_raw, w_raw
    else:
        # 保持原样
        w_out, h_out = w_raw, h_raw

    logger.info(f"📐 Target Dims: {w_out}x{h_out}")
    rgb_frames_buffer = []

    # === Phase A: SAM 2 ===
    if ENABLE_SAM2:
        logger.info("🎨 [SAM2] Starting Segmentation...")
        # 注意：我们将 'rotate_code' 传给 SAM2 仅用于它内部逻辑
        # 但既然我们发现环境会自动旋转，这里传 None 也是安全的，
        # 或者只在 need_manual_rotate 为 True 时传。

        eff_rotate_code = rotate_code if need_manual_rotate else None

        with open_dict(cfg):
            if 'pipeline' not in cfg: cfg.pipeline = {}
            # 只有需要手动转的时候，才告诉 SAM2 去处理旋转
            if eff_rotate_code is not None:
                cfg.pipeline.input_rotate_code = int(eff_rotate_code)
            else:
                cfg.pipeline.input_rotate_code = None

        sam2 = SAM2Wrapper(cfg)

        if sam2.predictor:
            vis_writer = FFmpegWriter(workspace / "segmentation_vis.mp4", w_out, h_out, 30)
            generator = sam2.run_generator(str(video_path), output_dir=workspace)
            cap_read = cv2.VideoCapture(str(video_path))
            current_idx = 0

            try:
                # 获取总帧数用于进度条
                total_frames = int(cap_read.get(cv2.CAP_PROP_FRAME_COUNT))
                pbar = tqdm(total=total_frames, desc="Processing")

                for result in generator:
                    if result.get("status") == "cancelled": break

                    frame_idx = result["frame_idx"]
                    all_masks = result["masks"]

                    # 同步读取视频帧
                    while current_idx <= frame_idx:
                        ret, raw_frame = cap_read.read()
                        current_idx += 1
                    if not ret: break

                    # === 这里的 frame_upright 是给所有下游任务 (MASt3R, Vis) 用的标准帧 ===
                    if need_manual_rotate and rotate_code is not None:
                        frame_upright = cv2.rotate(raw_frame, rotate_code)
                    else:
                        frame_upright = raw_frame

                    # 保存 RGB 帧
                    cv2.imwrite(str(images_dir / f"{frame_idx:05d}.png"), frame_upright)
                    rgb_frames_buffer.append(cv2.cvtColor(frame_upright, cv2.COLOR_BGR2RGB))

                    # 处理 Mask 和可视化
                    vis_frame = frame_upright.copy()

                    for obj_id, mask_in in all_masks.items():
                        mask_u8 = (mask_in * 255).astype(np.uint8)

                        # 检查 Mask 尺寸是否需要调整 (通常 SAM2 内部如果不转，这里 mask 也是 raw 尺寸)
                        mh, mw = mask_u8.shape[:2]
                        th, tw = frame_upright.shape[:2]

                        if mh != th or mw != tw:
                            # 只有在尺寸不匹配时才 Resize/Rotate
                            mask_upright = cv2.resize(mask_u8, (tw, th), interpolation=cv2.INTER_NEAREST)
                        else:
                            mask_upright = mask_u8

                        filename = f"{frame_idx:05d}.png"
                        target_dir = mask_obj_dir if obj_id == 1 else mask_robot_dir
                        cv2.imwrite(str(target_dir / filename), mask_upright)

                        # 可视化叠加
                        color_vis = (0, 0, 255) if obj_id == 1 else (255, 0, 0)
                        mask_bool = mask_upright > 128
                        if mask_bool.any():
                            # 简易半透明叠加
                            overlay = vis_frame.copy()
                            overlay[mask_bool] = color_vis
                            cv2.addWeighted(overlay, 0.4, vis_frame, 0.6, 0, vis_frame)

                    vis_writer.write(vis_frame)
                    pbar.update(1)
                pbar.close()
            finally:
                vis_writer.release()
                cap_read.release()

    else:
        # =========================================================
        # 🆕 新增逻辑：如果 SAM2 被禁用，我们需要手动读取视频！
        # =========================================================
        logger.info("🎥 SAM2 disabled. Reading video frames manually for MASt3R...")

        cap_read = cv2.VideoCapture(str(video_path))
        total_frames = int(cap_read.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Reading Frames")
        frame_idx = 0

        while True:
            ret, raw_frame = cap_read.read()
            if not ret: break

            # === 核心：应用之前检测到的旋转逻辑 ===
            if need_manual_rotate and rotate_code is not None:
                frame_upright = cv2.rotate(raw_frame, rotate_code)
            else:
                frame_upright = raw_frame

            # 1. 保存到 Buffer 给 MASt3R
            rgb_frames_buffer.append(cv2.cvtColor(frame_upright, cv2.COLOR_BGR2RGB))

            # 2. (可选) 保存图片到磁盘，方便你检查旋转是否正确
            # 既然是 Debug 阶段，强烈建议保存一下，确认图片是不是正的
            cv2.imwrite(str(images_dir / f"{frame_idx:05d}.png"), frame_upright)

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap_read.release()
        logger.info(f"✅ Loaded {len(rgb_frames_buffer)} frames.")

    # === Phase B: MASt3R ===
    # (确保这里判断了 buffer 不为空)
    if ENABLE_MAST3R and MASt3RWrapper is not None and len(rgb_frames_buffer) > 0:
        logger.info(f"🧠 [MASt3R] Starting Geometry Reconstruction with {len(rgb_frames_buffer)} frames...")
        mast3r = MASt3RWrapper(device="cuda")

        # 这里的 rgb_frames_buffer 已经是 正确朝向 (Upright) 的了
        # 不需要再传 rotate_code，直接喂进去
        # 创建调试目录
        debug_dir = workspace / "debug_mast3r"
        debug_dir.mkdir(exist_ok=True)

        poses, cloud = mast3r.run(
            frames=rgb_frames_buffer,
            keyframe_interval=3,   # 降低间隔保证高重叠率
            debug_dir=debug_dir    # 启用调试可视化
        )

        # 保存结果
        np.save(workspace / "camera_poses.npy", poses)
        np.save(workspace / "sparse_cloud.npy", cloud)

        # 🆕 保存彩色 PLY
        if cloud.shape[0] > 0:
            ply_path = workspace / "sparse_cloud.ply"

            # 分离坐标和颜色
            xyz = cloud[:, :3]
            rgb = cloud[:, 3:].astype(np.uint8) # 转回整数以便写入

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
                "end_header\n"
            )

            with open(ply_path, "w") as f:
                f.write(header)
                for p, c in zip(xyz, rgb):
                    # 写入: X Y Z R G B
                    f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

            logger.info(f"✅ Saved COLOR point cloud to {ply_path}")
            logger.info(f"🖼️ Check debug images in {debug_dir}")
    else:
        if len(rgb_frames_buffer) == 0:
            logger.error("❌ No frames loaded! Cannot run MASt3R.")

if __name__ == "__main__":
    run_step1()