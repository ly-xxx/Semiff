import sys
import cv2
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig, open_dict
from tqdm import tqdm
import os
import subprocess
import json

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from semiff.perception.sam2_wrapper import SAM2Wrapper
from semiff.core.logger import get_logger

logger = get_logger("vis_tool")

def get_video_rotation(video_path):
    """使用 ffprobe 检测视频的旋转元数据"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_streams', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0: return None, False

        data = json.loads(result.stdout)
        video_stream = next((s for s in data.get('streams', []) if s['codec_type'] == 'video'), None)
        if not video_stream: return None, False
            
        tags = video_stream.get('tags', {})
        rotate = int(tags.get('rotate', 0))
        
        if rotate != 0: logger.info(f"🕵️ Auto-detected Rotation Metadata: {rotate}°")

        if rotate == 90: return cv2.ROTATE_90_CLOCKWISE, True
        elif rotate == 180: return cv2.ROTATE_180, False
        elif rotate == 270: return cv2.ROTATE_90_COUNTERCLOCKWISE, True
        else: return None, False
    except Exception:
        return None, False

class FFmpegWriter:
    def __init__(self, filename, width, height, fps):
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
            _, stderr = self.process.communicate()
            logger.error(f"FFmpeg Error: {stderr.decode()}")
            raise

    def release(self):
        if self.process:
            self.process.stdin.close()
            self.process.wait()

@hydra.main(config_path="../src/semiff/config", config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    cv2.setNumThreads(0)
    os.environ["OMP_NUM_THREADS"] = "1"
    
    video_path = str(cfg.data.video_path)
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === 新增：掩码保存目录 ===
    mask_obj_dir = output_dir / "masks_object"
    mask_robot_dir = output_dir / "masks_robot"
    mask_obj_dir.mkdir(exist_ok=True)
    mask_robot_dir.mkdir(exist_ok=True)
    logger.info(f"📂 Storage initialized:\n  - {mask_obj_dir}\n  - {mask_robot_dir}")

    # 1. 旋转检测
    rotate_code, swap_dims = get_video_rotation(video_path)

    cap = cv2.VideoCapture(video_path)
    w_raw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_raw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 2. 初始化
    if rotate_code is not None:
        with open_dict(cfg):
            if 'pipeline' not in cfg: cfg.pipeline = {}
            cfg.pipeline.input_rotate_code = int(rotate_code)

    sam2 = SAM2Wrapper(cfg)
    
    # 3. 输出尺寸
    if swap_dims:
        w_out, h_out = h_raw, w_raw
    else:
        w_out, h_out = w_raw, h_raw
    
    if w_out % 2 != 0: w_out -= 1
    if h_out % 2 != 0: h_out -= 1
    
    out_path = output_dir / "final_vis_ffmpeg.mp4"
    writer = FFmpegWriter(out_path, w_out, h_out, fps)

    generator = sam2.run_generator(video_path)
    
    logger.info("🚀 Processing started...")
    pbar = tqdm(total=total_frames, unit="frame", desc="Generating")
    
    try:
        current_idx = 0
        cap_read = cv2.VideoCapture(video_path)
        
        for result in generator:
            if result.get("status") == "cancelled":
                logger.warning("❌ Canceled.")
                break
                
            frame_idx = result["frame_idx"]
            all_masks = result["masks"] # Dict {1: mask, 2: mask}

            # 同步读取
            while current_idx <= frame_idx:
                ret, frame = cap_read.read()
                current_idx += 1
                if not ret: break
            if not ret: break

            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)

            # 准备可视化画布
            vis = frame.copy()
            
            # === 处理每个 ID ===
            # ID 1: Object -> Red
            # ID 2: Robot -> Blue
            
            for obj_id, mask in all_masks.items():
                # Mask 尺寸对齐
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                
                # --- A. 保存 Mask (PNG) ---
                binary_mask = (mask.astype(np.uint8) * 255)
                filename = f"{frame_idx:05d}.png"
                
                if obj_id == 1:
                    cv2.imwrite(str(mask_obj_dir / filename), binary_mask)
                    # 可视化: Red
                    color_mask = np.zeros_like(frame)
                    color_mask[:, :, 2] = 255 
                    vis[mask] = cv2.addWeighted(frame[mask], 0.6, color_mask[mask], 0.4, 0)
                    
                elif obj_id == 2:
                    cv2.imwrite(str(mask_robot_dir / filename), binary_mask)
                    # 可视化: Blue
                    color_mask = np.zeros_like(frame)
                    color_mask[:, :, 0] = 255
                    vis[mask] = cv2.addWeighted(frame[mask], 0.6, color_mask[mask], 0.4, 0)

            # 写入视频
            if vis.shape[1] != w_out or vis.shape[0] != h_out:
                vis = cv2.resize(vis, (w_out, h_out))

            writer.write(vis)
            pbar.update(1)

    except KeyboardInterrupt:
        logger.warning("Interrupted.")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pbar.close()
        writer.release()
        cap_read.release()
        logger.info(f"✅ Video saved: {out_path}")
        logger.info(f"✅ Masks saved to: {output_dir}")

if __name__ == "__main__":
    main()