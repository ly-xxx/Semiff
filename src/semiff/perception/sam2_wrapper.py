"""
SAM 2 Wrapper: Video Segmentation with Auto-Prompting (Multi-Object Support)
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Generator, Tuple
from pathlib import Path
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
import subprocess
import json
import logging

# 使用项目统一 Logger
logger = logging.getLogger("SAM2Wrapper")

class SAM2Wrapper:
    def __init__(self, config: Dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 兼容 OmegaConf 和 dict 的取值方式
        get_cfg = lambda k, d: config.get(k, d) if hasattr(config, "get") else getattr(config, k, d)
        
        # 获取配置
        self.checkpoint = get_cfg("checkpoint", "checkpoints/sam2_hiera_large.pt")
        self.model_cfg = get_cfg("model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml")
        
        # 尝试解析相对路径到绝对路径
        if not os.path.exists(self.model_cfg):
            potential_root = Path(os.getcwd()).resolve()
            # 尝试多种路径组合
            candidates = [
                potential_root / self.model_cfg,
                potential_root.parent / self.model_cfg,
                Path(__file__).parents[3] / self.model_cfg # 尝试相对于 semiff 根目录
            ]
            for c in candidates:
                if c.exists():
                    self.model_cfg = str(c)
                    break

        # 解析 Pipeline 配置
        pipeline_cfg = get_cfg("pipeline", {})
        get_p_cfg = lambda k, d: pipeline_cfg.get(k, d) if hasattr(pipeline_cfg, "get") else getattr(pipeline_cfg, k, d)
        
        self.interactive_mode = get_p_cfg("interactive_mode", False)
        self.input_rotate_code = get_p_cfg("input_rotate_code", None)

        self.predictor = self._init_model()
        self.detected_rotate_code = None 

    def _init_model(self):
        try:
            import sam2
            from hydra.core.global_hydra import GlobalHydra
            from hydra import initialize_config_module, initialize_config_dir

            # 1. 清理 Hydra 状态
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()

            from sam2.build_sam import build_sam2_video_predictor
            
            # 2. 检查 Checkpoint
            if not os.path.exists(self.checkpoint):
                logger.error(f"❌ Checkpoint not found: {self.checkpoint}")
                return None

            logger.info(f"Loading SAM 2 model...")
            logger.info(f"  - Checkpoint: {self.checkpoint}")

            # 3. 智能配置加载 (Smart Config Loading)
            if os.path.exists(self.model_cfg):
                # === Case A: 本地文件存在 ===
                logger.info(f"  - Config (Local): {self.model_cfg}")
                abs_config_path = os.path.abspath(self.model_cfg)
                config_dir = os.path.dirname(abs_config_path)
                config_name = os.path.basename(abs_config_path)
                
                # 强制 Hydra 搜索该目录
                initialize_config_dir(config_dir=config_dir, version_base="1.2")
                return build_sam2_video_predictor(config_name, self.checkpoint, device=self.device)
            
            else:
                # === Case B: 本地文件不存在，尝试使用内置配置 ===
                # 提取文件名，例如 "sam2.1_hiera_l.yaml" -> "sam2_hiera_l.yaml"
                # 注意：SAM2 官方配置名通常不带 "2.1" 前缀，如果你的配置名包含它，可能需要修正
                fallback_name = os.path.basename(self.model_cfg)
                
                # 简单的修正逻辑：如果用户写了 sam2.1_hiera_l.yaml 但官方包里是 sam2_hiera_l.yaml
                if "sam2.1" in fallback_name and "hiera" in fallback_name:
                    fallback_name = fallback_name.replace("sam2.1", "sam2")
                
                logger.warning(f"⚠️ Local config not found: {self.model_cfg}")
                logger.warning(f"🔄 Falling back to package config: {fallback_name}")
                
                initialize_config_module("sam2", version_base="1.2")
                try:
                    return build_sam2_video_predictor(fallback_name, self.checkpoint, device=self.device)
                except Exception as e:
                    logger.error(f"❌ Fallback failed. Hydra could not find '{fallback_name}' in package 'sam2'.")
                    raise e

        except ImportError:
            logger.error("❌ SAM 2 library not installed. Please install it first.")
            return None
        except Exception as e:
            logger.error(f"❌ SAM 2 init failed: {e}")
            return None

    def _detect_video_rotation(self, video_path: str) -> Optional[int]:
        """使用 ffprobe 检测视频旋转元数据"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0: return None

            data = json.loads(result.stdout)
            video_stream = next((s for s in data.get('streams', []) if s['codec_type'] == 'video'), None)
            if not video_stream: return None

            tags = video_stream.get('tags', {})
            rotate = int(tags.get('rotate', 0))

            if rotate == 90: return cv2.ROTATE_90_CLOCKWISE
            elif rotate == 180: return cv2.ROTATE_180
            elif rotate == 270: return cv2.ROTATE_90_COUNTERCLOCKWISE
            return None
        except Exception:
            return None

    def _get_interactive_prompt(self, frame: np.ndarray, output_dir: Path, rotate_code: Optional[int] = None) -> Dict[int, np.ndarray]:
        """交互式点击获取 Prompt"""
        collected_points = {1: [], 2: []} # 1: Object, 2: Robot
        
        if os.environ.get('DISPLAY', '') == '':
            logger.warning("⚠️ No DISPLAY detected. Using center point default.")
            return {1: np.array([[frame.shape[1] // 2, frame.shape[0] // 2]], dtype=np.float32)}

        try:
            matplotlib.use('TkAgg')
        except:
            pass
            
        logger.info("\n>>> INTERACTIVE MODE <<<\n[Left Click]: Object (Red)\n[Right Click]: Robot (Blue)\n[Close Window]: Start Segmentation\n")
        
        display_frame = frame.copy()
        if rotate_code is not None:
            display_frame = cv2.rotate(display_frame, rotate_code)
            
        rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h_disp, w_disp = display_frame.shape[:2]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(rgb_display)
        ax.set_title("L-Click: Object | R-Click: Robot | Close to Finish")
        ax.axis('off')

        def on_click(event):
            if event.xdata is None or event.ydata is None: return
            if event.inaxes != ax: return

            click_x, click_y = event.xdata, event.ydata
            final_x = np.clip(click_x, 0, w_disp - 1)
            final_y = np.clip(click_y, 0, h_disp - 1)
            
            if event.button == 1: # Left
                obj_id = 1
                color = 'r*'
                logger.info(f"📍 Object (ID 1) point: {int(final_x)}, {int(final_y)}")
            elif event.button == 3: # Right
                obj_id = 2
                color = 'b*'
                logger.info(f"🤖 Robot (ID 2) point: {int(final_x)}, {int(final_y)}")
            else:
                return

            collected_points[obj_id].append([final_x, final_y])
            ax.plot(event.xdata, event.ydata, color, markersize=12)
            fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show(block=True)
        plt.close(fig)
        
        result_prompts = {}
        debug_frame = display_frame.copy()
        
        for obj_id, pts in collected_points.items():
            if not pts: continue
            result_prompts[obj_id] = np.array(pts, dtype=np.float32)
            color = (0, 0, 255) if obj_id == 1 else (255, 0, 0)
            for (px, py) in pts:
                cv2.drawMarker(debug_frame, (int(px), int(py)), color, markerType=cv2.MARKER_CROSS, thickness=2)

        debug_path = output_dir / "debug_prompts.jpg"
        cv2.imwrite(str(debug_path), debug_frame)
        
        return result_prompts if result_prompts else None

    def run_generator(self, video_path: str, detected_rotate_code: Optional[int] = None) -> Generator[Dict, None, None]:
        if self.predictor is None:
            logger.error("❌ Predictor is None. Cannot run.")
            return

        output_dir = Path("outputs") 
        if "outputs" in video_path:
            parts = Path(video_path).parts
            if "outputs" in parts:
                idx = parts.index("outputs")
                output_dir = Path(*parts[:idx+2])

        logger.info(f"Initializing SAM 2 state with {video_path}...")
        try:
            inference_state = self.predictor.init_state(video_path=video_path)
        except Exception as e:
            logger.error(f"Failed to init SAM2 state: {e}")
            raise

        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Cannot read video file: {video_path}")

        # 使用传入的旋转代码，或者使用配置中的，或者自动检测
        effective_rotate_code = detected_rotate_code or self.input_rotate_code
        if effective_rotate_code is None:
            effective_rotate_code = self._detect_video_rotation(video_path)

        # 1. 获取 Prompt
        prompts_dict = {}
        if self.interactive_mode:
            # 传入检测到的旋转代码，确保显示帧与SAM2坐标系一致
            prompts_dict = self._get_interactive_prompt(first_frame, output_dir, effective_rotate_code)
            if prompts_dict is None:
                yield {"status": "cancelled"}
                return
        else:
            h, w = first_frame.shape[:2]
            # 对于自动模式，也需要考虑旋转后的坐标系
            if effective_rotate_code in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                h, w = w, h  # 交换宽高
            prompts_dict = {1: np.array([[w // 2, h // 2]], dtype=np.float32)}

        # 2. 注册 Prompt
        for obj_id, points in prompts_dict.items():
            logger.info(f"👉 Adding {len(points)} points for ID {obj_id}")
            labels = np.array([1] * len(points), dtype=np.int32)
            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )

        # 3. 视频推理
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            frame_masks = {}
            for i, obj_id in enumerate(out_obj_ids):
                mask_tensor = (out_mask_logits[i] > 0.0)
                mask_np = mask_tensor.cpu().numpy().squeeze()
                frame_masks[obj_id] = mask_np

            yield {
                "status": "running",
                "frame_idx": out_frame_idx,
                "masks": frame_masks
            }