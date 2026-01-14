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
import sys

# 导入统一路径管理工具
try:
    from semiff.core.workspace import WorkspaceManager
except ImportError:
    # Fallback: 如果导入失败，添加 src 到路径
    _current_file = Path(__file__).resolve()
    _src_dir = _current_file.parents[2]  # src/
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))
    from semiff.core.workspace import WorkspaceManager

logger = logging.getLogger("SAM2Wrapper")

class SAM2Wrapper:
    def __init__(self, config: Dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        get_cfg = lambda k, d: config.get(k, d) if hasattr(config, "get") else getattr(config, k, d)
        
        # 从 sam2 子配置中读取参数
        sam2_cfg = get_cfg("sam2", {})
        get_sam2_cfg = lambda k, d: sam2_cfg.get(k, d) if hasattr(sam2_cfg, "get") else getattr(sam2_cfg, k, d)
        
        # 🔧 使用统一路径解析工具
        checkpoint_rel = get_sam2_cfg("checkpoint", "checkpoints/sam2_hiera_large.pt")
        model_cfg_rel = get_sam2_cfg("model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml")
        
        self.checkpoint = str(WorkspaceManager.resolve_path(checkpoint_rel))
        self.model_cfg = str(WorkspaceManager.resolve_path(model_cfg_rel))
        
        logger.info(f"🔍 Resolved checkpoint: {self.checkpoint}")
        logger.info(f"🔍 Resolved model_cfg: {self.model_cfg}")

        pipeline_cfg = get_cfg("pipeline", {})
        get_p_cfg = lambda k, d: pipeline_cfg.get(k, d) if hasattr(pipeline_cfg, "get") else getattr(pipeline_cfg, k, d)
        
        self.interactive_mode = get_p_cfg("interactive_mode", False)
        self.input_rotate_code = get_p_cfg("input_rotate_code", None)

        self.predictor = self._init_model()

    def _init_model(self):
        try:
            import sam2
            from hydra.core.global_hydra import GlobalHydra
            from hydra import initialize_config_module, initialize_config_dir

            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()

            from sam2.build_sam import build_sam2_video_predictor
            
            if not os.path.exists(self.checkpoint):
                logger.error(f"❌ Checkpoint not found: {self.checkpoint}")
                return None

            logger.info(f"Loading SAM 2 model...")
            
            if os.path.exists(self.model_cfg):
                abs_config_path = os.path.abspath(self.model_cfg)
                config_dir = os.path.dirname(abs_config_path)
                config_name = os.path.basename(abs_config_path)
                initialize_config_dir(config_dir=config_dir, version_base="1.2")
                return build_sam2_video_predictor(config_name, self.checkpoint, device=self.device)
            else:
                fallback_name = os.path.basename(self.model_cfg).replace("sam2.1", "sam2")
                logger.warning(f"🔄 Falling back to package config: {fallback_name}")
                initialize_config_module("sam2", version_base="1.2")
                return build_sam2_video_predictor(fallback_name, self.checkpoint, device=self.device)

        except ImportError:
            logger.error("❌ SAM 2 library not installed.")
            return None
        except Exception as e:
            logger.error(f"❌ SAM 2 init failed: {e}")
            return None

    def _detect_video_rotation(self, video_path: str) -> Optional[int]:
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
            elif rotate == -90: return cv2.ROTATE_90_COUNTERCLOCKWISE
            return None
        except Exception:
            return None

    def _inverse_rotate_points(self, points: np.ndarray, h_vis: int, w_vis: int, rotate_code: Optional[int]) -> np.ndarray:
        """
        将可视化坐标映射回 Raw 视频坐标。
        修正逻辑：如果 Raw 图像已经被 OpenCV 自动旋转（即长宽比和 Vis 一致），则直接使用原坐标。
        """
        # 1. 自动检测是否需要旋转
        # 我们无法直接在这里获取 Raw 尺寸，但我们可以通过 rotate_code 推断。
        # 如果 rotate_code 是 90度，通常意味着 Raw 是横屏，Vis 是竖屏。
        # 但如果 OpenCV 已经自动旋转了，那么我们在外部看到的 frame.shape 已经是竖屏了。

        # 最稳妥的方式：直接返回 points。
        # 因为在 run_generator 中，我们是这样获取 interactive prompt 的：
        # prompts_dict = self._get_interactive_prompt(first_frame, ...)
        # 这里的 first_frame 是从 cap.read() 读出来的。
        # 如果 cap.read() 读出来的是竖屏（正如你的 debug 图所示），
        # 那么点击坐标(Vis) 和 SAM2 输入坐标(Raw) 就是同一个坐标系！

        return points

        # 下面的旧代码全部注释掉或删除，因为你的 OpenCV 环境会自动处理旋转
        """
        if rotate_code is None:
            return points

        new_pts = points.copy()

        # ... (旧的复杂旋转逻辑) ...

        return new_pts
        """

    def _get_interactive_prompt(self, frame: np.ndarray, output_dir: Path, rotate_code: Optional[int] = None) -> Dict[int, np.ndarray]:
        collected_points = {1: [], 2: []}
        
        if os.environ.get('DISPLAY', '') == '':
            logger.warning("⚠️ No DISPLAY detected. Using center point default.")
            h, w = frame.shape[:2]
            return {1: np.array([[w // 2, h // 2]], dtype=np.float32)}

        try:
            matplotlib.use('TkAgg')
        except:
            pass
            
        logger.info("\n>>> INTERACTIVE MODE <<<\n[Left Click]: Object (Red)\n[Right Click]: Robot (Blue)\n[Close Window]: Start Segmentation\n")
        
        # 1. 生成可视化帧 (竖屏)
        display_frame = frame.copy()
        if rotate_code is not None:
            display_frame = cv2.rotate(display_frame, rotate_code)
            
        rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h_disp, w_disp = display_frame.shape[:2]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(rgb_display)
        ax.set_title(f"Click Objects (Rot: {rotate_code})")
        ax.axis('off')

        def on_click(event):
            if event.xdata is None or event.ydata is None: return
            if event.inaxes != ax: return

            click_x, click_y = event.xdata, event.ydata
            
            if event.button == 1:
                obj_id = 1
                color = 'r*'
                logger.info(f"📍 Obj(1) Vis: {int(click_x)}, {int(click_y)}")
            elif event.button == 3:
                obj_id = 2
                color = 'b*'
                logger.info(f"🤖 Rob(2) Vis: {int(click_x)}, {int(click_y)}")
            else:
                return

            collected_points[obj_id].append([click_x, click_y])
            ax.plot(event.xdata, event.ydata, color, markersize=12)
            fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show(block=True)
        plt.close(fig)
        
        # 2. 坐标映射
        result_prompts = {}
        h_raw, w_raw = frame.shape[:2]
        
        # Debug 画布 (竖屏 Vis) - 验证你的点击
        debug_vis_check = display_frame.copy()

        for obj_id, pts in collected_points.items():
            if not pts: continue
            pts_np = np.array(pts, dtype=np.float32)
            
            # === 执行映射 ===
            raw_pts = self._inverse_rotate_points(pts_np, h_disp, w_disp, rotate_code)
            
            # === 越界保护 (Clip to Raw Dimensions) ===
            raw_pts[:, 0] = np.clip(raw_pts[:, 0], 0, w_raw - 1)
            raw_pts[:, 1] = np.clip(raw_pts[:, 1], 0, h_raw - 1)
            
            result_prompts[obj_id] = raw_pts
            
            # 在 Debug 图上画点
            color = (0, 0, 255) if obj_id == 1 else (255, 0, 0)
            for (vx, vy) in pts:
                vx_i = int(np.clip(vx, 0, w_disp-1))
                vy_i = int(np.clip(vy, 0, h_disp-1))
                cv2.drawMarker(debug_vis_check, (vx_i, vy_i), color, markerType=cv2.MARKER_STAR, markerSize=30, thickness=3)

        # 3. 保存 Debug 图
        if output_dir:
            if not output_dir.exists(): output_dir.mkdir(parents=True, exist_ok=True)
            debug_save_path = output_dir / "debug_vis_checks.jpg"
            cv2.imwrite(str(debug_save_path), debug_vis_check)
            print(f"✅ [DEBUG] Saved: {debug_save_path}")

        # === 新增：Raw 坐标验证 (True Verification) ===
        # 这张图显示的是传给 SAM2 的真实坐标是否落在了横屏图像的正确物体上
        debug_raw_check = frame.copy() # 这是原始 Raw 横屏图
        h_raw, w_raw = frame.shape[:2]

        for obj_id, raw_pts in result_prompts.items():
            color = (0, 0, 255) if obj_id == 1 else (255, 0, 0)
            for (rx, ry) in raw_pts:
                rx_i = int(np.clip(rx, 0, w_raw-1))
                ry_i = int(np.clip(ry, 0, h_raw-1))
                # 画一个大叉，确保看清
                cv2.drawMarker(debug_raw_check, (rx_i, ry_i), color, markerType=cv2.MARKER_CROSS, markerSize=30, thickness=3)

        if output_dir:
            raw_debug_path = output_dir / "debug_RAW_checks.jpg"
            cv2.imwrite(str(raw_debug_path), debug_raw_check)
            print(f"✅ [DEBUG] Saved RAW Verification: {raw_debug_path} (Check THIS one!)")

        return result_prompts if result_prompts else None

    def run_generator(self, video_path: str, output_dir: Optional[Path] = None) -> Generator[Dict, None, None]:
        if self.predictor is None:
            logger.error("❌ Predictor is None. Cannot run.")
            return

        # output_dir 必须由调用者提供（工作区目录），不使用回退逻辑
        if output_dir is None:
            raise ValueError("output_dir is required and must be set to workspace directory")
        
        if not output_dir.exists(): 
            output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing SAM 2 state with {video_path}...")
        inference_state = self.predictor.init_state(video_path=video_path)
        
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        cap.release()

        if not ret: raise RuntimeError(f"Cannot read video file: {video_path}")

        effective_rotate_code = self.input_rotate_code
        if effective_rotate_code is None:
            effective_rotate_code = self._detect_video_rotation(video_path)

        prompts_dict = {}
        if self.interactive_mode:
            prompts_dict = self._get_interactive_prompt(first_frame, output_dir, effective_rotate_code)
            if prompts_dict is None:
                yield {"status": "cancelled"}
                return
        else:
            h, w = first_frame.shape[:2]
            prompts_dict = {1: np.array([[w // 2, h // 2]], dtype=np.float32)}

        for obj_id, points in prompts_dict.items():
            labels = np.array([1] * len(points), dtype=np.int32)
            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )

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