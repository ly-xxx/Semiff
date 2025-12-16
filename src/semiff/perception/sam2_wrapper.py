"""
SAM 2 Wrapper: Video Segmentation with Auto-Prompting (Multi-Object Support)
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Generator
from pathlib import Path
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt

from ..core.logger import get_logger

logger = get_logger(__name__)

# ==========================================
# 🔧 点击坐标镜像配置
# ==========================================
CLICK_COORDS_FLIP = None 

class SAM2Wrapper:
    def __init__(self, config: Dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint = config.get("checkpoint", "checkpoints/sam2_hiera_large.pt")
        self.model_cfg = config.get("model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml")
        
        pipeline_cfg = config.get("pipeline", {})
        if hasattr(pipeline_cfg, 'get'):
            self.interactive_mode = pipeline_cfg.get("interactive_mode", False)
            self.input_rotate_code = pipeline_cfg.get("input_rotate_code", None)
        else:
            self.interactive_mode = getattr(pipeline_cfg, 'interactive_mode', False)
            self.input_rotate_code = getattr(pipeline_cfg, 'input_rotate_code', None)

        self.predictor = self._init_model()

    def _init_model(self):
        try:
            import sam2
            from hydra.core.global_hydra import GlobalHydra
            from hydra import initialize_config_module

            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()

            initialize_config_module("sam2", version_base="1.2")
            from sam2.build_sam import build_sam2_video_predictor
            
            if not os.path.exists(self.checkpoint):
                logger.error(f"Checkpoint not found: {self.checkpoint}")
                return None

            return build_sam2_video_predictor(self.model_cfg, self.checkpoint, device=self.device)
        except Exception as e:
            logger.error(f"SAM 2 init failed: {e}")
            return None

    def _get_interactive_prompt(self, frame: np.ndarray, output_dir: Path) -> Dict[int, np.ndarray]:
        """
        返回格式: {1: np.array([[x,y],...]), 2: np.array([[x,y],...])}
        """
        collected_points = {1: [], 2: []} # 1: Object (Left), 2: Robot (Right)
        
        if os.environ.get('DISPLAY', '') == '':
            # 无头模式默认只标记中心为物块
            return {1: np.array([[frame.shape[1] // 2, frame.shape[0] // 2]], dtype=np.float32)}

        try:
            matplotlib.use('TkAgg')
        except:
            pass
            
        logger.info(">>> Left Click: Object (Red) | Right Click: Robot (Blue) | Close window to Finish")
        
        # 1. 准备显示图像 (旋转处理)
        display_frame = frame.copy()
        if self.input_rotate_code is not None:
            display_frame = cv2.rotate(display_frame, self.input_rotate_code)
            logger.info(f"🔄 Rotated display for interaction (Code: {self.input_rotate_code})")

        rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h_disp, w_disp = display_frame.shape[:2]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgb_display)
        ax.set_title("L-Click: Object | R-Click: Robot | Close to Finish", fontsize=15)
        ax.axis('off')

        # 2. 定义点击回调函数
        def on_click(event):
            if event.xdata is None or event.ydata is None: return
            
            # Matplotlib 工具栏点击过滤
            if event.inaxes != ax: return

            click_x, click_y = event.xdata, event.ydata
            
            # 镜像修正
            if CLICK_COORDS_FLIP == 'H':
                click_x = w_disp - 1 - click_x
            elif CLICK_COORDS_FLIP == 'V':
                click_y = h_disp - 1 - click_y

            final_x = np.clip(click_x, 0, w_disp - 1)
            final_y = np.clip(click_y, 0, h_disp - 1)
            
            # 区分左右键
            # event.button: 1=Left, 2=Middle, 3=Right
            if event.button == 1:
                obj_id = 1
                color = 'r*'
                logger.info(f"📍 Object (ID 1) marked at: {int(final_x)}, {int(final_y)}")
            elif event.button == 3:
                obj_id = 2
                color = 'b*'
                logger.info(f"🤖 Robot (ID 2) marked at: {int(final_x)}, {int(final_y)}")
            else:
                return

            collected_points[obj_id].append([final_x, final_y])
            
            # 在图上画点反馈
            ax.plot(event.xdata, event.ydata, color, markersize=12)
            fig.canvas.draw()

        # 绑定事件
        fig.canvas.mpl_connect('button_press_event', on_click)
        
        try:
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
        except:
            pass

        # 阻塞直到窗口关闭
        plt.show(block=True)
        
        # 3. 整理结果
        result_prompts = {}
        
        # 保存 Debug 图
        debug_frame = display_frame.copy()
        
        for obj_id, pts in collected_points.items():
            if not pts: continue
            pts_np = np.array(pts, dtype=np.float32)
            result_prompts[obj_id] = pts_np
            
            # 画 Debug
            color = (0, 0, 255) if obj_id == 1 else (255, 0, 0) # BGR: Red vs Blue
            for (px, py) in pts:
                cv2.drawMarker(debug_frame, (int(px), int(py)), color, 
                               markerType=cv2.MARKER_CROSS, markerSize=30, thickness=3)

        debug_path = output_dir / "debug_click_check.jpg"
        cv2.imwrite(str(debug_path), debug_frame)
        logger.info(f"🛑 DEBUG Image Saved: {debug_path}")

        return result_prompts if result_prompts else None

    def run_generator(self, video_path: str) -> Generator[Dict, None, None]:
        if self.predictor is None:
            raise RuntimeError("SAM 2 not initialized")
            
        output_dir = Path("outputs") 
        if "outputs" in video_path:
            output_dir = Path(video_path).parent.parent / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing inference state...")
        inference_state = self.predictor.init_state(video_path=video_path)
        
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError("Cannot read video")

        # === 1. 获取提示 (支持多目标) ===
        prompts_dict = {} # {obj_id: points}
        
        if self.interactive_mode:
            prompts_dict = self._get_interactive_prompt(first_frame, output_dir)
            if prompts_dict is None:
                yield {"status": "cancelled"}
                return
        else:
            # 默认只给 obj_id 1
            h, w = first_frame.shape[:2]
            prompts_dict = {1: np.array([[w // 2, h // 2]], dtype=np.float32)}

        # === 2. 注册提示到 SAM 2 ===
        # SAM 2 需要为每个 Object ID 分别调用 add_new_points
        for obj_id, points in prompts_dict.items():
            logger.info(f"👉 Registering ID {obj_id} with {len(points)} points.")
            labels = np.array([1] * len(points), dtype=np.int32) # 1 = Positive click
            
            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )

        # === 3. 开始传播 ===
        # propagate_in_video 会返回这一帧里所有被追踪的 objects
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            
            # 解析多目标 Mask
            # out_mask_logits shape: [N, H, W] where N is number of objects
            # out_obj_ids: list of IDs, e.g., [1, 2]
            
            frame_masks = {}
            for i, obj_id in enumerate(out_obj_ids):
                # 提取 mask 并转为 numpy boolean
                mask_tensor = (out_mask_logits[i] > 0.0)
                mask_np = mask_tensor.cpu().numpy().squeeze()
                frame_masks[obj_id] = mask_np

            yield {
                "status": "running",
                "frame_idx": out_frame_idx,
                "masks": frame_masks # 格式: {1: mask, 2: mask}
            }