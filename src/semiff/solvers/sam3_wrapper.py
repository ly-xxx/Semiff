"""
SAM 3 Wrapper: Full Video Processing (VRAM-Safe)
✅ 修复：跑完所有帧 (Fix 21 frames bug)
✅ 取消：降采样逻辑 (使用原始分辨率)
"""
import torch
import numpy as np
import logging
import os
import sys
import cv2
import gc
from typing import List, Dict, Generator, Optional
from pathlib import Path

# 必须使用 transformers 库加载你的 sam3 目录结构
try:
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    from transformers.video_utils import load_video
except ImportError:
    print("❌ Critical: Transformers library not found or outdated.")
    sys.exit(1)

# 导入路径管理工具
try:
    from semiff.core.workspace import WorkspaceManager
except ImportError:
    _current_file = Path(__file__).resolve()
    _src_dir = _current_file.parents[2]
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))
    from semiff.core.workspace import WorkspaceManager

logger = logging.getLogger("SAM3Wrapper")

class SAM3Wrapper:
    def __init__(self, config: Dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # === 1. 解析配置与路径 ===
        get_cfg = lambda k, d: config.get(k, d) if hasattr(config, "get") else getattr(config, k, d)
        sam3_cfg = get_cfg("sam3", {})

        checkpoint_path = sam3_cfg.get("checkpoint", "checkpoints/sam3")
        resolved_path = WorkspaceManager.resolve_path(checkpoint_path)
        project_root = WorkspaceManager.find_project_root()

        if not resolved_path.exists():
            logger.error(f"❌ Checkpoint not found: {resolved_path}")
            self.model = None
            return

        cwd = Path.cwd()
        model_path_for_hf = checkpoint_path
        need_chdir = (cwd != project_root)

        self.model_path = str(resolved_path)
        self.prompts_robot = sam3_cfg.get("prompts_robot", ["robot arm"])
        self.prompts_object = sam3_cfg.get("prompts_object", ["object"])

        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        # === 2. 加载模型 ===
        try:
            original_cwd = None
            if need_chdir:
                original_cwd = Path.cwd()
                os.chdir(project_root)

            try:
                self.model = Sam3VideoModel.from_pretrained(
                    model_path_for_hf,
                    torch_dtype=self.dtype,
                    local_files_only=True,
                    trust_remote_code=True
                ).to(self.device)

                self.processor = Sam3VideoProcessor.from_pretrained(
                    model_path_for_hf,
                    local_files_only=True,
                    trust_remote_code=True
                )

                if hasattr(self.model, "enable_memory_efficient_attention"):
                     self.model.enable_memory_efficient_attention()

                logger.info("✅ SAM 3 Model loaded!")
            finally:
                if original_cwd is not None:
                    os.chdir(original_cwd)

        except Exception as e:
            logger.error(f"❌ Init Failed: {e}")
            self.model = None

    def run_generator(self, video_path: str, output_dir: Optional[Path] = None) -> Generator[Dict, None, None]:
        if not self.model:
            return

        logger.info(f"🤖 SAM 3 Processing Video: {video_path}")

        # === 3. 加载视频（取消降采样）===
        try:
            logger.info("🎞️ Loading video frames...")
            all_frames, _ = load_video(video_path)

            if all_frames is None or len(all_frames) == 0:
                logger.error("❌ load_video returned empty frames!")
                return

            total_frames = len(all_frames)
            orig_h, orig_w = all_frames[0].shape[:2]
            logger.info(f"✨ Using original resolution: {orig_w}x{orig_h} | Total frames: {total_frames}")

        except Exception as e:
            logger.error(f"❌ Video load error: {e}")
            return

        # === 4. 初始化 Session（全视频一次性）===
        try:
            inference_session = self.processor.init_video_session(
                video=all_frames,
                inference_device=self.device,
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=self.dtype,
            )
        except Exception as e:
            logger.error(f"❌ Session Init Error: {e}")
            return

        # === 5. 传递提示词 ===
        prompts_to_add = self.prompts_robot + self.prompts_object
        logger.info(f"📝 Applying prompts: {prompts_to_add}")

        for p_text in prompts_to_add:
            try:
                inference_session = self.processor.add_text_prompt(
                    inference_session=inference_session, text=p_text
                )
            except Exception as e:
                logger.error(f"⚠️ Prompt error '{p_text}': {e}")

        # === 6. 传播（全视频）===
        frame_count = 0
        try:
            logger.info("🚀 Starting propagation...")

            iterator = self.model.propagate_in_video_iterator(
                inference_session=inference_session,
                max_frame_num_to_track=None
            )

            for i, model_outputs in enumerate(iterator):
                # 显存守护：每 5 帧清理一次
                if i % 5 == 0:
                    torch.cuda.empty_cache()

                processed_outputs = self.processor.postprocess_outputs(
                    inference_session, model_outputs
                )

                pred_masks = processed_outputs["masks"].cpu().numpy()
                obj_ids = processed_outputs["object_ids"].cpu().numpy()

                frame_masks = {}
                for idx, obj_id in enumerate(obj_ids):
                    raw_mask = pred_masks[idx]

                    if isinstance(raw_mask, np.ndarray):
                        m = (raw_mask > 0.0).astype(np.uint8) if raw_mask.dtype != bool else raw_mask.astype(np.uint8)
                    else:
                        m = np.array(raw_mask > 0).astype(np.uint8)

                    if m.ndim > 2:
                        m = m.squeeze()
                    if m.ndim == 3:
                        m = m[0]

                    # 确保尺寸匹配（不降采样时一般不需要，但保险起见）
                    if m.shape[:2] != (orig_h, orig_w):
                        m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                    frame_masks[int(obj_id)] = m

                yield {
                    "status": "running",
                    "frame_idx": model_outputs.frame_idx,
                    "masks": frame_masks
                }
                frame_count += 1

        except Exception as e:
            logger.error(f"❌ Propagation Error: {e}")
            import traceback
            traceback.print_exc()

            if "out of memory" in str(e).lower():
                logger.error("💥 显存不足！可以考虑重新开启分块或降采样")

        finally:
            del inference_session
            del all_frames
            gc.collect()
            torch.cuda.empty_cache()

        logger.info(f"✅ Success: Processed {frame_count} frames.")