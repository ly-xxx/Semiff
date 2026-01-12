import torch
import numpy as np
import sys
import logging
import cv2
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ==================== 路径配置 ====================
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3] 

MAST3R_ROOT = PROJECT_ROOT / "third_party" / "mast3r"
DUST3R_ROOT = MAST3R_ROOT / "dust3r"

if DUST3R_ROOT.exists() and str(DUST3R_ROOT) not in sys.path:
    sys.path.insert(0, str(DUST3R_ROOT))
if MAST3R_ROOT.exists() and str(MAST3R_ROOT) not in sys.path:
    sys.path.insert(0, str(MAST3R_ROOT))

try:
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from mast3r.model import AsymmetricMASt3R
except ImportError as e:
    logger.error(f"❌ Critical Import Error: {e}")
    inference = make_pairs = global_aligner = GlobalAlignerMode = AsymmetricMASt3R = None

class MASt3RWrapper:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = self._load_model()
        
    def _load_model(self):
        model_path = PROJECT_ROOT / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        if not model_path.exists():
            logger.error(f"❌ Model not found at: {model_path}")
            return None
        logger.info(f"... loading model from {model_path}")
        try:
            model = AsymmetricMASt3R.from_pretrained(str(model_path)).to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return None

    def _estimate_safe_max_frames(self, target_h, target_w):
        """
        🔥 核心功能：根据显存大小自动计算安全帧数上限
        基于经验公式：ViT-Large 512px 下，每帧约消耗 250-300MB 显存 (含图连接开销)
        """
        if not torch.cuda.is_available():
            return 10 # CPU 保底
        
        # 获取显存总量 (GB)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved_sys = 4.0 # 预留给系统、模型权重(1.5G)、PyTorch上下文的显存
        
        available_mem = max(2.0, total_mem - reserved_sys)
        
        # 经验系数：SwIN-2 模式下，每 GB 显存大约能处理 4-5 帧
        # 24GB -> 20GB avail -> ~80-90 frames
        # 12GB -> 8GB avail -> ~32-40 frames
        estimated_frames = int(available_mem * 5)
        
        logger.info(f"💾 VRAM Analysis: Total {total_mem:.1f}GB. Estimated safe capacity: {estimated_frames} frames.")
        return estimated_frames

    def _smart_frame_selection(self, frames: List[np.ndarray], max_frames: int) -> List[int]:
        """
        🔥 核心功能：基于图像内容的智能关键帧选择
        """
        n_frames = len(frames)
        if n_frames <= max_frames:
            return list(range(n_frames))

        logger.info("🧠 Analyzing video content for smart selection...")
        
        # 1. 计算相邻帧的差异分数 (L2 Norm of resized diff)
        # 降采样加速计算
        small_frames = [cv2.resize(f, (64, 64)) for f in frames]
        diffs = []
        for i in range(n_frames - 1):
            d = cv2.absdiff(small_frames[i], small_frames[i+1])
            score = np.mean(d) # 平均像素差异
            diffs.append(score)
        
        # 2. 累积差异法选择
        # 我们希望选出的帧之间累积差异达到一定阈值
        # 这是一个贪心算法：只要变化够大，就选一帧
        
        selected_indices = [0]
        current_idx = 0
        
        # 动态调整阈值，直到选出的帧数接近 max_frames
        # 二分查找合适的阈值
        low_thresh = 0.0
        high_thresh = max(diffs) * 2
        best_indices = np.linspace(0, n_frames-1, max_frames, dtype=int).tolist() # 保底方案
        
        for _ in range(10): # 10次迭代逼近
            mid_thresh = (low_thresh + high_thresh) / 2
            temp_indices = [0]
            accum_diff = 0
            
            for i, d in enumerate(diffs):
                accum_diff += d
                if accum_diff >= mid_thresh:
                    temp_indices.append(i + 1)
                    accum_diff = 0
            
            if len(temp_indices) > max_frames:
                low_thresh = mid_thresh # 阈值太低，选多了
            elif len(temp_indices) < max_frames * 0.8: # 允许少一点，但不能太少
                high_thresh = mid_thresh # 阈值太高，选少了
            else:
                best_indices = temp_indices
                break
        
        # 如果最终还是选多了，均匀降采样
        if len(best_indices) > max_frames:
             sub_indices = np.linspace(0, len(best_indices)-1, max_frames, dtype=int)
             best_indices = [best_indices[i] for i in sub_indices]
             
        logger.info(f"✅ Smart Selection: Reduced {n_frames} -> {len(best_indices)} distinct frames.")
        return best_indices

    def run(self, frames: List[np.ndarray], masks: Optional[List[np.ndarray]] = None, keyframe_interval: int = 1, debug_dir: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None: 
            return [], np.array([]), np.array([])
        
        torch.cuda.empty_cache()

        # --- 1. 准备参数 ---
        raw_h, raw_w = frames[0].shape[:2]
        MODEL_LONG_EDGE = 512.0
        scale = MODEL_LONG_EDGE / max(raw_h, raw_w)
        target_h = (int(raw_h * scale) // 16) * 16
        target_w = (int(raw_w * scale) // 16) * 16

        # --- 2. 智能显存计算 & 帧选择 ---
        # 步骤 A: 计算物理上限
        safe_limit = self._estimate_safe_max_frames(target_h, target_w)
        
        # 步骤 B: 初步按 interval 采样 (先去除完全冗余的)
        raw_indices = np.arange(0, len(frames), keyframe_interval)
        subset_frames = [frames[i] for i in raw_indices]
        
        # 步骤 C: 基于内容的二次筛选
        # 从 subset_frames 中挑出最有代表性的 safe_limit 个
        smart_subset_indices = self._smart_frame_selection(subset_frames, safe_limit)
        
        # 映射回原始索引
        final_indices = [raw_indices[i] for i in smart_subset_indices]
        
        logger.info(f"🚀 Processing {len(final_indices)} frames (Resolution: {target_w}x{target_h})...")

        images = []
        for i, idx in enumerate(final_indices):
            img_tensor = self._preprocess_image_strict(frames[idx], target_h, target_w)
            
            mask_tensor = None
            if masks is not None and idx < len(masks) and masks[idx] is not None:
                m_resized = cv2.resize(masks[idx], (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                mask_tensor = torch.from_numpy(m_resized).bool()
            
            images.append({
                'img': img_tensor,
                'idx': i,
                'true_shape': torch.tensor([[target_h, target_w]], dtype=torch.long), 
                'instance': str(i),
                'mask': mask_tensor 
            })

        if not images: return [], np.array([]), np.array([])

        # --- 3. 动态图策略 (Graph Strategy) ---
        # 如果帧数依然很多 (>60)，为了稳妥，我们降低连接密度
        # swin-2: 每个节点连2个邻居 (dense)
        # swin-1: 每个节点连1个邻居 (sparse, save memory)
        graph_type = "swin-2"
        if len(images) > 60:
            logger.info("⚠️ Frame count is high. Switching to 'swin-1' graph to save VRAM.")
            graph_type = "swin-1"

        pairs = make_pairs(images, scene_graph=graph_type, prefilter=None, symmetrize=True)
        logger.info(f"🔗 Inference on {len(pairs)} pairs (Graph: {graph_type})...")
        
        # --- 4. Inference ---
        # 捕获可能的 OOM，如果失败提示用户降低分辨率
        try:
            output = inference(pairs, self.model, self.device, batch_size=1, verbose=False)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("❌ OOM during inference! Try reducing `MAX_FRAMES` heuristic coefficient or resolution.")
                torch.cuda.empty_cache()
                return [], np.array([]), np.array([])
            raise e

        # --- 5. Global Alignment ---
        mode = GlobalAlignerMode.PointCloudOptimizer if len(images) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=self.device, mode=mode, verbose=False)

        del output, pairs
        torch.cuda.empty_cache()

        if mode == GlobalAlignerMode.PointCloudOptimizer:
            scene.compute_global_alignment(init="mst", niter=500, schedule='linear', lr=0.01)

        # --- 6. 提取数据 (带标签) ---
        # 稍微放宽一点置信度，因为 swin-1 连接较弱
        scene.min_conf_thr = 3.0 
        scene = scene.clean_pointcloud()

        refined_poses = [p.detach().cpu().numpy() for p in scene.get_im_poses()]
        focals = scene.get_focals().detach().cpu().numpy()
        
        intrinsics = []
        scale_w = raw_w / target_w
        scale_h = raw_h / target_h
        for f_val in focals:
            K = np.eye(3, dtype=np.float32)
            K[0,0], K[1,1], K[0,2], K[1,2] = f_val * scale_w, f_val * scale_h, raw_w / 2.0, raw_h / 2.0
            intrinsics.append(K)

        pts_tensor = scene.get_pts3d()
        all_data = []
        
        imgs_tensors = [d['img'] for d in images]
        masks_tensors = [d['mask'] for d in images]

        for i in range(len(images)):
            pts_np = pts_tensor[i].detach().cpu().numpy().reshape(-1, 3)
            rgb_np = imgs_tensors[i].squeeze(0).permute(1, 2, 0).cpu().numpy().reshape(-1, 3)
            rgb_u8 = (rgb_np * 255).astype(np.uint8)
            
            labels_u8 = np.zeros((pts_np.shape[0], 1), dtype=np.uint8)
            if masks_tensors[i] is not None:
                m_flat = masks_tensors[i].flatten().cpu().numpy()
                labels_u8[m_flat] = 1 
            
            norm = np.linalg.norm(pts_np, axis=1)
            valid = (norm > 1e-6) & (np.isfinite(pts_np).all(axis=1))
            
            p_valid, c_valid, l_valid = pts_np[valid], rgb_u8[valid], labels_u8[valid]

            if p_valid.shape[0] > 20000:
                choice = np.random.choice(p_valid.shape[0], 20000, replace=False)
                p_valid, c_valid, l_valid = p_valid[choice], c_valid[choice], l_valid[choice]

            frame_data = np.hstack([p_valid, c_valid.astype(np.float32), l_valid.astype(np.float32)])
            all_data.append(frame_data)

        del scene, imgs_tensors
        torch.cuda.empty_cache()

        if all_data:
            full_cloud = np.concatenate(all_data, axis=0)
        else:
            full_cloud = np.zeros((0, 7))

        return np.array(refined_poses), full_cloud, np.array(intrinsics)

    def _preprocess_image_strict(self, image: np.ndarray, target_h: int, target_w: int):
        import torchvision.transforms.functional as TF
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_resized = TF.resize(img_tensor, [target_h, target_w], antialias=True)
        return img_resized.unsqueeze(0)