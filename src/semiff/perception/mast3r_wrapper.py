import json
import torch
import numpy as np
import sys
import logging
import cv2
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ==================== 1. 路径与全局变量定义 (修复点) ====================
# 必须在这里定义 PROJECT_ROOT，以便类内部可以访问
CURRENT_FILE = Path(__file__).resolve()
# 假设结构是 src/semiff/perception/mast3r_wrapper.py
# parents[0]=perception, [1]=semiff(pkg), [2]=src, [3]=ProjectRoot
PROJECT_ROOT = CURRENT_FILE.parents[3] 

MAST3R_ROOT = PROJECT_ROOT / "third_party" / "mast3r"
DUST3R_ROOT = MAST3R_ROOT / "dust3r"

# 添加到系统路径以便 import
if DUST3R_ROOT.exists() and str(DUST3R_ROOT) not in sys.path:
    sys.path.insert(0, str(DUST3R_ROOT))
if MAST3R_ROOT.exists() and str(MAST3R_ROOT) not in sys.path:
    sys.path.insert(0, str(MAST3R_ROOT))

# ==================== 2. Import 模型 ====================
try:
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from mast3r.model import AsymmetricMASt3R
except ImportError as e:
    logger.error(f"❌ Critical Import Error: {e}")
    inference = make_pairs = global_aligner = GlobalAlignerMode = AsymmetricMASt3R = None

# ==================== 3. 辅助类 ====================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# ==================== 4. 主类定义 ====================
class MASt3RWrapper:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = self._load_model()
        
    def _load_model(self):
        # 这里使用了全局变量 PROJECT_ROOT
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

    def run(self, frames: List[np.ndarray], keyframe_interval: int = 2, debug_dir: Optional[Path] = None) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        运行 MASt3R 重建流程，包含显存优化的 Global Alignment
        """
        if self.model is None: 
            logger.error("❌ Model is not loaded, skipping execution.")
            return [], np.array([])
        
        # --- 显存清理：开始前先清理 ---
        torch.cuda.empty_cache()

        # --- 1. 准备图像 ---
        raw_h, raw_w = frames[0].shape[:2]
        MODEL_LONG_EDGE = 512.0
        scale = MODEL_LONG_EDGE / max(raw_h, raw_w)
        target_h = (int(raw_h * scale) // 16) * 16
        target_w = (int(raw_w * scale) // 16) * 16

        images = []
        # 限制最大帧数，保护显存 (RTX 3090/4090 建议 40-50 帧)
        MAX_FRAMES = 42 
        key_indices = list(range(0, len(frames), keyframe_interval))
        
        if len(key_indices) > MAX_FRAMES:
            logger.warning(f"⚠️ Limiting frames from {len(key_indices)} to {MAX_FRAMES} for memory safety.")
            key_indices = np.linspace(0, len(frames)-1, MAX_FRAMES, dtype=int).tolist()

        logger.info(f"Preparing {len(key_indices)} frames for Global Alignment...")

        for i, idx in enumerate(key_indices):
            img_tensor = self._preprocess_image_strict(frames[idx], target_h, target_w)
            images.append({
                'img': img_tensor,
                'idx': i,
                # 标准化 true_shape 格式为 tensor
                'true_shape': torch.tensor([[target_h, target_w]], dtype=torch.long), 
                'instance': str(i)
            })

        if not images: return [], np.array([])

        # --- 2. 构建 Pair ---
        # swin-2 意味着每个节点连接 2 层的邻居，比全连接省内存
        pairs = make_pairs(images, scene_graph="swin-2", prefilter=None, symmetrize=True)

        logger.info(f"🚀 Running Inference on {len(pairs)} pairs...")

        # --- 3. Inference & Global Alignment ---
        # 这一步产生大量中间数据
        output = inference(pairs, self.model, self.device, batch_size=1, verbose=False)

        # 初始化 GlobalAligner
        mode = GlobalAlignerMode.PointCloudOptimizer if len(images) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=self.device, mode=mode, verbose=False)

        # 🔥【关键内存优化】：scene 初始化后，output 中的 heavy data 已经被 scene 接管或不再需要
        # 必须显式删除 output 并清空缓存，否则显存会双倍占用，极易 OOM
        del output
        del pairs
        torch.cuda.empty_cache()

        if mode == GlobalAlignerMode.PointCloudOptimizer:
            scene.compute_global_alignment(
                init="mst", 
                niter=300, 
                schedule='linear', 
                lr=0.01
            )

        # --- 4. 提取结果 ---
        # 设定置信度阈值，过滤掉不可靠的点（比如天空、反光区域）
        scene.min_conf_thr = 5.0 
        
        # 这一步会根据 min_conf_thr 进行裁剪
        scene = scene.clean_pointcloud()

        # 提取位姿 (World_T_Camera)
        refined_poses = [p.detach().cpu().numpy() for p in scene.get_im_poses()]
        
        # 提取点云
        # GlobalAligner 已经把所有点都转到了 World 坐标系
        pts_tensor = scene.get_pts3d() # [N_imgs, H, W, 3]
        
        all_pts = []
        all_cols = []
        
        # 只需要用来取颜色的 raw tensor
        imgs_tensors = [d['img'] for d in images]

        for i in range(len(images)):
            # 获取坐标
            pts_np = pts_tensor[i].detach().cpu().numpy().reshape(-1, 3)
            
            # 获取颜色
            rgb_np = imgs_tensors[i].squeeze(0).permute(1, 2, 0).cpu().numpy().reshape(-1, 3)
            rgb_u8 = (rgb_np * 255).astype(np.uint8)

            # 过滤逻辑：GlobalAligner 会把被过滤的点设为 0 或 inf
            # 我们只需要保留非零且有效的点
            norm = np.linalg.norm(pts_np, axis=1)
            valid = (norm > 1e-6) & (np.isfinite(pts_np).all(axis=1))
            
            p_valid = pts_np[valid]
            c_valid = rgb_u8[valid]

            # 降采样：每张图最多贡献 2w 个点，防止总点云过大
            if p_valid.shape[0] > 20000:
                choice = np.random.choice(p_valid.shape[0], 20000, replace=False)
                p_valid = p_valid[choice]
                c_valid = c_valid[choice]

            all_pts.append(p_valid)
            all_cols.append(c_valid)

        # 清理 scene
        del scene
        del imgs_tensors
        torch.cuda.empty_cache()

        # 合并
        if all_pts:
            final_xyz = np.concatenate(all_pts, axis=0)
            final_rgb = np.concatenate(all_cols, axis=0)
            # 拼接 xyz 和 rgb (Nx6)
            full_cloud = np.hstack([final_xyz, final_rgb.astype(np.float32)])
        else:
            full_cloud = np.zeros((0, 6))

        return np.array(refined_poses), full_cloud

    def _preprocess_image_strict(self, image: np.ndarray, target_h: int, target_w: int):
        import torchvision.transforms.functional as TF
        # [H, W, 3] -> [3, H, W], CPU float
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        # Resize
        img_resized = TF.resize(img_tensor, [target_h, target_w], antialias=True)
        # [1, 3, H, W]
        return img_resized.unsqueeze(0)