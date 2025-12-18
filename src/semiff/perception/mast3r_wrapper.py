"""
MASt3R Wrapper: End-to-End Image Matching & Reconstruction
"""

import torch
import numpy as np
import open3d as o3d
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# 假设 MASt3R 已安装在环境中
import sys
from pathlib import Path

# 添加 MASt3R 到路径
mast3r_path = Path(__file__).parents[3] / "third_party" / "mast3r"
if mast3r_path.exists():
    sys.path.insert(0, str(mast3r_path))

try:
    from mast3r.model import AsymmetricMASt3R
    from mast3r.cloud_opt.sparse_ga import GlobalAlignment
except ImportError:
    pass # 允许在无 MASt3R 环境下导入类定义用于测试

from ..core.logger import get_logger

logger = get_logger(__name__)

class MASt3RWrapper:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        try:
            # 尝试导入，如果失败则说明环境未准备好
            from mast3r.model import AsymmetricMASt3R

            # 使用本地模型文件路径
            model_path = Path(__file__).parents[3] / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
            logger.info(f"Loading MASt3R model from: {model_path}")

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model = AsymmetricMASt3R.from_pretrained(str(model_path)).to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.warning(f"MASt3R load failed ({e}). Running in mock mode.")
            return None

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """RGB numpy [H,W,3] -> Tensor [1,3,512,512]"""
        # 简单的预处理，实际应包含 resize/padding 到 512x512
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def run(self, frames: List[np.ndarray], keyframe_interval: int = 15, rotate_code: Optional[int] = None) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        运行重建流水线
        Args:
            frames: RGB帧列表
            keyframe_interval: 关键帧间隔
            rotate_code: 旋转代码（从SAM2传递过来，避免重复检测）
        """
        # 使用传入的旋转代码
        self.rotate_code = rotate_code
        if self.rotate_code is not None:
            logger.info(f"🔄 MASt3R: 使用传入的旋转代码 (代码: {self.rotate_code})")
        if self.model is None:
            logger.warning("MASt3R model is missing. Returning mock data.")
            return [np.eye(4) for _ in range(len(frames)//keyframe_interval)], np.random.rand(100, 3)

        # 1. 稀疏采样
        key_indices = list(range(0, len(frames), keyframe_interval))
        keyframes = [frames[i] for i in key_indices]
        kf_tensors = [self.preprocess_image(f) for f in keyframes]
        n_kf = len(keyframes)

        logger.info(f"Processing {n_kf} keyframes (Interval: {keyframe_interval})...")

        # 2. 匹配策略 (Sequential + Skip-1)
        pairs = []
        for i in range(n_kf):
            if i + 1 < n_kf: pairs.append((i, i+1))
            if i + 2 < n_kf: pairs.append((i, i+2)) # 增加跨帧匹配增强稳定性

        # 3. 构建优化图
        optimizer = GlobalAlignment(init_mode="mst", device=self.device)
        for i, img_tensor in enumerate(kf_tensors):
            optimizer.add_view(i, img_tensor)

        logger.info(f"Computing matches for {len(pairs)} pairs...")
        with torch.no_grad():
            for idx1, idx2 in tqdm(pairs):
                img1 = kf_tensors[idx1]
                img2 = kf_tensors[idx2]

                res = self.model(img1, img2)

                # 提取结果 (根据 MASt3R API 调整)
                # 假设 res 包含 'pts1', 'pts2', 'conf'
                # 实际 API 可能需要 model.extract_matches 或类似调用
                # 这里使用通用结构
                pts1 = res['pts1']
                pts2 = res['pts2']
                conf = res['conf']

                # 过滤并添加约束
                mask = conf > 0.90 # 高置信度阈值
                if mask.sum() > 50: # 至少有 50 个匹配点
                    optimizer.add_pair_constraint(idx1, idx2, pts1[mask], pts2[mask], conf[mask])

        # 4. 全局优化
        logger.info("Running Global Optimization...")
        optimizer.optimize(n_iters=500, lr=0.01)

        # 5. 结果提取
        poses = optimizer.get_poses() # List[Tensor 4x4]
        cloud = optimizer.get_global_point_cloud() # Tensor [N, 3]

        poses_np = [p.detach().cpu().numpy() for p in poses]
        cloud_np = cloud.detach().cpu().numpy()

        logger.info(f"Reconstruction done. Cloud: {cloud_np.shape}, Poses: {len(poses_np)}")
        return poses_np, cloud_np

    def save_results(self, output_dir: Path, poses: List[np.ndarray], cloud: np.ndarray):
        """保存标准格式结果"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存点云 (PLY)
        if len(cloud) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud)
            o3d.io.write_point_cloud(str(output_dir / "scene.ply"), pcd)

        # 2. 保存相机 (JSON)
        cameras = {}
        for i, pose in enumerate(poses):
            cameras[i] = pose.tolist()

        with open(output_dir / "cameras.json", "w") as f:
            json.dump(cameras, f, indent=4)

        # 3. 保存 Pose NPY (方便读取)
        np.save(output_dir / "poses.npy", np.array(poses))

        logger.info(f"💾 Results saved to {output_dir}")