"""
MASt3R Wrapper: End-to-End Image Matching & Reconstruction
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# 假设 MASt3R 已安装在环境中
try:
    from mast3r.model import AsymmetricMASt3R
    from mast3r.fast_nn import fast_reciprocal_nn_matching
    from mast3r.optimization import GlobalAlignment
except ImportError:
    print("Warning: MASt3R library not found. Mocking for structure verification.")

from ..core.logger import get_logger

logger = get_logger(__name__)

class MASt3RWrapper:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        # 实际加载逻辑，这里使用预训练模型名称
        model_name = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        logger.info(f"Loading MASt3R model: {model_name}")
        try:
            model = AsymmetricMASt3R.from_pretrained(model_name).to(self.device)
            model.eval()
            return model
        except:
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

    def run(self, frames: List[np.ndarray], keyframe_interval: int = 10) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        运行完整的重建流水线

        Args:
            frames: 原始视频帧列表
            keyframe_interval: 关键帧采样间隔

        Returns:
            camera_poses: List[4x4 matrix] 每个关键帧的相机位姿 (World -> Camera)
            sparse_cloud: [N, 3] 场景稀疏点云
        """
        if self.model is None:
            raise RuntimeError("MASt3R model not initialized.")

        # 1. 稀疏采样
        key_indices = list(range(0, len(frames), keyframe_interval))
        keyframes = [frames[i] for i in key_indices]
        kf_tensors = [self.preprocess_image(f) for f in keyframes]
        n_kf = len(keyframes)

        logger.info(f"Processing {n_kf} keyframes...")

        # 2. 两两匹配 (Pairwise Matching)
        # 策略：每个关键帧与前一帧匹配 (Sequential Matching)
        # 为了更好的全局一致性，可以使用滑动窗口或全连接，这里演示 Sequential
        pairs = []
        for i in range(n_kf - 1):
            pairs.append((i, i+1))

        # 3. 推理与构建图优化
        # 使用 MASt3R 的 GlobalAlignment 类
        optimizer = GlobalAlignment(init_mode="mst", device=self.device)

        # 将图像注册到优化器
        # 注意: 实际 API 可能需要 features，这里简化为 image 传入
        for i, img_tensor in enumerate(kf_tensors):
            optimizer.add_view(i, img_tensor)

        # 添加两两约束
        logger.info("Computing pairwise matches...")
        with torch.no_grad():
            for idx1, idx2 in pairs:
                img1 = kf_tensors[idx1]
                img2 = kf_tensors[idx2]

                # MASt3R Forward
                res = self.model(img1, img2)

                # 提取点对 (Matches) 和置信度
                # 这是一个简化调用，实际需要处理 output 结构
                pts1 = res['pts1'] # [B, H, W, 3]
                pts2 = res['pts2']
                conf = res['conf']

                # 筛选高置信度点加入优化器
                mask = conf > 0.95
                optimizer.add_pair_constraint(idx1, idx2, pts1[mask], pts2[mask], conf[mask])

        # 4. 全局优化求解
        logger.info("Running Global Optimization...")
        optimizer.optimize(n_iters=300, lr=0.01)

        # 5. 提取结果
        poses = optimizer.get_poses() # [N, 4, 4]
        cloud = optimizer.get_global_point_cloud() # [N_points, 3]

        # 转换为 Numpy
        poses_np = [p.cpu().numpy() for p in poses]
        cloud_np = cloud.cpu().numpy()

        logger.info(f"Reconstruction done. Cloud points: {len(cloud_np)}")
        return poses_np, cloud_np