"""
Step 3: 可微姿态对齐
使用 SoftIoU Loss 进行端到端的 Sim2Real 对齐
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yourdfpy
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

# Path hack to ensure imports work
sys.path.append(str(Path(__file__).parents[1] / "src"))

from semiff.core.math_utils import transform_points, rotation_6d_to_matrix, gpu_mem_guard
from semiff.core.losses import SoftIoULoss
from semiff.core.render import DifferentiableRasterizer
from semiff.core.workspace import WorkspaceManager  # [新增]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Step3")


class PoseOptimizer(nn.Module):
    """姿态优化器 - 可微分地优化机器人姿态"""

    def __init__(self, urdf_path, init_pose, device='cuda'):
        super().__init__()
        self.device = device
        self.robot = yourdfpy.URDF.load(urdf_path)
        self._preload_meshes()

        # 可学习参数
        # 基座旋转 (6D), 平移 (3), 全局缩放 (1)
        self.rot_6d = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], device=device))
        self.trans = nn.Parameter(torch.tensor([init_pose[:3]], dtype=torch.float32, device=device))
        self.scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32, device=device))

    def _preload_meshes(self):
        """预加载URDF meshes用于批渲染"""
        self.meshes = {}
        for link in self.robot.link_names:
            mesh = self.robot.scene.geometry.get(link)
            if mesh:
                self.meshes[link] = {
                    'v': torch.from_numpy(mesh.vertices).float().to(self.device),
                    'f': torch.from_numpy(mesh.faces).int().to(self.device)
                }

    def forward(self, joint_cfg, rasterizer, K):
        """
        前向传播：FK + 变换 + 渲染

        Args:
            joint_cfg: 关节配置字典
            rasterizer: 渲染器实例
            K: (3, 3) 相机内参

        Returns:
            (1, H, W) 预测mask
        """
        # A. 更新正向运动学 (CPU 安全)
        self.robot.update_cfg(joint_cfg)

        all_v = []
        all_f = []
        offset = 0

        # B. 构造世界坐标系下的mesh
        R_base = rotation_6d_to_matrix(self.rot_6d)[0]  # (3,3)

        for link, data in self.meshes.items():
            # T_link_local (FK结果)
            T_fk = torch.from_numpy(self.robot.get_transform(link)).float().to(self.device)

            # 应用可学习基座变换: T_world = T_base @ T_fk
            # 注意：我们优化 T_base，URDF FK是相对于基座的
            # 所以 P_world = s * (R_base @ (T_fk @ P_local) + t_base)

            # 1. 本地 -> 基座坐标系
            v_base = transform_points(data['v'], T_fk)

            # 2. 基座坐标系 -> 世界坐标系 (可学习)
            v_world = (v_base @ R_base.T) * self.scale + self.trans

            all_v.append(v_world)
            all_f.append(data['f'] + offset)
            offset += v_world.shape[0]

        full_v = torch.cat(all_v)
        full_f = torch.cat(all_f)

        # C. 渲染
        # 扩展到批次大小 1
        return rasterizer.render(full_v.unsqueeze(0), full_f, K.unsqueeze(0))


def run_step3(cfg_path):
    """运行Step 3: 姿态对齐"""
    # [新增逻辑] ------------------------------------------------
    # Step 3 依赖 Step 1 的 processed_data.npz
    # 如果直接运行此脚本，自动去最新的 valid workspace 找
    ws_mgr = WorkspaceManager(cfg_path)

    # 策略：寻找包含 'processed_data.npz' 的最新目录
    workspace = ws_mgr.resolve(mode="auto", required_input_files=["processed_data.npz"])

    #以此 workspace 为准加载配置
    runtime_cfg_path = workspace / "runtime_config.yaml"
    if runtime_cfg_path.exists():
        conf = OmegaConf.load(runtime_cfg_path)
    else:
        conf = OmegaConf.load(cfg_path) # Fallback
    # ----------------------------------------------------------

    device = conf.pipeline.device

    # Load Data (从自动解析的 workspace 加载)
    data_path = workspace / "processed_data.npz"
    if not data_path.exists():
        logger.error(f"❌ Input not found: {data_path}")
        return

    logger.info(f"📂 Loading data from: {data_path}")
    data = np.load(data_path)
    gt_mask = torch.from_numpy(data['mask']).float().unsqueeze(0).to(device)
    K = torch.from_numpy(data['intrinsic']).to(device)
    joint_cfg = data['qpos'].item()  # 关节配置
    H, W = data['img_size']

    # 初始化模型
    urdf_path = Path(conf.data.root_dir) / conf.robot.urdf_rel_path
    logger.info(f"🔧 Loading URDF: {urdf_path}")
    model = PoseOptimizer(str(urdf_path), conf.alignment.init_trans, device=device)

    # 设置损失函数和优化器
    loss_fn = SoftIoULoss(smooth=conf.alignment.iou_smooth)
    optimizer = optim.Adam([
        {'params': model.rot_6d, 'lr': conf.optimization.lr_pose},
        {'params': model.trans, 'lr': conf.optimization.lr_trans},
        {'params': model.scale, 'lr': conf.optimization.lr_scale}
    ])

    rasterizer = DifferentiableRasterizer(H, W, device)

    # 优化循环
    logger.info("🚀 Starting Pose Optimization...")
    with gpu_mem_guard():
        pbar = tqdm(range(conf.optimization.iterations))
        for i in pbar:
            optimizer.zero_grad()

            # 配置: {'joint1': 0.0, ...}
            # joint_cfg = data['qpos']  # 实际加载
            joint_cfg = {"joint1": 0.0}  # 占位符

            pred_mask = model(joint_cfg, rasterizer, K)

            loss = loss_fn(pred_mask, gt_mask)
            loss.backward()
            optimizer.step()

            pbar.set_description(".4f")

    # 保存结果
    out_path = Path(conf.pipeline.workspace) / "alignment.npz"

    # 导出参数
    R_final = rotation_6d_to_matrix(model.rot_6d)[0].detach().cpu().numpy()
    t_final = model.trans.detach().cpu().numpy()
    s_final = model.scale.detach().cpu().item()

    T_final = np.eye(4)
    T_final[:3, :3] = R_final * s_final
    T_final[:3, 3] = t_final

    np.savez(out_path, transform=T_final, scale=s_final)
    logger.info(f"✅ Alignment saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run_step3(args.config)