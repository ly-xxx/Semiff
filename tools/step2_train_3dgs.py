import argparse
import subprocess
import numpy as np
import logging
from pathlib import Path
from omegaconf import OmegaConf

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parents[1] / "src"))

from semiff.core.workspace import WorkspaceManager  # [新增]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Step2")

def create_dummy_ply(path):
    """生成一个包含随机点的 Mock PLY 文件"""
    header = """ply
format ascii 1.0
element vertex 5000
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(path, 'w') as f:
        f.write(header)
        for _ in range(5000):
            # 生成一个以原点为中心的球体点云
            v = np.random.randn(3)
            v = v / np.linalg.norm(v) * (0.5 + 0.1 * np.random.rand()) # Radius ~0.5m
            f.write(f"{v[0]} {v[1]} {v[2]} 0 0 0 255 0 0\n")

def run_step2(cfg_path):
    # [新增逻辑] ------------------------------------------------
    # Step 2 依赖 Step 1 的 processed_data.npz
    ws_mgr = WorkspaceManager(cfg_path)

    # 寻找包含 processed_data.npz 的最新目录
    workspace = ws_mgr.resolve(mode="auto", required_input_files=["processed_data.npz"])

    # 加载该目录下的冻结配置，保证参数一致性
    runtime_cfg_path = workspace / "runtime_config.yaml"
    conf = OmegaConf.load(runtime_cfg_path if runtime_cfg_path.exists() else cfg_path)
    # ----------------------------------------------------------

    data_dir = Path(conf.data.root_dir)

    out_ply = workspace / "point_cloud.ply"

    # 检查是否启用 Mock 模式
    if conf.training_3dgs.mock:
        logger.warning("🟡 Mock Mode Enabled: Skipping Nerfstudio training.")
        create_dummy_ply(out_ply)
        logger.info(f"✅ Mock PLY generated: {out_ply}")
        return

    # 真实训练逻辑
    logger.info(f"🚀 Starting 3DGS Training on {data_dir}")

    # 1. 训练 (ns-train)
    # 注意：nerfstudio 的输出目录结构比较深，需要后续处理
    ns_output_dir = workspace / "ns_output"
    cmd_train = [
        "ns-train", "splatfacto",
        "--data", str(data_dir),
        "--output-dir", str(ns_output_dir),
        "--experiment-name", "semiff_exp",
        "--pipeline.model.cull_alpha_thresh", str(conf.training_3dgs.cull_alpha_thresh),
        "--max-num-iterations", str(conf.training_3dgs.iterations),
        "--vis", "viewer"
    ]

    try:
        subprocess.check_call(cmd_train)

        # 2. 导出点云 (ns-export)
        # 我们需要找到生成的 config.yml 路径
        # 通常结构: {ns_output_dir}/semiff_exp/splatfacto/{timestamp}/config.yml
        # 这里做一个简单的查找
        config_files = list(ns_output_dir.glob("**/config.yml"))
        if not config_files:
            raise FileNotFoundError("Nerfstudio config.yml not found after training.")

        latest_config = sorted(config_files)[-1] # 取最新的

        cmd_export = [
            "ns-export", "pointcloud",
            "--load-config", str(latest_config),
            "--output-dir", str(workspace),
            "--ply-filename", "point_cloud.ply"
        ]
        subprocess.check_call(cmd_export)
        logger.info(f"✅ Point cloud exported to {out_ply}")

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"❌ 3DGS Training Failed: {e}")
        # Fallback to dummy if training fails? Optional.
        # create_dummy_ply(out_ply)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()
    run_step2(args.config)
