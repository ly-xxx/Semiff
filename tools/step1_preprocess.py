import argparse
import json
import numpy as np
import logging
from pathlib import Path
from omegaconf import OmegaConf

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parents[1] / "src"))

from semiff.core.workspace import WorkspaceManager  # [新增]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Step1")

def run_step1(cfg_path):
    # [新增逻辑] ------------------------------------------------
    # Step 1 通常是起点，但为了保持一致性，我们也使用 WorkspaceManager
    ws_mgr = WorkspaceManager(cfg_path)

    # Step 1 通常需要创建新目录，除非是 resume 模式
    raw_conf = OmegaConf.load(cfg_path)
    mode = raw_conf.pipeline.mode

    workspace = ws_mgr.resolve(mode=mode)  # 对于 new 会创建，对于 resume 会找最新的

    # 加载该目录下的配置（如果是 resume）或原始配置
    runtime_cfg_path = workspace / "runtime_config.yaml"
    conf = OmegaConf.load(runtime_cfg_path if runtime_cfg_path.exists() else cfg_path)
    # ----------------------------------------------------------

    root_dir = Path(conf.data.root_dir)
    json_path = root_dir / conf.data.robot_config

    logger.info(f"📖 Reading Robot Config: {json_path}")

    # 1. Load Robot Joint Config
    if not json_path.exists():
        logger.error(f"❌ Config not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        qpos_dict = json.load(f)

    logger.info(f"   Joints: {qpos_dict}")

    # 2. Mocking Vision Data (Masks & Intrinsics)
    # 在实际生产中，这里应该加载 SAM2 模型处理 conf.data.video_path
    logger.info("⚠️ Generating Mock Vision Data (Replace with SAM2 in production)")

    H, W = 720, 1280
    # 模拟一个简单的相机内参
    K = np.array([[1000., 0., W/2], [0., 1000., H/2], [0., 0., 1.]])

    # 模拟一个 Mask (假设机器人是一个方块)
    # 真实场景中，这里是 sam2_predictor.predict(video)
    mask = np.zeros((H, W), dtype=np.float32)
    mask[200:500, 400:700] = 1.0

    # 3. Save to Workspace
    out_path = workspace / "processed_data.npz"
    np.savez(out_path,
             qpos=qpos_dict, # Save dict directly (allow_pickle=True)
             mask=mask,
             intrinsic=K,
             img_size=(H, W))

    logger.info(f"✅ Preprocessed data saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()
    run_step1(args.config)
