"""
SEMIFF 统一运行器
串联所有4个步骤：数据预处理 → 3DGS训练 → 姿态对齐 → 资产构建
"""

import argparse
import logging
import sys
import subprocess
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omegaconf import OmegaConf
from semiff.core.workspace import WorkspaceManager

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("SEMIFF")


def run_module(module_path, config_path):
    """Helper to run a python module as a subprocess"""
    cmd = [sys.executable, module_path, "--config", config_path]
    logger.info(f"▶️  Running {module_path}...")
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()

    # 1. 解析工作区
    ws_mgr = WorkspaceManager(args.config)
    raw_conf = OmegaConf.load(args.config)
    mode = raw_conf.pipeline.mode # "new" or "resume"

    # 这里我们只负责"创建"或"确认"工作区存在
    # 如果是 new，Resolve 会创建新文件夹
    # 如果是 resume，Resolve 会返回最新的文件夹
    workspace = ws_mgr.resolve(mode=mode)

    # 2. 冻结配置
    # 无论 new 还是 resume，我们都更新/保存一下当前的配置到该目录
    # 这样所有子步骤读取这个 workspace 下的 runtime_config.yaml 就能拿到最新参数
    frozen_config_path = workspace / "runtime_config.yaml"
    OmegaConf.save(raw_conf, frozen_config_path)

    logger.info(f"🚀 Pipeline Mode: {mode.upper()}")
    logger.info(f"📂 Active Workspace: {workspace}")

    try:
        # Step 1: Preprocess (总是会生成 processed_data.npz)
        # 如果是 Resume 模式，且文件已存在，可以选择跳过？
        # 这里为了逻辑简单，假设 Resume 只是为了复用目录，步骤还是依次检查
        if mode == "resume" and (workspace / "processed_data.npz").exists():
             logger.info("⏩ Step 1 data exists, skipping...")
        else:
             run_module("tools/step1_preprocess.py", args.config)

        # Step 2: 3DGS
        if mode == "resume" and (workspace / "point_cloud.ply").exists():
             logger.info("⏩ Step 2 data exists, skipping...")
        else:
             run_module("tools/step2_train_3dgs.py", args.config)

        # Step 3: Align
        # Step 3 比较快，通常建议重跑，或者是调试的重点
        run_module("tools/step3_align_pose.py", args.config)

        # Step 4: Build Assets
        run_module("tools/step4_build_assets.py", args.config)

        logger.info("🎉 Pipeline Completed Successfully.")
        logger.info(f"👉 Final Assets: {workspace / 'assets.pkl'}")

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Pipeline Aborted. Fix the error and re-run with 'mode: resume' in config.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Critical Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
