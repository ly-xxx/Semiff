"""
Step 4: 资产构建
使用自适应几何绑定器进行"手术级"切割
"""

import sys
import numpy as np
import argparse
import pickle
import trimesh
import yourdfpy
import logging
from pathlib import Path
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parents[1] / "src"))
from semiff.core.geometry import GeometryBinder
from semiff.core.workspace import WorkspaceManager  # [新增]

logger = logging.getLogger("Step4")
logging.basicConfig(level=logging.INFO)


def load_ply_vertices(path):
    """简易PLY加载器"""
    import plyfile
    ply = plyfile.PlyData.read(path)
    v = ply['vertex']
    return np.stack([v['x'], v['y'], v['z']], axis=-1)


def run_step4(cfg_path):
    """运行Step 4: 资产构建"""
    # [新增逻辑] 自动寻找同时包含 Step 2 (ply) 和 Step 3 (npz) 的最新目录
    ws_mgr = WorkspaceManager(cfg_path)
    workspace = ws_mgr.resolve(
        mode="auto",
        required_input_files=["point_cloud.ply", "alignment.npz"]
    )

    # 加载该目录下的冻结配置，保证参数一致性
    runtime_cfg_path = workspace / "runtime_config.yaml"
    conf = OmegaConf.load(runtime_cfg_path if runtime_cfg_path.exists() else cfg_path)

    logger.info(f"📍 Working in: {workspace}")

    # 1. 加载输入数据
    ply_path = workspace / "point_cloud.ply"  # 来自 Step 2
    align_path = workspace / "alignment.npz"  # 来自 Step 3
    urdf_path = Path(conf.data.root_dir) / conf.robot.urdf_rel_path

    if not ply_path.exists():
        logger.error(f"❌ Point cloud not found: {ply_path}")
        return

    logger.info("📦 Loading Assets...")
    points_gs = load_ply_vertices(str(ply_path))
    align_data = np.load(align_path)
    T_world_base = align_data['transform']

    # 2. 将点云变换到机器人基座坐标系
    # 我们需要在该坐标系中进行几何查询
    logger.info(f"   Transforming {len(points_gs)} points to Robot Base Frame...")
    T_inv = np.linalg.inv(T_world_base)

    # 齐次变换
    pts_homo = np.hstack([points_gs, np.ones((len(points_gs), 1))])
    pts_base = (pts_homo @ T_inv.T)[:, :3]

    # 3. 自适应几何绑定
    logger.info("🧩 Initializing Geometry Binder...")
    robot = yourdfpy.URDF.load(urdf_path)

    # 将URDF meshes转换为trimesh字典
    meshes_in_base = {}
    # 使用零配置或来自Step 1的特定配置
    # 假设静态机器人用于资产构建 (零配置通常足够用于T-pose绑定)
    robot.update_cfg(np.zeros(len(robot.actuated_joints)))

    for link in robot.link_names:
        mesh = robot.scene.geometry.get(link)
        if mesh:
            # 应用FK将mesh放到基座坐标系
            T_link = robot.get_transform(link)
            mesh_copy = mesh.copy()
            mesh_copy.apply_transform(T_link)
            meshes_in_base[link] = mesh_copy

    binder = GeometryBinder(
        meshes_in_base,
        method=conf.geometry.binding_method,
        adaptive_percentile=conf.geometry.adaptive_percentile
    )

    is_robot, link_indices = binder.bind(pts_base)

    # 4. 保存结果
    out_file = workspace / "assets.pkl"
    assets = {
        'meta': {
            'urdf': str(urdf_path),
            'scale': align_data['scale']
        },
        'robot': {
            'xyz': points_gs[is_robot],  # 在视觉世界坐标系中保存
            'link_indices': link_indices[is_robot]
        },
        'background': {
            'xyz': points_gs[~is_robot]
        }
    }

    with open(out_file, 'wb') as f:
        pickle.dump(assets, f)
    logger.info(f"✅ Assets exported: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run_step4(args.config)