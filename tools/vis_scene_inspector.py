import open3d as o3d
import numpy as np
import sys
from pathlib import Path

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    from yourdfpy import URDF
except ImportError:
    print("❌ Please install yourdfpy: pip install yourdfpy")
    sys.exit(1)

from semiff.core.logger import get_logger

logger = get_logger("scene_inspector")

def load_and_transform_robot(urdf_path, T_world_path):
    """加载并变换机器人模型"""
    if not Path(urdf_path).exists():
        logger.warning(f"URDF file not found: {urdf_path}")
        return None

    if not Path(T_world_path).exists():
        logger.warning(f"Transform file not found: {T_world_path}")
        return None

    try:
        # 加载变换矩阵
        T_world = np.load(T_world_path)

        # 加载 URDF
        robot = URDF.load(urdf_path)

        # 获取 Trimesh 并应用变换
        robot_mesh = robot.scene.dump(concatenate=True)

        # 应用变换 (注意方向：T_world 将视觉坐标系变换到机器人基座坐标系)
        # 这里我们需要将机器人变换到视觉坐标系，所以应用逆变换
        robot_mesh.apply_transform(np.linalg.inv(T_world))

        # 转换为 Open3D
        o3d_robot = o3d.geometry.TriangleMesh()
        o3d_robot.vertices = o3d.utility.Vector3dVector(robot_mesh.vertices)
        o3d_robot.triangles = o3d.utility.Vector3dVector(robot_mesh.faces)
        o3d_robot.compute_vertex_normals()
        o3d_robot.paint_uniform_color([0, 0.5, 1])  # 蓝色机器人

        return o3d_robot

    except Exception as e:
        logger.error(f"Failed to load robot: {e}")
        return None

def main():
    base_dir = Path("outputs")

    geometries = []
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # 1. 加载背景 (完整场景点云，作为背景)
    scene_path = base_dir / "mast3r_result" / "scene.ply"
    if scene_path.exists():
        bg_pcd = o3d.io.read_point_cloud(str(scene_path))
        # 稍微调暗作为背景
        colors = np.asarray(bg_pcd.colors)
        colors = colors * 0.7  # 调暗
        bg_pcd.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(bg_pcd)
        print("✅ Loaded Background Scene")
    else:
        print("⚠️ Background scene not found")

    # 2. 加载提取的物体 (高亮显示)
    obj_path = base_dir / "assets" / "object_raw.ply"
    if obj_path.exists():
        obj_pcd = o3d.io.read_point_cloud(str(obj_path))
        # 标红以区分
        obj_pcd.paint_uniform_color([1, 0, 0])  # 红色物体
        geometries.append(obj_pcd)
        print("✅ Loaded Object Cloud")
    else:
        print("⚠️ Object cloud not found")

    # 3. 加载对齐后的机器人
    urdf_path = "path/to/your/robot.urdf"  # 请修改为真实路径
    t_path = base_dir / "T_world.npy"

    # 如果没有指定URDF路径，尝试从配置中读取
    config_urdf = None
    try:
        import hydra
        from omegaconf import OmegaConf
        cfg = OmegaConf.load("src/semiff/config/defaults.yaml")
        config_urdf = cfg.data.robot_urdf
    except:
        pass

    if config_urdf and Path(config_urdf).exists():
        urdf_path = config_urdf

    robot_mesh = load_and_transform_robot(urdf_path, str(t_path))
    if robot_mesh:
        geometries.append(robot_mesh)
        print("✅ Loaded Aligned Robot")
    else:
        print("⚠️ Robot not loaded (URDF path or transform missing)")

    # 4. 加载对齐后的机器人点云 (用于验证)
    aligned_robot_path = base_dir / "aligned_robot.ply"
    if aligned_robot_path.exists():
        aligned_pcd = o3d.io.read_point_cloud(str(aligned_robot_path))
        geometries.append(aligned_pcd)
        print("✅ Loaded Aligned Robot Cloud (for verification)")

    # 添加坐标系
    geometries.append(coordinate_frame)

    if not geometries:
        print("❌ No geometries to visualize!")
        return

    print(f"\n🎯 Scene Inspector Ready!")
    print(f"Loaded {len(geometries)} geometries:")
    print("- Background scene (gray)")
    print("- Object cloud (red)")
    print("- Robot mesh (blue)")
    print("- Coordinate frame (RGB)")
    print("\nControls:")
    print("- Mouse: Rotate, pan, zoom")
    print("- Ctrl+Mouse: Pan")
    print("- Shift+Mouse: Zoom")
    print("- R: Reset view")

    # 可视化
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Semiff Scene Inspector",
        width=1200,
        height=800,
        left=50,
        top=50,
        point_show_normal=False
    )

if __name__ == "__main__":
    main()