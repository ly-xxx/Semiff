"""
Interactive Joint Pose Recorder
使用Sapien Viewer交互式调整机器人关节角度，并保存为JSON配置文件

Usage:
    python tools/make_joint_json.py --urdf data/example_01/robot/robot.urdf --out data/example_01/config/align_pose.json
"""

import argparse
import json
import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Interactive Joint Pose Recorder")
    parser.add_argument("--urdf", type=str, required=True, help="Path to URDF file")
    parser.add_argument("--out", type=str, default="align_pose.json", help="Output JSON path")
    args = parser.parse_args()

    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        print(f"❌ URDF file not found: {urdf_path}")
        return

    # 1. 初始化 Sapien 引擎
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    # 2. 加载机器人
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    
    # 设置URDF路径的父目录为资源路径，以便找到meshes
    urdf_dir = urdf_path.parent
    robot = loader.load(str(urdf_path), package_dir=str(urdf_dir))
    
    if not robot:
        print(f"❌ Failed to load URDF: {urdf_path}")
        return

    # 设置初始位置
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

    # 添加灯光
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    # 3. 启动 Viewer
    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=1.5, y=0.0, z=1.0)
    viewer.set_camera_rpy(r=0, p=-0.5, y=3.14)

    # 获取活动关节
    active_joints = robot.get_active_joints()
    joint_names = [j.name for j in active_joints]
    qpos = robot.get_qpos()

    print("\n🎮 Interactive Joint Pose Recorder")
    print("=" * 50)
    print(f"Robot: {urdf_path.name}")
    print(f"Active joints: {len(joint_names)}")
    print("\nControls:")
    print("  - Use the GUI sliders to adjust joints")
    print("  - Close the window to SAVE the JSON")
    print("=" * 50)

    # 4. 循环渲染
    while not viewer.closed:
        scene.step()
        scene.update_render()
        viewer.render()

    # 5. 保存结果
    # 在窗口关闭后读取当前关节角
    final_qpos = robot.get_qpos()
    
    pose_dict = {}
    for name, angle in zip(joint_names, final_qpos):
        # 保留4位小数，看起来整洁
        pose_dict[name] = round(float(angle), 4)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(pose_dict, f, indent=4)
    
    print(f"\n✅ Saved joint positions to: {out_path}")
    print("\nJoint configuration:")
    print(json.dumps(pose_dict, indent=2))


if __name__ == "__main__":
    main()