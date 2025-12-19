import sys
from pathlib import Path
import numpy as np
import open3d as o3d
import json
import pickle

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    from yourdfpy import URDF
except ImportError:
    print("Warning: yourdfpy not installed. Robot visualization will be disabled.")

from semiff.core.logger import get_logger

logger = get_logger("step5_integrate")

def load_and_validate_outputs():
    """
    加载并验证所有步骤的输出
    返回验证结果和加载的数据
    """
    logger.info("🔍 Validating pipeline outputs...")

    base_dir = Path("outputs")
    validation_results = {}
    loaded_data = {}

    # 1. 检查 Step 1 输出
    mast3r_dir = base_dir / "mast3r_result"
    scene_ply = mast3r_dir / "scene.ply"
    poses_npy = mast3r_dir / "poses.npy"

    validation_results["step1"] = {
        "scene_ply": scene_ply.exists(),
        "poses_npy": poses_npy.exists()
    }

    if scene_ply.exists():
        pcd = o3d.io.read_point_cloud(str(scene_ply))
        loaded_data["scene_pcd"] = pcd
        validation_results["step1"]["point_count"] = len(pcd.points)
        logger.info(f"✅ Step 1: Scene with {len(pcd.points)} points")
    else:
        logger.warning("❌ Step 1: Scene PLY not found")

    # 2. 检查 Step 2 输出
    masks_obj_dir = base_dir / "masks_object"
    masks_robot_dir = base_dir / "masks_robot"
    vis_video = base_dir / "final_vis_ffmpeg.mp4"

    validation_results["step2"] = {
        "masks_object": masks_obj_dir.exists(),
        "masks_robot": masks_robot_dir.exists(),
        "vis_video": vis_video.exists()
    }

    if masks_obj_dir.exists():
        obj_masks = list(masks_obj_dir.glob("*.png"))
        validation_results["step2"]["object_masks_count"] = len(obj_masks)
        logger.info(f"✅ Step 2: {len(obj_masks)} object masks")

    if masks_robot_dir.exists():
        robot_masks = list(masks_robot_dir.glob("*.png"))
        validation_results["step2"]["robot_masks_count"] = len(robot_masks)
        logger.info(f"✅ Step 2: {len(robot_masks)} robot masks")

    # 3. 检查 Step 3 输出 (Assets)
    assets_dir = base_dir / "assets"
    obj_ply = assets_dir / "object_raw.ply"
    obj_mesh = assets_dir / "object.obj"
    obj_collision = assets_dir / "object_collision.obj"
    asset_info = assets_dir / "asset_info.json"

    validation_results["step3"] = {
        "object_ply": obj_ply.exists(),
        "object_mesh": obj_mesh.exists(),
        "object_collision": obj_collision.exists(),
        "asset_info": asset_info.exists()
    }

    if obj_ply.exists():
        obj_pcd = o3d.io.read_point_cloud(str(obj_ply))
        loaded_data["object_pcd"] = obj_pcd
        validation_results["step3"]["object_points"] = len(obj_pcd.points)
        logger.info(f"✅ Step 3: Object with {len(obj_pcd.points)} points")

    # 4. 检查 Step 4 输出 (Robot)
    gs_training_dir = base_dir / "robot_gs_training"
    gs_ply = gs_training_dir / "robot_gs.ply"
    binding_dir = assets_dir / "robot_binding"
    binding_data = binding_dir / "binding_data.pkl"

    validation_results["step4"] = {
        "gs_training": gs_training_dir.exists(),
        "gs_ply": gs_ply.exists(),
        "binding_data": binding_data.exists()
    }

    # 尝试加载机器人数据
    robot_data = None
    if binding_data.exists():
        try:
            with open(binding_data, 'rb') as f:
                robot_data = pickle.load(f)
            validation_results["step4"]["gaussians"] = robot_data["num_gaussians"]
            logger.info(f"✅ Step 4: 3DGS binding with {robot_data['num_gaussians']} Gaussians")
        except Exception as e:
            logger.warning(f"Could not load binding data: {e}")

    return validation_results, loaded_data

def create_integrated_visualization(loaded_data, validation_results):
    """
    创建集成的可视化
    """
    logger.info("🎨 Creating integrated visualization...")

    geometries = []
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # 1. 添加背景场景
    if "scene_pcd" in loaded_data:
        bg_pcd = loaded_data["scene_pcd"]
        # 调暗作为背景
        colors = np.asarray(bg_pcd.colors)
        colors = colors * 0.6
        bg_pcd.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(bg_pcd)
        logger.info("✅ Added background scene")

    # 2. 添加物体
    if "object_pcd" in loaded_data:
        obj_pcd = loaded_data["object_pcd"]
        obj_pcd.paint_uniform_color([1, 0, 0])  # 红色物体
        geometries.append(obj_pcd)
        logger.info("✅ Added object point cloud")

    # 3. 添加机器人 (3DGS绑定)
    binding_data_path = Path("outputs/assets/robot_binding/binding_data.pkl")
    if binding_data_path.exists():
        try:
            # 这里可以添加 3DGS 渲染器的可视化
            # 暂时只显示绑定统计
            with open(binding_data_path, 'rb') as f:
                binding_data = pickle.load(f)
            logger.info(f"✅ 3DGS binding ready: {len(binding_data['link_names'])} links, {binding_data['num_gaussians']} Gaussians")
        except Exception as e:
            logger.warning(f"Could not load 3DGS binding data: {e}")

    # 添加坐标系
    geometries.append(coordinate_frame)

    return geometries

def generate_summary_report(validation_results, loaded_data):
    """
    生成摘要报告
    """
    logger.info("📊 Generating summary report...")

    report = {
        "pipeline_status": "completed",
        "steps_completed": 0,
        "total_points": 0,
        "assets_generated": [],
        "recommendations": []
    }

    # 计算完成步骤
    for step, results in validation_results.items():
        if isinstance(results, dict):
            if any(results.values()):
                report["steps_completed"] += 1

    # 统计点数
    if "scene_pcd" in loaded_data:
        report["total_points"] += len(loaded_data["scene_pcd"].points)
    if "object_pcd" in loaded_data:
        report["total_points"] += len(loaded_data["object_pcd"].points)

    # 检查资产
    assets_dir = Path("outputs/assets")
    if (assets_dir / "object.obj").exists():
        report["assets_generated"].append("object_mesh")
    if (assets_dir / "object_collision.obj").exists():
        report["assets_generated"].append("object_collision")

    # 生成建议
    if validation_results["step4"]["3dgs"]["binding_data"]:
        report["recommendations"].append("Ready for differentiable physics simulation with 3DGS binding")
    elif validation_results["step4"]["traditional"]["t_world"]:
        report["recommendations"].append("Traditional alignment completed, consider upgrading to 3DGS for better rendering")

    if not validation_results["step3"]["object_mesh"]:
        report["recommendations"].append("Consider improving object segmentation for better mesh generation")

    return report

def main():
    logger.info("🚀 [Step 5] Integration & Validation...")

    # 1. 验证所有输出
    validation_results, loaded_data = load_and_validate_outputs()

    # 2. 生成摘要报告
    report = generate_summary_report(validation_results, loaded_data)

    print("\n" + "="*60)
    print("🎯 PIPELINE VALIDATION REPORT")
    print("="*60)
    print(f"Steps completed: {report['steps_completed']}/5")
    print(f"Total points processed: {report['total_points']:,}")
    print(f"Assets generated: {', '.join(report['assets_generated']) if report['assets_generated'] else 'None'}")

    if report["recommendations"]:
        print("\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")

    print("\n" + "="*60)

    # 3. 保存详细报告
    report_path = Path("outputs/pipeline_report.json")
    with open(report_path, 'w') as f:
        json.dump({
            "validation_results": validation_results,
            "summary": report,
            "timestamp": str(np.datetime64('now'))
        }, f, indent=2, default=str)

    logger.info(f"✅ Detailed report saved to {report_path}")

    # 4. 创建集成可视化
    geometries = create_integrated_visualization(loaded_data, validation_results)

    if geometries:
        logger.info("🎨 Launching integrated visualization...")
        print("\n🎮 Controls:")
        print("  • Mouse: Rotate, pan, zoom")
        print("  • R: Reset view")
        print("  • Close window to exit")

        try:
            o3d.visualization.draw_geometries(
                geometries,
                window_name="Semiff - Integrated Scene Inspector",
                width=1200,
                height=800,
                left=50,
                top=50,
                point_show_normal=False
            )
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
            logger.info("You can run individual visualization scripts instead")
    else:
        logger.warning("No geometries available for visualization")

    logger.info("🎯 [Step 5] Integration completed!")
    print("\n🎉 Pipeline execution finished!")
    print("   Check outputs/ for all generated assets")
    print("   Run individual steps if you need to iterate")

if __name__ == "__main__":
    main()