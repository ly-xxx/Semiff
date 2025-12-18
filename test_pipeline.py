#!/usr/bin/env python3
"""
测试 Semiff 流水线
"""

import os
import sys
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """测试所有模块导入"""
    print("🔍 Testing imports...")

    try:
        from semiff.core.io import VideoReader, RobotLogger
        from semiff.perception.mast3r_wrapper import MASt3RWrapper
        from semiff.perception.sam2_wrapper import SAM2Wrapper
        from semiff.calibration.robot_aligner import align_visual_to_robot
        from semiff.calibration.space_trans import RigidTransform
        from semiff.geometry.meshing import Mesher
        from semiff.geometry.decomposition import ColliderBuilder
        from semiff.rendering.dataset_prep import NerfstudioConverter
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_components():
    """测试基础组件"""
    print("\n🔧 Testing basic components...")

    try:
        # 测试 RigidTransform
        from semiff.calibration.space_trans import RigidTransform
        transform = RigidTransform()
        points = np.random.rand(10, 3)
        transformed = transform.transform_points(points)
        assert transformed.shape == points.shape
        print("✅ RigidTransform works")

        # 测试 Mesher (不实际运行，只测试初始化)
        from semiff.geometry.meshing import Mesher
        mesher = Mesher()
        print("✅ Mesher initialized")

        # 测试 ColliderBuilder
        from semiff.geometry.decomposition import ColliderBuilder
        collider = ColliderBuilder()
        print("✅ ColliderBuilder initialized")

        return True
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 Testing Semiff Pipeline")
    print("=" * 50)

    # 测试导入
    if not test_imports():
        return 1

    # 测试基础组件
    if not test_basic_components():
        return 1

    print("\n🎉 All tests passed!")
    print("\n📝 Next steps:")
    print("1. Prepare test data (video, robot logs, URDF)")
    print("2. Configure paths in config/defaults.yaml")
    print("3. Run: PYTHONPATH=src python main.py")

    return 0

if __name__ == "__main__":
    exit(main())



