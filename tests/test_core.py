"""
工业级核心模块测试套件
测试配置系统、损失函数、几何处理等核心组件
"""
import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile
import sys
import os

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semiff.core.config import ConfigManager
from semiff.core.losses import SoftIoULoss, DiceLoss, FocalLoss, get_loss_function
from semiff.core.geometry import (
    compute_adaptive_threshold,
    statistical_outlier_removal,
    bind_geometry_adaptive,
    validate_transforms
)


class TestConfigManager(unittest.TestCase):
    """测试配置管理系统"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"

        # 创建测试配置
        test_config = """
pipeline:
  name: "test_pipeline"
  workspace: "test_workspace"
  resume: false

robot:
  urdf_path: "test.urdf"
  base_frame: "base"
  ee_frame: "ee"

optimization:
  lr_pose: 0.001
  lr_scale: 0.005
  iterations: 100
  loss:
    type: "soft_iou"
    smooth: 1e-6

geometry:
  binding_method: "adaptive_knn"
"""
        with open(self.config_path, 'w') as f:
            f.write(test_config)

    def tearDown(self):
        # 清理临时文件
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_config(self):
        """测试配置加载"""
        config = ConfigManager.load(str(self.config_path))

        self.assertEqual(config.pipeline.name, "test_pipeline")
        self.assertEqual(config.robot.urdf_path, "test.urdf")
        self.assertEqual(config.optimization.lr_pose, 0.001)

    def test_validate_config(self):
        """测试配置验证"""
        config = ConfigManager.load(str(self.config_path))
        self.assertTrue(ConfigManager.validate_config(config))

    def test_invalid_config(self):
        """测试无效配置"""
        # 创建无效配置（缺少必需字段）
        invalid_config = """
pipeline:
  name: "invalid"
"""
        invalid_path = Path(self.temp_dir) / "invalid.yaml"
        with open(invalid_path, 'w') as f:
            f.write(invalid_config)

        config = ConfigManager.load(str(invalid_path))
        self.assertFalse(ConfigManager.validate_config(config))


class TestLossFunctions(unittest.TestCase):
    """测试损失函数"""

    def setUp(self):
        # 创建测试数据
        self.batch_size = 2
        self.height, self.width = 10, 10

        # 生成预测和真实mask
        self.pred_mask = torch.rand(self.batch_size, self.height, self.width, requires_grad=True)
        self.gt_mask = torch.randint(0, 2, (self.batch_size, self.height, self.width)).float()

    def test_soft_iou_loss(self):
        """测试SoftIoULoss"""
        loss_fn = SoftIoULoss()
        loss = loss_fn(self.pred_mask, self.gt_mask)

        # 检查损失值范围
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertLessEqual(loss.item(), 1.0)

        # 检查梯度传播
        loss.backward()
        self.assertIsNotNone(self.pred_mask.grad)
        self.assertFalse(torch.isnan(self.pred_mask.grad).any())

    def test_soft_iou_computation(self):
        """测试IoU计算"""
        loss_fn = SoftIoULoss()

        # 完全匹配的情况
        perfect_pred = self.gt_mask.clone()
        iou = loss_fn.compute_iou(perfect_pred, self.gt_mask)
        self.assertAlmostEqual(iou.item(), 1.0, places=5)

        # 完全不匹配的情况
        inverse_pred = 1 - self.gt_mask
        iou = loss_fn.compute_iou(inverse_pred, self.gt_mask)
        self.assertLess(iou.item(), 0.5)  # 应该远小于0.5

    def test_dice_loss(self):
        """测试DiceLoss"""
        loss_fn = DiceLoss()
        loss = loss_fn(self.pred_mask, self.gt_mask)

        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertLessEqual(loss.item(), 1.0)

    def test_focal_loss(self):
        """测试FocalLoss"""
        loss_fn = FocalLoss()
        loss = loss_fn(self.pred_mask, self.gt_mask)

        self.assertGreaterEqual(loss.item(), 0.0)

    def test_loss_factory(self):
        """测试损失函数工厂"""
        # 测试不同类型的损失函数
        soft_iou = get_loss_function("soft_iou")
        dice = get_loss_function("dice")
        focal = get_loss_function("focal")

        self.assertIsInstance(soft_iou, SoftIoULoss)
        self.assertIsInstance(dice, DiceLoss)
        self.assertIsInstance(focal, FocalLoss)

        # 测试无效类型
        with self.assertRaises(ValueError):
            get_loss_function("invalid_type")

    def test_shape_mismatch(self):
        """测试形状不匹配的情况"""
        loss_fn = SoftIoULoss()
        wrong_shape_pred = torch.rand(1, 5, 5)  # 不同形状

        with self.assertRaises(ValueError):
            loss_fn(wrong_shape_pred, self.gt_mask)


class TestGeometryFunctions(unittest.TestCase):
    """测试几何处理函数"""

    def setUp(self):
        # 创建测试点云
        np.random.seed(42)
        self.test_points = np.random.rand(1000, 3) * 2.0  # 2m x 2m x 2m 空间

        # 创建测试距离数组
        self.test_distances = np.random.exponential(0.05, 500)

    def test_adaptive_threshold_percentile(self):
        """测试百分位数自适应阈值"""
        threshold = compute_adaptive_threshold(self.test_distances, 'percentile', 95)
        self.assertGreater(threshold, 0.0)

        # 95%分位数应该大于中位数
        median = np.median(self.test_distances)
        self.assertGreaterEqual(threshold, median)

    def test_adaptive_threshold_mad(self):
        """测试MAD自适应阈值"""
        threshold = compute_adaptive_threshold(self.test_distances, 'mad', 3.0)
        self.assertGreater(threshold, 0.0)

    def test_empty_distances(self):
        """测试空距离数组"""
        empty_distances = np.array([])
        threshold = compute_adaptive_threshold(empty_distances)
        self.assertEqual(threshold, 0.02)  # 应该返回默认值

    def test_statistical_outlier_removal(self):
        """测试统计异常值移除"""
        # 添加一些明显的异常值
        noisy_points = np.vstack([
            self.test_points,
            np.array([[10, 10, 10], [-5, -5, -5]])  # 明显的异常点
        ])

        filtered = statistical_outlier_removal(noisy_points)

        # 过滤后的点数应该减少
        self.assertLess(len(filtered), len(noisy_points))

        # 正常点应该被保留
        self.assertGreaterEqual(len(filtered), len(self.test_points) - 10)  # 允许小量误删

    def test_transform_validation(self):
        """测试变换矩阵验证"""
        # 有效的变换矩阵
        valid_transform = np.eye(4)
        transforms = {"test": valid_transform}

        self.assertTrue(validate_transforms(transforms))

        # 无效的变换矩阵（非正交旋转）
        invalid_transform = np.eye(4)
        invalid_transform[0, 1] = 0.5  # 破坏正交性
        transforms = {"test": invalid_transform}

        self.assertFalse(validate_transforms(transforms))

    def test_bind_geometry_adaptive(self):
        """测试自适应几何绑定"""
        # 创建简单的模拟mesh
        try:
            import trimesh

            # 创建一个简单的立方体mesh
            box = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
            urdf_meshes = {"link1": box}

            # 点云中包含一些接近mesh的点
            close_points = np.random.rand(50, 3) * 0.1 + np.array([0, 0, 0])  # 围绕原点
            far_points = np.random.rand(50, 3) * 2.0 + np.array([1, 1, 1])    # 远离原点
            test_points = np.vstack([close_points, far_points])

            is_robot, link_indices = bind_geometry_adaptive(
                test_points, urdf_meshes, method='adaptive_knn'
            )

            # 检查输出形状
            self.assertEqual(len(is_robot), len(test_points))
            self.assertEqual(len(link_indices), len(test_points))

            # 检查数据类型
            self.assertTrue(is_robot.dtype == bool or is_robot.dtype == np.bool_)
            self.assertTrue(link_indices.dtype == np.int32)

        except ImportError:
            self.skipTest("trimesh not available")


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_full_pipeline_config_loading(self):
        """测试完整pipeline配置加载"""
        # 创建临时配置
        temp_dir = tempfile.mkdtemp()
        config_path = Path(temp_dir) / "integration_config.yaml"

        config_content = """
pipeline:
  name: "integration_test"
  workspace: "test_workspace"
  resume: true

robot:
  urdf_path: "test.urdf"
  base_frame: "base"
  ee_frame: "ee"

optimization:
  lr_pose: 0.001
  lr_scale: 0.005
  iterations: 50
  loss:
    type: "soft_iou"
    smooth: 1e-6

geometry:
  binding_method: "adaptive_knn"
  outlier_removal_std: 2.0
  adaptive_threshold_percentile: 95

gpu:
  max_memory_gb: 4.0
  enable_memory_monitoring: true

logging:
  level: "INFO"
  file: "test.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
"""

        try:
            with open(config_path, 'w') as f:
                f.write(config_content)

            config = ConfigManager.load(str(config_path))
            self.assertTrue(ConfigManager.validate_config(config))

            # 测试损失函数工厂
            loss_fn = get_loss_function(config.optimization.loss.type)
            self.assertIsInstance(loss_fn, SoftIoULoss)

        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestConfigManager))
    suite.addTests(loader.loadTestsFromTestCase(TestLossFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestGeometryFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 返回测试结果
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Running SEMIFF Core Tests...")

    success = run_tests()

    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
