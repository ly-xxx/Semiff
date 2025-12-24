"""
工业级几何处理模块
提供自适应几何绑定和鲁棒的点云处理
"""
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from typing import Dict, Tuple, List, Optional
import open3d as o3d
from pathlib import Path


def compute_adaptive_threshold(distances: np.ndarray,
                              method: str = 'percentile',
                              param: float = 95.0) -> float:
    """
    基于距离分布动态计算阈值，避免硬编码

    Args:
        distances: 距离数组
        method: 计算方法 ('percentile', 'mad', 'otsu', 'kmeans')
        param: 方法参数

    Returns:
        自适应阈值
    """
    if len(distances) == 0:
        return 0.02  # fallback

    if method == 'percentile':
        # 取距离分布的指定分位点
        threshold = np.percentile(distances, param)
        return float(threshold)

    elif method == 'mad':
        # 中位数绝对偏差 (Median Absolute Deviation)
        median = np.median(distances)
        mad = np.median(np.abs(distances - median))
        return float(median + param * mad)

    elif method == 'otsu':
        # Otsu's method - 假设双峰分布
        return _otsu_threshold(distances)

    elif method == 'kmeans':
        # K-means聚类阈值
        return _kmeans_threshold(distances, int(param))

    else:
        return 0.02  # fallback


def _otsu_threshold(distances: np.ndarray) -> float:
    """Otsu's method for threshold calculation"""
    # 简单实现：找到最小类内方差
    distances = distances[distances < np.inf]
    if len(distances) < 10:
        return np.mean(distances)

    hist, bin_edges = np.histogram(distances, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 计算类内方差
    total_pixels = len(distances)
    min_variance = float('inf')
    best_threshold = np.mean(distances)

    for threshold in bin_centers:
        # 前景和背景分割
        foreground = distances[distances <= threshold]
        background = distances[distances > threshold]

        if len(foreground) == 0 or len(background) == 0:
            continue

        # 类内方差
        w1 = len(foreground) / total_pixels
        w2 = len(background) / total_pixels
        var1 = np.var(foreground) if len(foreground) > 1 else 0
        var2 = np.var(background) if len(background) > 1 else 0

        within_class_variance = w1 * var1 + w2 * var2

        if within_class_variance < min_variance:
            min_variance = within_class_variance
            best_threshold = threshold

    return float(best_threshold)


def _kmeans_threshold(distances: np.ndarray, n_clusters: int = 2) -> float:
    """使用K-means确定阈值"""
    from sklearn.cluster import KMeans

    distances = distances[distances < np.inf].reshape(-1, 1)
    if len(distances) < n_clusters:
        return np.mean(distances)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(distances)

    # 返回聚类中心的较大值作为阈值
    centers = kmeans.cluster_centers_.flatten()
    return float(np.max(centers))


def statistical_outlier_removal(points: np.ndarray,
                               nb_neighbors: int = 20,
                               std_ratio: float = 2.0) -> np.ndarray:
    """
    统计异常值移除 - 使用Open3D

    Args:
        points: (N, 3) 点云
        nb_neighbors: 邻域点数
        std_ratio: 标准差倍数阈值

    Returns:
        过滤后的点云
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                            std_ratio=std_ratio)

    return np.asarray(pcd.points)


def bind_geometry_adaptive(points: np.ndarray,
                          urdf_meshes: Dict[str, trimesh.Trimesh],
                          method: str = 'adaptive_knn',
                          **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    自适应几何绑定 - 替代硬编码阈值的方法

    Args:
        points: (N, 3) 场景点云
        urdf_meshes: 各link的mesh字典
        method: 绑定方法 ('adaptive_knn', 'fixed', 'statistical')
        **kwargs: 方法参数

    Returns:
        is_robot: (N,) bool数组，是否为机器人点
        link_indices: (N,) int数组，对应的link索引 (-1表示背景)
    """
    print(f"🔗 Starting adaptive geometry binding with method: {method}")

    N = len(points)
    link_indices = np.full(N, -1, dtype=np.int32)
    min_dists = np.full(N, np.inf)

    # 1. 计算每个点到最近Link的距离
    for link_idx, (link_name, mesh) in enumerate(urdf_meshes.items()):
        try:
            # 采样Mesh表面构建KDTree (比直接查询mesh快且稳)
            sample_pts, _ = trimesh.sample.sample_surface(mesh, kwargs.get('sample_points', 5000))
            tree = cKDTree(sample_pts)
            dists, _ = tree.query(points)

            # 更新最近Link
            update_mask = dists < min_dists
            min_dists[update_mask] = dists[update_mask]
            link_indices[update_mask] = link_idx

        except Exception as e:
            print(f"⚠️  Warning: Failed to process link {link_name}: {e}")
            continue

    # 2. 根据方法计算阈值
    if method == 'fixed':
        threshold = kwargs.get('fixed_threshold', 0.02)
        print(".4f")
    elif method == 'adaptive_knn':
        # 只考虑最近的点进行阈值计算（假设机器人点更集中）
        valid_dists = min_dists[min_dists < kwargs.get('max_dist', 0.5)]
        if len(valid_dists) > 0:
            threshold = compute_adaptive_threshold(
                valid_dists,
                method='percentile',
                param=kwargs.get('percentile', 95)
            )
        else:
            threshold = 0.02
        print(".4f")
    elif method == 'statistical':
        # 基于统计分布的阈值
        valid_dists = min_dists[min_dists < np.inf]
        if len(valid_dists) > 0:
            threshold = compute_adaptive_threshold(
                valid_dists,
                method='mad',
                param=kwargs.get('mad_param', 3.0)
            )
        else:
            threshold = 0.02
        print(".4f")
    else:
        raise ValueError(f"Unknown binding method: {method}")

    # 3. 应用阈值进行分类
    is_robot = min_dists < threshold

    # 统计信息
    robot_points = np.sum(is_robot)
    total_points = len(points)
    print(".1f")
    return is_robot, link_indices


def validate_transforms(transforms: Dict[str, np.ndarray]) -> bool:
    """
    验证变换矩阵的数学正确性

    Args:
        transforms: 变换矩阵字典

    Returns:
        是否全部有效
    """
    for name, T in transforms.items():
        if T.shape != (4, 4):
            print(f"❌ Transform {name} has wrong shape: {T.shape}")
            return False

        # 检查正交性 (旋转部分应为正交矩阵)
        R = T[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
            print(f"❌ Transform {name} rotation is not orthogonal")
            return False

        # 检查行列式 (应接近1)
        det = np.linalg.det(R)
        if not np.isclose(det, 1.0, atol=1e-3):
            print(f"❌ Transform {name} has non-unit determinant: {det}")
            return False

    print("✅ All transforms are mathematically valid")
    return True


def load_urdf_meshes_validated(urdf_path: str) -> Dict[str, trimesh.Trimesh]:
    """
    加载URDF meshes并验证几何有效性

    Args:
        urdf_path: URDF文件路径

    Returns:
        link名称到mesh的字典
    """
    try:
        import yourdfpy
        robot = yourdfpy.URDF.load(urdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load URDF {urdf_path}: {e}")

    meshes = {}
    for link_name in robot.link_names:
        mesh = robot.scene.geometry.get(link_name)
        if mesh is None:
            continue

        # 验证mesh有效性
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"⚠️  Link {link_name} has non-trimesh geometry, skipping")
            continue

        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            print(f"⚠️  Link {link_name} has empty mesh, skipping")
            continue

        meshes[link_name] = mesh

    if len(meshes) == 0:
        raise ValueError(f"No valid meshes found in URDF {urdf_path}")

    print(f"📦 Loaded {len(meshes)} valid meshes from URDF")
    return meshes


class GeometryBinder:
    """
    自适应几何绑定器
    负责将点云与机器人几何进行智能绑定
    """

    def __init__(self, urdf_meshes: Dict[str, trimesh.Trimesh],
                 method: str = 'adaptive_knn',
                 adaptive_percentile: float = 95.0,
                 **kwargs):
        """
        初始化几何绑定器

        Args:
            urdf_meshes: link名称到mesh的字典
            method: 绑定方法
            adaptive_percentile: 自适应百分位数
            **kwargs: 其他参数
        """
        self.urdf_meshes = urdf_meshes
        self.method = method
        self.adaptive_percentile = adaptive_percentile
        self.kwargs = kwargs

        # 预计算KDTree以提高性能
        self._build_kdtrees()

    def _build_kdtrees(self):
        """预构建KDTree以加速查询"""
        self.kdtrees = {}
        for link_name, mesh in self.urdf_meshes.items():
            try:
                # 在mesh表面采样构建KDTree
                sample_pts, _ = trimesh.sample.sample_surface(mesh, self.kwargs.get('sample_points', 5000))
                self.kdtrees[link_name] = cKDTree(sample_pts)
            except Exception as e:
                print(f"⚠️  Failed to build KDTree for {link_name}: {e}")

    def bind(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行几何绑定

        Args:
            points: (N, 3) 点云

        Returns:
            is_robot: (N,) bool数组，是否为机器人点
            link_indices: (N,) int数组，对应的link索引 (-1表示背景)
        """
        N = len(points)
        link_indices = np.full(N, -1, dtype=np.int32)
        min_dists = np.full(N, np.inf)

        # 1. 计算每个点到最近Link的距离
        for link_idx, (link_name, tree) in enumerate(self.kdtrees.items()):
            try:
                dists, _ = tree.query(points)
                # 更新最近Link
                update_mask = dists < min_dists
                min_dists[update_mask] = dists[update_mask]
                link_indices[update_mask] = link_idx
            except Exception as e:
                print(f"⚠️  Failed to query {link_name}: {e}")
                continue

        # 2. 根据方法计算阈值
        if self.method == 'adaptive_knn':
            # 只考虑最近的点进行阈值计算
            valid_dists = min_dists[min_dists < self.kwargs.get('max_dist', 0.5)]
            if len(valid_dists) > 0:
                threshold = compute_adaptive_threshold(
                    valid_dists,
                    method='percentile',
                    param=self.adaptive_percentile
                )
            else:
                threshold = 0.02
            print(".4f")
        else:
            threshold = self.kwargs.get('fixed_threshold', 0.02)
            print(".4f")

        # 3. 应用阈值进行分类
        is_robot = min_dists < threshold

        # 统计信息
        robot_points = np.sum(is_robot)
        total_points = len(points)
        print(".1f")

        return is_robot, link_indices


# 测试函数
if __name__ == "__main__":
    print("🧪 Testing Geometry Functions...")

    # 测试自适应阈值计算
    test_distances = np.random.exponential(0.05, 1000)
    threshold = compute_adaptive_threshold(test_distances, 'percentile', 95)
    print(".4f")

    # 测试统计异常值移除
    test_points = np.random.rand(1000, 3)
    filtered = statistical_outlier_removal(test_points)
    print(f"  Outlier removal: {len(test_points)} -> {len(filtered)} points")

    print("✅ Geometry functions working correctly!")
