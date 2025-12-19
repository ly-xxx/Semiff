"""
ICP utilities for point cloud alignment
Adapted from real2sim-eval project
"""

import numpy as np
import open3d as o3d
from typing import Tuple


def preprocess_for_features(pcd: o3d.geometry.PointCloud, voxel_size: float) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """
    Preprocess point cloud for feature extraction

    Args:
        pcd: Input point cloud
        voxel_size: Voxel size for downsampling

    Returns:
        Tuple of (downsampled_pcd, fpfh_features)
    """
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Estimate normals
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )

    # Compute FPFH features
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )

    return pcd_down, fpfh


def global_registration_ransac(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
    voxel_size: float
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Perform global registration using RANSAC

    Args:
        source_down: Downsampled source point cloud
        target_down: Downsampled target point cloud
        source_fpfh: FPFH features for source
        target_fpfh: FPFH features for target
        voxel_size: Voxel size used for preprocessing

    Returns:
        Registration result
    """
    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    return result


def refine_with_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial_transform: np.ndarray,
    voxel_size: float
) -> Tuple[o3d.pipelines.registration.RegistrationResult, o3d.pipelines.registration.RegistrationResult]:
    """
    Refine registration with ICP

    Args:
        source: Source point cloud
        target: Target point cloud
        initial_transform: Initial transformation from RANSAC
        voxel_size: Voxel size

    Returns:
        Tuple of (coarse_icp_result, fine_icp_result)
    """
    distance_threshold_coarse = voxel_size * 2.0
    distance_threshold_fine = voxel_size * 0.5

    # Coarse ICP
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold_coarse, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )

    # Fine ICP
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold_fine, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
    )

    return icp_coarse, icp_fine