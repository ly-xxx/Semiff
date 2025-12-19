# Pipeline Changes: Unified Real2Sim Toolchain

This document summarizes the changes made to unify the semiff pipeline with real2sim-eval's robust Sapien-based approach.

## Key Changes

### 1. Toolchain Unification
- **Deprecated**: `src/semiff/calibration/robot_aligner.py` (yourdfpy-based, limited functionality)
- **Adopted**: Sapien-based robot processing from real2sim-eval
- **Added**: `src/semiff/utils/robot/robot_pc_sampler.py` - Sapien-powered robot point cloud generation
- **Added**: `src/semiff/utils/gs/` - Gaussian Splatting processing utilities

### 2. Pipeline Updates

#### New Step 2: 3DGS Training
- **New Script**: `tools/step2_train_splat.py`
- **Purpose**: Train high-quality Gaussian Splatting models from MASt3R reconstruction
- **Methods**: Nerfstudio (recommended) or gsplat
- **Output**: `outputs/splat/scene.ply` with SH colors

#### Updated Step 3: Asset Building & Alignment
- **Rewritten**: `tools/step3_build_assets.py`
- **New Features**:
  - Sapien-based robot point cloud sampling
  - Automatic RANSAC + ICP alignment
  - Direct Gaussian Splatting transformation
  - Unified coordinate system (URDF base frame)

### 3. Dependencies Added
```toml
"sapien>=3.0.0",        # Physics simulation engine
"kornia>=0.7.0",        # Computer vision library
"nerfstudio>=1.0.0",    # 3DGS training (optional)
```

## Benefits of New Approach

1. **Coordinate System Consistency**: Sapien ensures perfect alignment between simulation and real-world URDF models
2. **Robust Alignment**: RANSAC + ICP provides reliable Sim2Real transformation
3. **Plug-and-Play Assets**: Transformed scenes work directly with Warp physics
4. **Future-Proof**: Unified toolchain for all robot manipulation tasks

## Migration Notes

- Old `robot_aligner.py` is deprecated but kept for reference
- New pipeline requires Step 2 (3DGS training) before Step 3
- Install new dependencies: `pip install sapien kornia`
- For 3DGS training: `pip install nerfstudio` or use gsplat

## Usage

```bash
# Step 1: Reconstruction (unchanged)
python tools/step1_reconstruct.py

# Step 2: Train 3DGS (NEW)
python tools/step2_train_splat.py --method nerfstudio

# Step 3: Build aligned assets (UPDATED)
python tools/step3_build_assets.py
```

The new pipeline provides a solid foundation for Sim2Real robot manipulation with guaranteed coordinate system consistency.