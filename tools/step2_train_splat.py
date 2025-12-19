"""
Step 2: Train 3D Gaussian Splatting Model
This step trains a high-quality 3DGS model from the reconstructed data.

Dependencies:
- pip install nerfstudio  # or gsplat
- pip install torch torchvision

Usage:
python tools/step2_train_splat.py
"""

import sys
from pathlib import Path
import subprocess
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train 3DGS model for scene reconstruction")
    parser.add_argument(
        "--method",
        choices=["nerfstudio", "gsplat"],
        default="nerfstudio",
        help="Training method to use"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="outputs/mast3r_result",
        help="Input data directory from Step 1"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/splat",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30000,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1,
        help="Resolution scaling factor (1 = full resolution)"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 [Step 2] Training 3D Gaussian Splatting Model...")

    # Check input data
    required_files = ["poses.npy", "images"]
    for file in required_files:
        if not (data_dir / file).exists():
            if file == "images":
                if not (data_dir / file).is_dir():
                    print(f"❌ Required input not found: {data_dir / file}")
                    print("Run Step 1 (reconstruction) first.")
                    return
            else:
                print(f"❌ Required input not found: {data_dir / file}")
                print("Run Step 1 (reconstruction) first.")
                return

    print(f"Using method: {args.method}")
    print(f"Input data: {data_dir}")
    print(f"Output dir: {output_dir}")

    if args.method == "nerfstudio":
        # Use Nerfstudio for training
        success = train_with_nerfstudio(data_dir, output_dir, args.iterations, args.resolution)
    else:
        # Use gsplat (alternative implementation)
        success = train_with_gsplat(data_dir, output_dir, args.iterations)

    if success:
        print("✅ [Step 2] Training completed successfully!")
        print(f"📁 Model saved to: {output_dir}")

        # Check for expected output file
        expected_ply = output_dir / "scene.ply"
        if expected_ply.exists():
            print(f"🎯 Found trained model: {expected_ply}")
        else:
            print("⚠️ Warning: Expected output file 'scene.ply' not found.")
            print("Check the training logs for the correct output location.")
    else:
        print("❌ [Step 2] Training failed. Check logs above.")


def train_with_nerfstudio(data_dir: Path, output_dir: Path, iterations: int, resolution: int) -> bool:
    """
    Train using Nerfstudio's Gaussian Splatting implementation

    Args:
        data_dir: Input data directory
        output_dir: Output directory
        iterations: Number of training iterations
        resolution: Resolution scaling

    Returns:
        True if successful
    """
    try:
        import nerfstudio
    except ImportError:
        print("❌ Nerfstudio not installed. Install with: pip install nerfstudio")
        print("Or use --method gsplat instead")
        return False

    # Prepare Nerfstudio data format
    ns_data_dir = output_dir / "ns_data"
    ns_data_dir.mkdir(exist_ok=True)

    # Convert poses.npy to transforms.json format expected by Nerfstudio
    try:
        convert_mast3r_to_nerfstudio(data_dir, ns_data_dir)
    except Exception as e:
        print(f"❌ Failed to convert data format: {e}")
        return False

    # Run training
    cmd = [
        "ns-train",
        "gaussian-splatting",
        "--data",
        str(ns_data_dir),
        "--output-dir",
        str(output_dir / "ns_outputs"),
        "--max-num-iterations",
        str(iterations),
        "--downscale-factor",
        str(resolution),
        "--save-iterations",
        str(iterations),  # Save final model
    ]

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=output_dir)

    if result.returncode == 0:
        # Try to find and copy the trained PLY file
        ns_output_dir = output_dir / "ns_outputs"
        ply_files = list(ns_output_dir.glob("**/*.ply"))
        if ply_files:
            # Copy the latest PLY file
            ply_files.sort(key=lambda x: x.stat().st_mtime)
            final_ply = ply_files[-1]
            shutil.copy2(final_ply, output_dir / "scene.ply")
            print(f"Copied trained model to: {output_dir / 'scene.ply'}")

        return True
    else:
        print(f"Training failed with return code: {result.returncode}")
        return False


def convert_mast3r_to_nerfstudio(mast3r_dir: Path, ns_dir: Path):
    """
    Convert MASt3R output format to Nerfstudio format

    Args:
        mast3r_dir: MASt3R output directory
        ns_dir: Nerfstudio data directory
    """
    import json
    import numpy as np
    from PIL import Image

    poses = np.load(mast3r_dir / "poses.npy")  # [N, 4, 4] world-to-camera
    images_dir = mast3r_dir / "images"
    image_files = sorted(list(images_dir.glob("*.png")))

    if len(image_files) != len(poses):
        raise ValueError(f"Mismatch: {len(image_files)} images but {len(poses)} poses")

    # Get image dimensions
    sample_img = Image.open(image_files[0])
    h, w = sample_img.size[::-1]  # PIL returns (w, h)

    # Create camera intrinsics (assume same for all)
    # MASt3R typically uses normalized coordinates, need to estimate focal length
    focal_length = w * 0.8  # Rough estimate
    cx, cy = w / 2, h / 2

    # Create transforms.json
    transforms = {
        "camera_model": "OPENCV",
        "fl_x": focal_length,
        "fl_y": focal_length,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "frames": []
    }

    for i, (img_file, pose) in enumerate(zip(image_files, poses)):
        # Convert world-to-camera to camera-to-world
        c2w = np.linalg.inv(pose)

        frame = {
            "file_path": f"images/{img_file.name}",
            "transform_matrix": c2w.tolist()
        }
        transforms["frames"].append(frame)

        # Copy image to ns_data/images/
        ns_images_dir = ns_dir / "images"
        ns_images_dir.mkdir(exist_ok=True)
        shutil.copy2(img_file, ns_images_dir / img_file.name)

    # Save transforms.json
    with open(ns_dir / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=2)


def train_with_gsplat(data_dir: Path, output_dir: Path, iterations: int) -> bool:
    """
    Train using gsplat library (alternative to Nerfstudio)

    Args:
        data_dir: Input data directory
        output_dir: Output directory
        iterations: Number of training iterations

    Returns:
        True if successful
    """
    try:
        import gsplat
    except ImportError:
        print("❌ gsplat not installed. Install with: pip install gsplat")
        print("Or use --method nerfstudio instead")
        return False

    # gsplat training implementation would go here
    # For now, just show a placeholder
    print("⚠️ gsplat training not yet implemented.")
    print("Please use --method nerfstudio for now.")
    return False


if __name__ == "__main__":
    main()