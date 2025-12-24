import torch
import torch.nn.functional as F
import numpy as np
import trimesh
import yourdfpy

# 尝试导入 nvdiffrast，如果不存在则允许代码在 "Mock" 模式下运行
try:
    import nvdiffrast.torch as dr
    HAS_NVD = True
except ImportError:
    HAS_NVD = False
    print("⚠️ nvdiffrast not found. Differentiable rendering will be mocked.")

# ==================== Math Utils ====================

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Zhou et al., CVPR 2019: Continuous 6D Rotation -> 3x3 Matrix"""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)

def transform_points(points: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Apply 4x4 Transform to (N, 3) points"""
    ones = torch.ones((*points.shape[:-1], 1), device=points.device)
    points_homo = torch.cat([points, ones], dim=-1)
    return torch.matmul(points_homo, T.transpose(1, 2))[..., :3]

def build_projection_matrix(K: torch.Tensor, H: int, W: int, near=0.1, far=100.0) -> torch.Tensor:
    """OpenCV Intrinsics -> OpenGL Projection Matrix (NDC)"""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    proj = torch.zeros((4, 4), device=K.device)
    proj[0, 0] = 2 * fx / W
    proj[1, 1] = -2 * fy / H  # Flip Y for OpenGL NDC
    proj[0, 2] = (W - 2 * cx) / W
    proj[1, 2] = (H - 2 * cy) / H
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -(2 * far * near) / (far - near)
    proj[3, 2] = -1.0
    return proj

# ==================== Rendering Utils ====================

class MeshRasterizer:
    def __init__(self, device='cuda'):
        self.device = device
        if HAS_NVD:
            self.ctx = dr.RasterizeCGLContext(device=device)
        else:
            self.ctx = None

    def render_mask(self, vertices, faces, K, H, W):
        if not HAS_NVD:
            # Mock return for testing without GPU/Nvdiffrast
            return torch.zeros((vertices.shape[0], H, W), device=self.device)

        B = vertices.shape[0]
        # 1. Project
        proj = build_projection_matrix(K, H, W).unsqueeze(0).repeat(B, 1, 1)
        ones = torch.ones((*vertices.shape[:2], 1), device=self.device)
        v_homo = torch.cat([vertices, ones], dim=-1)
        v_clip = torch.bmm(v_homo, proj.transpose(1, 2))

        # 2. Rasterize
        rast, _ = dr.rasterize(self.ctx, v_clip, faces, resolution=[H, W])

        # 3. Antialias Mask
        v_colors = torch.ones_like(vertices)
        mask, _ = dr.interpolate(v_colors, rast, faces)
        mask = dr.antialias(mask, rast, v_clip, faces)
        return mask[..., 0]

def load_urdf_meshes_gpu(urdf_path, device='cuda'):
    """加载 URDF 中的 mesh 到 GPU，用于渲染"""
    robot = yourdfpy.URDF.load(urdf_path)
    mesh_map = {}

    # 建立link名称到几何文件的映射
    link_to_geom = {}
    for link_name, link in robot.link_map.items():
        if link.visuals:
            for visual in link.visuals:
                if visual.geometry and visual.geometry.mesh and visual.geometry.mesh.filename:
                    # 从路径中提取文件名
                    geom_name = visual.geometry.mesh.filename.split('/')[-1].split('\\')[-1]
                    link_to_geom[link_name] = geom_name

    # 加载几何体
    for link_name, geom_name in link_to_geom.items():
        mesh = robot.scene.geometry.get(geom_name)
        if mesh:
            v = torch.from_numpy(mesh.vertices).float().to(device)
            f = torch.from_numpy(mesh.faces).int().to(device)
            mesh_map[link_name] = {'v': v, 'f': f}

    return robot, mesh_map
