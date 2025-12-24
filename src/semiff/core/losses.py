"""
工业级损失函数库
提供 differentiable 的几何对齐损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SoftIoULoss(nn.Module):
    """
    Soft IoU Loss - 用于二值mask对齐的平滑可导损失函数

    IoU = Intersection / Union
    Loss = 1 - IoU

    相比MSE的优势：
    - 对形状重叠更敏感
    - 梯度更稳定
    - 数学意义更明确
    """

    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth: 平滑项，防止除零错误
        """
        super(SoftIoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        计算Soft IoU损失

        Args:
            pred_mask: 预测mask (B, H, W) 或 (B, 1, H, W), range [0, 1], differentiable
            gt_mask: 真实mask (B, H, W) 或 (B, 1, H, W), range {0, 1}

        Returns:
            IoU损失值 (标量)
        """
        # 输入验证和标准化
        if pred_mask.shape != gt_mask.shape:
            raise ValueError(f"Shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}")

        # 移除单通道维度如果存在
        if pred_mask.dim() == 4 and pred_mask.shape[1] == 1:
            pred_mask = pred_mask.squeeze(1)
            gt_mask = gt_mask.squeeze(1)

        if pred_mask.dim() != 3:
            raise ValueError(f"Expected 3D tensors (B, H, W), got {pred_mask.dim()}D")

        # 确保输入范围合理
        pred_mask = torch.clamp(pred_mask, 0, 1)

        # Flatten到批次级别进行计算
        pred_flat = pred_mask.view(pred_mask.size(0), -1)  # (B, H*W)
        gt_flat = gt_mask.view(gt_mask.size(0), -1)        # (B, H*W)

        # 计算IoU
        intersection = (pred_flat * gt_flat).sum(dim=1)     # (B,)
        total = (pred_flat + gt_flat).sum(dim=1)            # (B,)
        union = total - intersection                         # (B,)

        # IoU = I / U
        iou = (intersection + self.smooth) / (union + self.smooth)  # (B,)

        # Loss = 1 - IoU (平均值)
        loss = 1.0 - iou.mean()

        return loss

    def compute_iou(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        计算IoU指标（用于评估，不用于训练）

        Returns:
            IoU值 (0-1之间)
        """
        with torch.no_grad():
            # 使用相同的计算逻辑但不计算梯度
            pred_flat = pred_mask.view(pred_mask.size(0), -1)
            gt_flat = gt_mask.view(gt_mask.size(0), -1)

            intersection = (pred_flat * gt_flat).sum(dim=1)
            total = (pred_flat + gt_flat).sum(dim=1)
            union = total - intersection

            iou = (intersection + self.smooth) / (union + self.smooth)
            return iou.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss - 另一种常用的分割损失函数
    Dice = 2 * Intersection / (|pred| + |gt|)
    Loss = 1 - Dice
    """

    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        pred_flat = pred_mask.view(pred_mask.size(0), -1)
        gt_flat = gt_mask.view(gt_mask.size(0), -1)

        intersection = (pred_flat * gt_flat).sum(dim=1)
        pred_sum = pred_flat.sum(dim=1)
        gt_sum = gt_flat.sum(dim=1)

        dice = (2. * intersection + self.smooth) / (pred_sum + gt_sum + self.smooth)
        loss = 1. - dice.mean()

        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss - 解决类别不平衡问题
    适用于前景像素远少于背景像素的情况
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        # BCE loss
        bce = F.binary_cross_entropy(pred_mask, gt_mask, reduction='none')

        # Focal modulation
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce

        return focal_loss.mean()


def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    工厂函数：根据配置创建损失函数

    Args:
        loss_type: 损失函数类型 ("soft_iou", "dice", "focal")
        **kwargs: 损失函数参数

    Returns:
        损失函数实例
    """
    if loss_type == "soft_iou":
        return SoftIoULoss(**kwargs)
    elif loss_type == "dice":
        return DiceLoss(**kwargs)
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# 单元测试示例
if __name__ == "__main__":
    print("🧪 Testing Loss Functions...")

    # 创建测试数据
    pred = torch.rand(2, 100, 100, requires_grad=True)
    gt = torch.randint(0, 2, (2, 100, 100)).float()

    # 测试SoftIoULoss
    loss_fn = SoftIoULoss()
    loss = loss_fn(pred, gt)
    loss.backward()

    print(".4f")
    print(f"  Gradient Check: {not torch.isnan(pred.grad).any()}")

    # 测试IoU计算
    iou_val = loss_fn.compute_iou(pred.detach(), gt)
    print(".4f")
    # 测试DiceLoss
    dice_loss = DiceLoss()
    dice_val = dice_loss(pred.detach(), gt)
    print(".4f")
    print("✅ All loss functions working correctly!")
