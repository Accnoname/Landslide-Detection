import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss đo lường độ trùng lặp giữa dự đoán và nhãn thật.
    Rất hiệu quả cho bài toán mất cân bằng lớp cực nặng (như sạt lở đất).
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice

class ComboLoss(nn.Module):
    """
    Kết hợp giữa Dice Loss và BCEWithLogitsLoss.
    - BCE giúp làm phẳng bề mặt đạo hàm ở giai đoạn đầu.
    - Dice Loss ép mô hình tối ưu trực tiếp cho F1-Score, kéo Recall lên cao.
    """
    def __init__(self, dice_weight=0.6, bce_weight=0.4):
        super(ComboLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        loss_dice = self.dice(logits, targets)
        loss_bce = self.bce(logits, targets)
        return self.dice_weight * loss_dice + self.bce_weight * loss_bce
