import torch

def calculate_metrics(logits, targets, threshold=0.5):
    """
    Tính toán 4 chỉ số cốt lõi cho bài toán phát hiện sạt lở:
      - Precision: Đoán đúng bao nhiêu % trong số các pixel báo sạt lở
      - Recall: Tìm được bao nhiêu % trong số tất cả vết sạt lở ngoài thực tế
      - F1-Score: Trung bình điều hòa của Precision và Recall
      - IoU (Intersection over Union): Tỷ lệ diện tích giao trên hợp
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    targets = targets.float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou
    }
