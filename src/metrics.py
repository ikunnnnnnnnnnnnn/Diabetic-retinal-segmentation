import torch
import numpy as np
from sklearn.metrics import auc, precision_recall_curve
import matplotlib.pyplot as plt  # 导入 matplotlib

# 定义类别名称
CLASS_NAMES = {
    1: 'EX',  # 硬性渗出液
    2: 'HE',  # 出血斑
    3: 'MA',  # 微动脉瘤
    4: 'SE'   # 软性渗出液
}
def calculate_tp_fp_fn(predictions, targets, threshold=0.5):
    """
    计算 TP, FP, FN (这里假设是二分类，多分类需要按类别计算).
    对于多分类任务，通常在 calculate_precision_recall 和 calculate_aupr 中处理。
    """
    pass  # 多分类通常不需要在这里显式计算全局 TP, FP, FN


def calculate_precision_recall(predictions, targets, num_classes=None):
    """
    计算全局的精确率和召回率 (基于像素).

    Args:
        predictions (torch.Tensor): 模型预测的概率图，形状为 (batch_size, num_classes, height, width).
        targets (torch.Tensor): 真实的标签，形状为 (batch_size, height, width), 类别索引.
        num_classes (int, 可选): 类别的数量. 如果为 None，则自动从 predictions 中推断.

    Returns:
        tuple: (precision, recall) - 全局精确率和召回率.
    """
    if num_classes is None:
        num_classes = predictions.shape[1]

    all_preds = predictions.argmax(dim=1).flatten()
    all_targets = targets.flatten()

    tp = torch.sum((all_preds == all_targets) & (all_targets != 0)).float()  # 排除背景
    total_predicted_positives = torch.sum(all_preds != 0).float()
    total_actual_positives = torch.sum(all_targets != 0).float()

    precision = tp / (total_predicted_positives + 1e-7)
    recall = tp / (total_actual_positives + 1e-7)

    return precision.item(), recall.item()


def calculate_pr_curve(predictions, targets, num_classes=None):
    """
    计算多类别的 Precision-Recall 曲线，并返回结果和图像.

    Args:
        predictions (torch.Tensor): 模型预测的概率图，形状为 (batch_size, num_classes, height, width).
        targets (torch.Tensor): 真实的标签，形状为 (batch_size, height, width), 类别索引.
        num_classes (int, 可选): 类别的数量 (不包括背景). 如果为 None，则自动从 predictions 中推断.

    Returns:
        tuple: (各类别 PR 曲线数据, matplotlib Figure 对象)
    """
    if num_classes is None:
        num_classes = predictions.shape[1]

    pr_data = {}
    plt.figure(figsize=(8, 6))

    for class_id in range(1, num_classes):  # 从类别 1 开始，跳过背景类别 0
        y_true = (targets == class_id).float().cpu().numpy().ravel()
        y_pred = predictions[:, class_id, :, :].float().cpu().numpy().ravel()

        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_data[f'Class_{class_id}'] = (precision, recall)
        aupr = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f'{CLASS_NAMES[class_id]} (AUPR = {aupr:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()

    pr_curve_fig = plt.gcf()  # 获取当前的 Figure 对象
    plt.close()  # 关闭 Figure，防止在控制台中显示

    return pr_data, pr_curve_fig


def calculate_aupr(predictions, targets, num_classes=None):
    """
    计算多类别的 AUPR.

    Args:
        predictions (torch.Tensor): 模型预测的概率图，形状为 (batch_size, num_classes, height, width).
        targets (torch.Tensor): 真实的标签，形状为 (batch_size, height, width), 类别索引.
        num_classes (int, 可选): 类别的数量. 如果为 None，则自动从 predictions 中推断.

    Returns:
        dict: 包含每个类别 AUPR 值的字典.
    """
    if num_classes is None:
        num_classes = predictions.shape[1]

    aupr_scores = {}
    for class_id in range(1, num_classes):  # 从类别 1 开始，跳过背景类别 0
        y_true = (targets == class_id).float().cpu().numpy().ravel()
        y_pred = predictions[:, class_id, :, :].float().cpu().numpy().ravel()

        if np.sum(y_true) > 0:
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            aupr = auc(recall, precision)
            aupr_scores[f'Class_{class_id}'] = aupr
        else:
            aupr_scores[f'Class_{class_id}'] = 0.0  # 或 float('nan')

    return aupr_scores


if __name__ == '__main__':
    # 示例用法
    batch_size = 2
    num_classes = 5
    height = 64
    width = 64

    predictions = torch.randn(batch_size, num_classes, height, width).softmax(dim=1)
    targets = torch.randint(0, num_classes, (batch_size, height, width))

    precision, recall = calculate_precision_recall(predictions, targets, num_classes)
    print(f"Global Precision: {precision:.4f}, Recall: {recall:.4f}")

    aupr_scores = calculate_aupr(predictions, targets, num_classes)
    print("AUPR Scores:", aupr_scores)

    pr_data, pr_curve_fig = calculate_pr_curve(predictions, targets, num_classes)
    pr_curve_fig.savefig('pr_curve_example_rewritten.png')
    print("P-R Curve plotted and saved to 'pr_curve_example_rewritten.png'")