import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import ResUNet  # 确保你的模型定义在这里
from data import DRDataset  # 确保你的数据集定义在这里
import tqdm
import os
from metrics import calculate_precision_recall, calculate_aupr, calculate_pr_curve # 导入评估指标函数
import matplotlib.pyplot as plt

# --- Hyperparameters ---
BATCH_SIZE = 4  # 可以根据你的硬件调整
IMAGE_SIZE = (512, 512)
NUM_WORKERS = 4
CHECKPOINT_PATH = 'checkpoints/best_model.pth'  # 加载训练好的模型路径
OUTPUT_DIR = 'test_output'  # 保存测试结果的目录

# --- CUDA Setup ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

def create_test_dataset(images_dir, labels_dir, image_size=(512, 512)):
    """创建测试数据集."""
    test_dataset = DRDataset(images_dir=images_dir, labels_dir=labels_dir, target_size=image_size)
    return test_dataset

def create_test_loader(test_dataset, batch_size=4, num_workers=4):
    """创建测试数据加载器."""
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不需要打乱
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return test_loader

def test_model(model, test_loader, device, output_dir):
    """测试模型并评估性能."""
    model.eval()  # 设置模型为评估模式
    criterion = nn.CrossEntropyLoss()  # 与训练时相同的损失函数
    test_loss = 0.0
    all_predictions = []
    all_targets_one_hot = [] # 保存 one-hot 编码的标签

    with torch.no_grad():
        loop = tqdm.tqdm(test_loader, leave=True)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)

            # 转换标签形状以计算损失
            loss = criterion(outputs, torch.argmax(labels, dim=1))
            test_loss += loss.item()

            predictions = outputs.softmax(dim=1).cpu() # 获取概率
            all_predictions.append(predictions)
            all_targets_one_hot.append(labels.cpu()) # 保存 one-hot 编码的标签

            loop.set_postfix(loss=loss.item())

    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets_one_hot = torch.cat(all_targets_one_hot, dim=0)
    all_targets_index = torch.argmax(all_targets_one_hot, dim=1) # 转换为类别索引

    num_classes = outputs.shape[1]

    # 计算 Precision, Recall, AUPR (假设这些函数期望类别索引作为 targets)
    precision, recall = calculate_precision_recall(all_predictions, all_targets_index)
    aupr_scores = calculate_aupr(all_predictions, all_targets_index, num_classes)

    print(f'Test Precision: {precision:.4f}, Recall: {recall:.4f}')
    for class_name, aupr in aupr_scores.items():
        print(f' {class_name}: {aupr:.4f}')

    # 计算并绘制 P-R 曲线 (calculate_pr_curve 期望类别索引)
    pr_data, pr_curve_fig = calculate_pr_curve(all_predictions, all_targets_index, num_classes)
    pr_curve_path = os.path.join(output_dir, 'pr_curve_test.png')
    pr_curve_fig.savefig(pr_curve_path)
    print(f'P-R Curve saved to {pr_curve_path}')

    # 保存测试指标到文件 (可选)
    with open(os.path.join(output_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f'Test Loss: {test_loss:.4f}\n')
        f.write(f'Test Precision: {precision:.4f}, Recall: {recall:.4f}\n')
        for class_name, aupr in aupr_scores.items():
            f.write(f' {class_name}: {aupr:.4f}\n')

if __name__ == '__main__':
    # --- Create Output Directory ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load Test Data ---
    test_images_dir = r'D:\DR2\data\test\images'  # 替换为你的测试图像目录
    test_labels_dir = r'D:\DR2\data\test\labels'  # 替换为你的测试标签目录
    test_dataset = create_test_dataset(test_images_dir, test_labels_dir, IMAGE_SIZE)
    test_loader = create_test_loader(test_dataset, BATCH_SIZE, NUM_WORKERS)

    # --- Load Trained Model ---
    model = ResUNet(in_channels=3, out_channels=5).to(DEVICE) # 确保这里的 out_channels 与你的模型定义一致
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    print(f'Loaded model weights from {CHECKPOINT_PATH}')

    # --- Test the Model ---
    test_model(model, test_loader, DEVICE, OUTPUT_DIR)

    print('Testing finished!')