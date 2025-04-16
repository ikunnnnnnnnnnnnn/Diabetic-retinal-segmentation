import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from model import ResUNet  # Make sure this UNet class matches the one you are using
from data import DRDataset  #  Make sure this DRDataset class matches the one you are using
import tqdm
import os

import numpy as np

from metrics import calculate_precision_recall, calculate_aupr, calculate_pr_curve  #  Make sure this metrics.py  matches the one you are using


# --- Hyperparameters ---
NUM_EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
IMAGE_SIZE = (512, 512)
VAL_PERCENT = 0.1
SAVE_CHECKPOINT = True
CHECKPOINT_PATH = 'checkpoints/best_model.pth'
ACCUMULATION_STEPS = 4  # 梯度累积步数
USE_AMP = True  # 是否使用混合精度训练


# --- CUDA Setup ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')


def create_datasets(images_dir, labels_dir, val_percent=0.1, image_size=(512, 512)):
    """创建训练集和验证集."""
    dataset = DRDataset(images_dir=images_dir, labels_dir=labels_dir, target_size=image_size)
    dataset_size = len(dataset)
    val_size = int(val_percent * dataset_size)
    train_indices = list(range(dataset_size - val_size))
    val_indices = list(range(dataset_size - val_size, dataset_size))
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset


def create_data_loaders(train_dataset, val_dataset, batch_size=4, num_workers=4):
    """创建训练集和验证集的数据加载器."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return train_loader, val_loader



def calculate_class_weights(train_loader, num_classes, device):
    """计算训练集中每个类别的权重 (内存高效版本)."""
    class_counts = torch.zeros(num_classes).to(device)
    total_pixels = 0
    for _, labels in train_loader:
        labels = labels.to(device)
        labels = torch.argmax(labels, dim=1)
        for i in range(num_classes):
            class_counts[i] += torch.sum(labels == i)
        total_pixels += torch.prod(torch.tensor(labels.shape, dtype=torch.float)).item()

    weights = total_pixels / (class_counts + 1e-6)
    weights /= weights.sum()
    return weights



def weighted_cross_entropy_loss(outputs, targets, weights):
    """带权重的交叉熵损失."""
    loss = F.cross_entropy(outputs, targets, reduction='none')
    loss = loss * weights[targets]
    return torch.mean(loss)



def dice_loss(outputs, targets, smooth=1e-6):
    """Dice 损失."""
    outputs = F.softmax(outputs, dim=1)
    outputs = torch.argmax(outputs, dim=1)
    targets = targets.long()
    outputs_one_hot = F.one_hot(outputs, num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()
    targets_one_hot = F.one_hot(targets, num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (outputs_one_hot * targets_one_hot).sum(dim=(2, 3))
    union = outputs_one_hot.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
    dice = ((2.0 * intersection + smooth) / (union + smooth)).mean(dim=1)
    return 1 - dice.mean()



class MixedLoss(nn.Module):
    """混合损失，结合交叉熵和 Dice 损失."""
    def __init__(self, alpha=0.5, use_weights=True):
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.use_weights = use_weights

    def forward(self, outputs, targets, weights=None):
        ce_loss = F.cross_entropy(outputs, targets)
        if self.use_weights and weights is not None:
            ce_loss = weighted_cross_entropy_loss(outputs, targets, weights)
        dc_loss = dice_loss(outputs, targets)
        return self.alpha * ce_loss + (1 - self.alpha) * dc_loss



def train_one_epoch(model, optimizer, criterion, train_loader, device, epoch, class_weights, scaler,
                    accumulation_steps):
    """训练一个epoch."""
    model.train()
    epoch_loss = 0.0
    loop = tqdm.tqdm(train_loader, leave=True)
    for batch_idx, (images, labels) in enumerate(loop):
        images = images.to(device)
        labels = labels.to(device, dtype=torch.long)

        with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):  # 混合精度
            outputs = model(images)
            labels = torch.argmax(labels, dim=1)
            # print("Outputs shape:", outputs.shape)
            # print("Outputs dtype:", outputs.dtype)
            # print("Labels shape:", labels.shape)
            # print("Labels dtype:", labels.dtype)
            # print("Outputs min:", torch.min(outputs))
            # print("Outputs max:", torch.max(outputs))
            # print("Labels min:", torch.min(labels))
            # print("Labels max:", torch.max(labels))
            loss = criterion(outputs, labels, class_weights)
            loss = loss / accumulation_steps  # 梯度累积

        scaler.scale(loss).backward()  # 混合精度

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accumulation_steps
        loop.set_postfix(loss=loss.item())

    epoch_loss /= len(train_loader)
    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Training Loss: {epoch_loss:.4f}')
    return epoch_loss



def validate_one_epoch(model, criterion, val_loader, device, epoch, output_dir=''):
    """验证一个epoch."""
    model.eval()
    val_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)

            with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):  # 混合精度
                outputs = model(images)
                labels = torch.argmax(labels, dim=1)
                loss = criterion(outputs, labels)  # 不传递 weights

            val_loss += loss.item()

            all_predictions.append(outputs.cpu())
            all_targets.append(labels.cpu())

    val_loss /= len(val_loader)
    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Validation Loss: {val_loss:.4f}')

    # 在验证结束时计算并保存 P-R 曲线
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    num_classes = outputs.shape[1]

    if output_dir:
        pr_curve_path = os.path.join(output_dir, f'pr_curve_epoch_{epoch + 1}.png')
        pr_data, pr_curve_fig = calculate_pr_curve(all_predictions, all_targets, num_classes)
        pr_curve_fig.savefig(pr_curve_path)
        print(f' P-R Curve saved to {pr_curve_path}')

    return val_loss



def save_model(model, path, val_loss, best_val_loss, save_checkpoint=True):
    """保存模型."""
    if save_checkpoint and val_loss < best_val_loss:
        torch.save(model.state_dict(), path)
        print(f'Checkpoint saved to {path}')
        return val_loss
    return best_val_loss



if __name__ == '__main__':
    # --- Data Loading ---
    train_images_dir = r'D:\DR2\data\train\images'  # 替换为你的训练图像目录
    train_labels_dir = r'D:\DR2\data\train\labels'  # 替换为你的训练标签目录
    output_dir = 'output_dir'  # 指定输出目录
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    train_dataset, val_dataset = create_datasets(
        images_dir=train_images_dir,  # 使用 train_images_dir
        labels_dir=train_labels_dir,  # 使用 train_labels_dir
        val_percent=VAL_PERCENT,
        image_size=IMAGE_SIZE
    )

    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # --- Model, Loss, Optimizer ---
    model = ResUNet(in_channels=3, out_channels=5).to(DEVICE)  #  Make sure the  in_channels and out_channels  match your dataset.
    criterion = MixedLoss(alpha=0.5, use_weights=True).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)  # 混合精度

    # --- Training Loop ---
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        class_weights = calculate_class_weights(train_loader, num_classes=5, device=DEVICE)  #这里的  num_classes  要和你的模型输出通道数一致
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, DEVICE, epoch, class_weights, scaler,
                                    ACCUMULATION_STEPS)
        val_loss = validate_one_epoch(model, criterion, val_loader, DEVICE, epoch, output_dir)
        best_val_loss = save_model(model, CHECKPOINT_PATH, val_loss, best_val_loss, SAVE_CHECKPOINT)

    print('Training finished!')
