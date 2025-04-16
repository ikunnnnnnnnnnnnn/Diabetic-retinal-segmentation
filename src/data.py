import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def create_multiclass_mask(label, num_classes=5):
    """
    将单通道多类别标签转换为多通道的 one-hot 编码掩膜。

    Args:
        label (numpy.ndarray): 单通道多类别标签图像，形状为 (height, width)。
        num_classes (int): 类别总数 (包括背景)。

    Returns:
        numpy.ndarray: 多通道的 one-hot 编码掩膜，形状为 (num_classes, height, width)。
    """

    height, width = label.shape
    multiclass_mask = np.zeros((num_classes, height, width), dtype=np.float32)

    for c in range(num_classes):
        multiclass_mask[c, :, :] = (label == c).astype(np.float32)

    return multiclass_mask

class DRDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, target_size=(512, 512)):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.png')])  # Adjust extension if needed
        self.label_paths = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.png')])  # Adjust extension if needed
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # --- Preprocessing ---
        image = cv2.resize(image, self.target_size)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.from_numpy(image)

        label = cv2.resize(label, self.target_size, interpolation=cv2.INTER_NEAREST)
        label = create_multiclass_mask(label, num_classes=5)  # Implement this function as shown before
        label = torch.from_numpy(label).float()  # Or .long() if using CrossEntropyLoss

        if self.transform:
            image, label = self.transform(image, label)

        return image, label

# --- Example Usage in train.py ---
if __name__ == '__main__':
    images_dir = r'/data/train/images'
    labels_dir = r'/data/train/labels'

    dataset = DRDataset(images_dir, labels_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for images, labels in dataloader:
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        break