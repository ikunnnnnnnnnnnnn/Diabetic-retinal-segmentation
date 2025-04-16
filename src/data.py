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


class DRDatasetPatched(Dataset):
    """
    用于从 DR 图像生成重叠图块的数据集类。
    """

    def __init__(self, images_dir, labels_dir, patch_size=(512, 512), overlap=128, transform=None):
        """
        初始化 DRDatasetPatched。

        Args:
            images_dir (str): 包含输入图像的目录。
            labels_dir (str): 包含 ground truth 标签的目录。
            patch_size (tuple, optional): 要生成的图块的大小 (height, width)。默认为 (512, 512)。
            overlap (int, optional): 相邻图块之间的重叠像素数。默认为 128。
            transform (callable, optional): 接受图像和标签并返回转换后版本的回调函数。
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.png')])
        self.label_paths = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.png')])
        self.patch_size = patch_size
        self.overlap = overlap
        self.transform = transform
        self.patches = self._generate_patches()

    def _generate_patches(self):
        """
        生成图块坐标和相应图像/标签路径的列表。

        Returns:
            list: 一个字典列表，其中每个字典包含 'image_path'、'label_path' 和 'coords' (元组 (h, w))。
        """
        patches = []
        for img_path, label_path in zip(self.image_paths, self.label_paths):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            height, width = img.shape[:2]
            patch_h, patch_w = self.patch_size
            step_h = patch_h - self.overlap
            step_w = patch_w - self.overlap

            # 计算填充以确保完全覆盖
            pad_h = (patch_h - (height % step_h)) % patch_h if height % step_h != 0 else 0
            pad_w = (patch_w - (width % step_w)) % patch_w if width % step_w != 0 else 0

            # 填充图像和标签
            img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            label_padded = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            height_padded, width_padded = img_padded.shape[:2]

            # 迭代填充后的图像以提取图块
            for start_h in range(0, height_padded - patch_h + 1, step_h):
                for start_w in range(0, width_padded - patch_w + 1, step_w):
                    patch_coords = (start_h, start_w)
                    patches.append({'image_path': img_path, 'label_path': label_path, 'coords': patch_coords})
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        """
        检索给定索引处的图像和标签图块。

        Args:
            idx (int): 要检索的图块的索引。

        Returns:
            tuple: 包含图像图块 (torch.Tensor) 和标签图块 (torch.Tensor) 的元组。
        """
        patch_info = self.patches[idx]
        image_path = patch_info['image_path']
        label_path = patch_info['label_path']
        start_h, start_w = patch_info['coords']  # 更有描述性的变量名称
        patch_h, patch_w = self.patch_size

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        image_patch = image[start_h:start_h + patch_h, start_w:start_w + patch_w]
        label_patch = label[start_h:start_h + patch_h, start_w:start_w + patch_w]
        # 检查 patch 大小是否正确。
        if image_patch.shape[:2] != self.patch_size:
            # 如果 patch 大小不正确，则使用黑色填充
            padded_image_patch = np.zeros((self.patch_size[0], self.patch_size[1], 3), dtype=np.uint8)
            padded_label_patch = np.zeros(self.patch_size, dtype=np.uint8)

            # 计算填充的起始位置
            h_offset = (self.patch_size[0] - image_patch.shape[0]) // 2
            w_offset = (self.patch_size[1] - image_patch.shape[1]) // 2

            # 将原始 patch 放入填充后的 patch 中
            padded_image_patch[h_offset:h_offset + image_patch.shape[0],
            w_offset:w_offset + image_patch.shape[1]] = image_patch
            padded_label_patch[h_offset:h_offset + image_patch.shape[0],
            w_offset:w_offset + image_patch.shape[1]] = label_patch
            image_patch = padded_image_patch
            label_patch = padded_label_patch

        # --- 预处理 ---
        image_patch = image_patch / 255.0
        image_patch = np.transpose(image_patch, (2, 0, 1)).astype(np.float32)
        image_patch = torch.from_numpy(image_patch)

        label_patch = create_multiclass_mask(label_patch, num_classes=5)
        label_patch = torch.from_numpy(label_patch).float()

        if self.transform:  # 如果提供了 transform，则应用它
            image_patch, label_patch = self.transform(image_patch, label_patch)

        return image_patch, label_patch
