import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# 自定义数据集类来读取 OASIS 的 PNG 切片
class OASISDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(root_dir, '*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # MR 图像是单通道灰度图，使用 'L' 模式
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image


if __name__ == '__main__':
    # 验证数据读取（使用集群上的绝对路径）
    train_dir = "/home/groups/comp3710/OASIS/keras_png_slices_train"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = OASISDataset(train_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"成功加载 OASIS 训练集！总图片数量: {len(dataset)}")

    for imgs in dataloader:
        print(f"Batch 张量形状 (Batch, Channel, Height, Width): {imgs.shape}")
        print(f"像素值范围: min={imgs.min().item():.4f}, max={imgs.max().item():.4f}")
        break