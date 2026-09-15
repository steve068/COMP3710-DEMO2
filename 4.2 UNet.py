import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ================= 1. Dataset Definition (图像分割) =================
class OASISSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(128, 128)):
        """
        img_dir: 原始核磁共振图像的路径
        mask_dir: 对应的分割标签掩码的路径
        """
        # 使用 sorted 确保原图和掩码的文件列表顺序完全一致，实现一一对应
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
        self.img_size = img_size

        # 确保找出的图片数量和掩码数量是对等的
        assert len(self.img_paths) == len(self.mask_paths), "图像和掩码的数量不一致！"

        # 输入图像的变换：转为 Tensor 会自动将像素值缩放到 [0.0, 1.0]
        self.img_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. 读取原始图像 (输入 X)
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('L')  # 保持单通道灰度
        image = self.img_transform(image)  # 形状: [1, 128, 128], 类型: FloatTensor

        # 2. 读取掩码图像 (标签 Y)
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path).convert('L')

        # [KEY 1]：掩码包含的是类别索引 (0, 1, 2...)。
        # 在缩放时绝不能使用默认的双线性插值，否则会产生 0.5 这样的假类别！
        # 必须使用最近邻插值 (NEAREST) 保持类别的整数纯净度。
        mask = mask.resize(self.img_size, Image.NEAREST)

        # [KEY 2]：对于分割和分类任务，PyTorch 要求标签 (Label) 必须是长整型 (LongTensor)。
        mask_array = np.array(mask, dtype=np.int64)

        # 根据数据集的具体情况，有时候掩码的像素值可能是 0, 85, 170, 255
        # 如果是这种情况，我们需要将它们映射为标准的 0, 1, 2, 3 类别索引
        # (这里预留了一个转换映射，实际运行时视具体掩码值而定)
        unique_values = np.unique(mask_array)
        for i, val in enumerate(unique_values):
            mask_array[mask_array == val] = i

        mask_tensor = torch.from_numpy(mask_array)  # 形状: [128, 128], 类型: LongTensor

        return image, mask_tensor

import torch
import torch.nn as nn
# ================= 2. UNet Model Definition (UNet 网络架构) =================
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        """
        in_channels: 1 (因为是单通道灰度 MRI 图像)
        out_channels: 4 (对应我们刚才测出来的 0, 1, 2, 3 四个类别)
        """
        super(UNet, self).__init__()

        # 编码器 Encoder (下采样阶段，提取特征)
        self.enc1 = self.conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self.conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        # 瓶颈层 Bottleneck (最深层)
        self.bottleneck = self.conv_block(128, 256)

        # 解码器 Decoder (上采样阶段，恢复分辨率 + 跳跃连接)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # 注意这里的 in_channels 是 256，因为 up3(128) + enc3(128) 拼接(concat)在一起
        self.dec3 = self.conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32)

        # 输出层 Output Layer (将通道数映射为类别数 4)
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_c, out_c):
        # 两次 3x3 卷积 + ReLU 激活函数
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),  # 加入 BatchNorm 加速收敛
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 1. 编码阶段 (提取特征，记录跳跃连接需要的特征图)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # 2. 瓶颈阶段
        b = self.bottleneck(self.pool3(e3))

        # 3. 解码阶段 (使用 torch.cat 实现跳跃连接 Skip Connections)
        d3 = self.up3(b)
        d3 = torch.cat((e3, d3), dim=1)  # 拼上 e3
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e2, d2), dim=1)  # 拼上 e2
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((e1, d1), dim=1)  # 拼上 e1
        d1 = self.dec1(d1)

        # 4. 输出预测的 Logits (形状: [Batch, 4, 128, 128])
        out = self.out_conv(d1)
        return out


# ================= 测试数据管道与 UNet 前向传播 =================
if __name__ == '__main__':
    # 路径配置
    TRAIN_IMG_DIR = "/home/groups/comp3710/OASIS/keras_png_slices_train"
    TRAIN_MASK_DIR = "/home/groups/comp3710/OASIS/keras_png_slices_seg_train"

    try:
        # 1. 测试数据管道
        dataset = OASISSegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        images, masks = next(iter(dataloader))
        print("--- 数据管道测试 ---")
        print(f"输入图像 (X) 形状: {images.shape}")
        print(f"标签掩码 (Y) 形状: {masks.shape}")

        # 2. 测试 UNet 前向传播
        print("\n--- UNet 前向传播测试 ---")
        model = UNet(in_channels=1, out_channels=4)

        # 将图像输入模型进行前向传播
        outputs = model(images)

        print(f"模型输出预测 形状: {outputs.shape}")

        # 验证输出规格是否正确 (期望: [Batch=4, Classes=4, H=128, W=128])
        if outputs.shape == (4, 4, 128, 128):
            print("✅ 前向传播测试通过：输出张量形状正确，跳跃连接正常运行。")
        else:
            print("❌ 输出形状错误。")

    except Exception as e:
        print(f"遇到错误: {e}")