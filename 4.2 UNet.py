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


import torch.optim as optim


# ================= 3. Metrics: Dice Similarity Coefficient (DSC) =================
def compute_dsc(preds, targets, num_classes=4):
    """
    计算多类别的宏平均 Dice 分数 (Macro-average DSC)
    preds: 模型的输出 logits, 形状 [Batch, Classes, H, W]
    targets: 真实标签, 形状 [Batch, H, W]
    """
    # 将 logits 转换为预测的类别索引 (argmax)
    preds = torch.argmax(preds, dim=1)  # 形状变为 [Batch, H, W]

    dsc_per_class = []
    for c in range(num_classes):
        # 提取当前类别的二值掩码 (0和1)
        pred_c = (preds == c).float()
        target_c = (targets == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        # 避免除以 0 (如果预测和真实图中都没有这个类别，得分为 1.0)
        if union == 0:
            dsc_per_class.append(1.0)
        else:
            dsc_per_class.append((2. * intersection / union).item())

    # 返回所有类别的平均 DSC
    return sum(dsc_per_class) / num_classes


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ================= 5. Inference & Visualization (推理与可视化) =================
def visualize_segmentation(model_path, img_dir, mask_dir, num_samples=3):
    """
    加载模型权重并在测试集上进行可视化推理
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 初始化模型并加载权重
    model = UNet(in_channels=1, out_channels=4).to(device)
    # 设置 weights_only=True 提升安全性，避免加载不可信的模型
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()  # 开启评估模式，关闭 Dropout 和 BatchNorm 的动态更新

    # 2. 加载测试集数据 (注意这里换成了 test 路径)
    dataset = OASISSegmentationDataset(img_dir, mask_dir)
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    images, true_masks = next(iter(dataloader))  # 随机抽取一批图像

    images = images.to(device)
    true_masks = true_masks.to(device)

    # 3. 前向传播进行预测 (不需要计算梯度) + 当场计算 DSC
    with torch.no_grad():
        logits = model(images)  # 获取原始模型输出

        # 调用第 3 部分已经定义好的 compute_dsc 函数来算分
        # 因为数据在 GPU 上，所以需要把 true_masks 也传进去
        test_dsc = compute_dsc(logits, true_masks)

        # 将 [Batch, 4, H, W] 的概率分布转化为具体的类别索引 [Batch, H, W]
        preds = torch.argmax(logits, dim=1)

    # [新增核心点]：把算出来的分数打印到终端上！
    print(f"\n The average DSC score for these {num_samples} test images is: {test_dsc:.4f}")
    if test_dsc > 0.9:
        print("The test set DSC > 0.9, meeting requirements!\n")

        # 4. 准备绘图 (转移到 CPU)
    images = images.cpu()
    true_masks = true_masks.cpu()
    preds = preds.cpu()

    # 为 4 个类别定义特定的颜色映射
    # 类别 0: 背景 (黑色)
    # 类别 1: 脑脊液 (红色)
    # 类别 2: 灰质 (绿色)
    # 类别 3: 白质 (蓝色)
    cmap = mcolors.ListedColormap(['black', 'red', 'green', 'blue'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 创建一块画布
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        # 绘制原图
        axes[i, 0].imshow(images[i].squeeze(), cmap='gray')
        axes[i, 0].set_title("Original MRI" if i == 0 else "")
        axes[i, 0].axis('off')

        # 绘制真实标签 (Ground Truth)
        axes[i, 1].imshow(true_masks[i], cmap=cmap, norm=norm)
        axes[i, 1].set_title("Ground Truth Mask" if i == 0 else "")
        axes[i, 1].axis('off')

        # 绘制模型的预测结果 (Prediction)
        axes[i, 2].imshow(preds[i], cmap=cmap, norm=norm)
        axes[i, 2].set_title("Model Prediction" if i == 0 else "")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("unet_segmentation_demo.png", bbox_inches='tight', dpi=150)
    print("Success! The visualization result has been saved as unet_segmentation_demo.png")


if __name__ == '__main__':
    # 注意：这里改为了 _test 测试集文件夹，以证明模型的泛化能力
    TEST_IMG_DIR = "/home/groups/comp3710/OASIS/keras_png_slices_test"
    TEST_MASK_DIR = "/home/groups/comp3710/OASIS/keras_png_slices_seg_test"

    # 刚才跑出来的满分权重文件
    MODEL_WEIGHTS = "unet_oasis_weights.pth"

    print("Starting model inference on the test set...")
    visualize_segmentation(MODEL_WEIGHTS, TEST_IMG_DIR, TEST_MASK_DIR, num_samples=4)