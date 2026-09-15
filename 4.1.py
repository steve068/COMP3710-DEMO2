import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
# ================= 1. dataset definition =================
# Data Pipeline(数据加载管道)
# 自定义数据集类来读取 OASIS 的 PNG 切片
# Tell which folder in the cluster to get the data from
# and specify to open these brain MRI images(大脑核磁共振图像)
# in grayscale mode (single channel)
#Batch 张量形状 (Batch, Channel, Height, Width): torch.Size([32, 1, 128, 128]) #32 pic a batch, every pic is a channel(grayscale image)
#像素值范围: min=0.0000, max=0.9569
class OASISDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(root_dir, '*.png'))
        self.transform = transform #Uniform pics into128 x 128 size, and convert into tensors that a neural network can understand

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # MR 图像是单通道灰度图，使用 'L' 模式
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image


# ================= 2. VAE model definition =================
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()

        # 编码器: 逐步缩小图片尺寸并提取特征
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.ReLU()
        )

        # 展平后的特征数：256通道 * 8宽 * 8高 = 16384
        self.fc_mu = nn.Linear(16384, latent_dim)
        self.fc_logvar = nn.Linear(16384, latent_dim)

        # 解码器: 将隐向量恢复为图片
        self.fc_decode = nn.Linear(latent_dim, 16384)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.Sigmoid()  # 限制输出在 0-1 之间，与原图归一化范围一致
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # 展平

        # 采样
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        # 解码
        out = self.fc_decode(z)
        out = out.view(out.size(0), 256, 8, 8)  # 恢复为多通道二维特征图
        out = self.decoder(out)

        return out, mu, logvar
if __name__ == '__main__':
    # 模拟一张脑部图像张量输入到模型中测试
    dummy_input = torch.randn(32, 1, 128, 128)
    print(f"输入形状: {dummy_input.shape}")

    model = VAE(latent_dim=128)
    reconstructed_img, mu, logvar = model(dummy_input)

    print(f"重建出来的图像形状: {reconstructed_img.shape}")
    print(f"均值 (mu) 形状: {mu.shape}")