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

        # 经过上面 4 次缩放，图片变成了 256个通道，大小是 8x8。
        # 展平后总共的数字个数是：256 * 8 * 8 = 16384。
        # 用全连接层 (Linear) 把这 16384 个特征映射为长度为 latent_dim (默认128) 的均值向量。
        self.fc_mu = nn.Linear(16384, latent_dim)
        # 同理，映射出长度为 128 的对数方差向量。
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
        std = torch.exp(0.5 * logvar) #std标准差
        eps = torch.randn_like(std) #随机噪声 (epsilon)
        return mu + eps * std #得到结果 z 既有随机性，又有 mu 和 logvar 的梯度，模型就可以正常反向传播Backpropagation

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


import torch.optim as optim
import torch.nn.functional as F

# ================= 3. Loss Function =================
def vae_loss(recon_x, x, mu, logvar):
    # 1. Reconstruction Loss(重建损失): 计算生成图和原图的均方误差 (MSE)
    # 使用 reduction='sum' 将 batch 内所有像素的误差累加
    RECON = F.mse_loss(recon_x, x, reduction='sum')

    # 2. KL Divergence(KL 散度): 衡量预测分布与标准正态分布的差异
    # 理论推导公式: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 最终 Loss 是两者的总和
    return RECON + KLD


# ================= 4. Training Loop =================
if __name__ == '__main__':
    # 超参数设置
    BATCH_SIZE = 32
    EPOCHS = 30  # 初始设为 30 轮，A100 跑起来很快
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Currently used computing device: {DEVICE}")

    # 数据加载
    train_dir = "/home/groups/comp3710/OASIS/keras_png_slices_train"
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = OASISDataset(train_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型和优化器
    model = VAE(latent_dim=128).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 开始训练
    print("Starting to train the VAE model...")
    model.train()

    for epoch in range(EPOCHS):
        train_loss = 0
        for batch_idx, data in enumerate(dataloader):
            # 将数据加载到显卡
            data = data.to(DEVICE)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播 (Forward)
            recon_batch, mu, logvar = model(data)

            # 计算损失 (Loss)
            loss = vae_loss(recon_batch, data, mu, logvar)

            # 反向传播 (Backward) 与 权重更新
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        # 打印每个 epoch 的平均损失
        avg_loss = train_loss / len(dataset)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Average Loss: {avg_loss:.4f}")

    # 训练结束后，保存模型的权重字典
    torch.save(model.state_dict(), "vae_oasis_weights.pth")
    print("Training complete! The model weights have been saved as vae_oasis_weights.pth")


# ================= 5. Manifold Visualization (流形可视化) =================
def visualize_manifold(model, device, num_images=10, img_size=128, latent_dim=128):
    import matplotlib.pyplot as plt
    import numpy as np

    print("正在生成流形可视化图像 (vae_manifold.png)...")
    model.eval()

    # 创建一个 10x10 的网格，在标准正态分布的范围内 (-3 到 3) 进行均匀采样
    grid_x = torch.linspace(-3, 3, num_images)
    grid_y = torch.linspace(-3, 3, num_images)

    # 创建一个大画布用来存放拼起来的 100 张图像
    manifold = torch.zeros((img_size * num_images, img_size * num_images))

    with torch.no_grad():
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                # 初始化一个全为 0 的潜在向量 [1, 128]
                z = torch.zeros(1, latent_dim).to(device)
                # 改变前两个维度的值，探索潜在空间的 2D 切面
                z[0, 0] = xi
                z[0, 1] = yi

                # 将 z 送入解码器生成图像
                out = model.fc_decode(z)
                out = out.view(out.size(0), 256, 8, 8)
                sample = model.decoder(out)

                # 将生成的 128x128 图像放入大画布的对应位置
                sample = sample.squeeze().cpu()
                manifold[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size] = sample

    # 保存图像到当前目录
    plt.figure(figsize=(10, 10))
    plt.imshow(manifold.numpy(), cmap='gray')
    plt.axis('off')
    plt.savefig('vae_manifold.png', bbox_inches='tight')
    plt.close()
    print("流形可视化图像保存成功！")


# 调用可视化函数
visualize_manifold(model, DEVICE, latent_dim=128)