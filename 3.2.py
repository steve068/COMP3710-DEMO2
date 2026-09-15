import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar10_dataloaders(batch_size=128):
    # 训练集的数据增强：随机裁剪和水平翻转
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # CIFAR-10 的标准均值和标准差归一化
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 测试集只需要转换为张量并归一化，不需要做增强
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 自动下载并加载数据
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

    return trainloader, testloader


# 定义 ResNet 的基础残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        # 如果步长不为 1，或者输入输出通道数不一致，需要用 1x1 卷积调整 shortcut 的维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接：加上输入 x
        out = F.relu(out)
        return out


# 构建整个 ResNet 架构
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # 【核心修改点】针对 CIFAR-10 32x32 的适配
        # 移除了原始 ImageNet 版本的 7x7 卷积和 MaxPool，改为 3x3 卷积，stride=1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 堆叠 4 层残差块
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # 全局平均池化
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# 实例化 ResNet-18 的辅助函数 (ResNet-18 的 blocks 配置为 [2, 2, 2, 2])
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

if __name__ == '__main__':
    import time
    import torch.optim as optim
    from torch.cuda.amp import autocast, GradScaler

    # 1. 硬件与数据准备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #print(f"正在使用的计算设备: {device}")
    print(f"The computing device in use: {device}")

    model = ResNet18().to(device)
    trainloader, testloader = get_cifar10_dataloaders(batch_size=128)

    # 2. 定义损失函数、优化器与超参数
    epochs = 48  # 使用 OneCycleLR，50 个 epoch 使其收敛到 94%
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # prevent 过拟合Overfitting

    # 使用带动量的 SGD 优化器，加入 Nesterov 动量和权重衰减
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # 3. 初始化 AMP Scaler 和 OneCycleLR 调度器
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.3,  # 允许达到的最大学习率
        steps_per_epoch=len(trainloader),
        epochs=epochs
    )

    # 4. 开始计时与训练循环
    #print("开始混合精度训练...")
    print("Starting mixed precision training...")

    total_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        epoch_start_time = time.time()

        for i, (inputs, labels) in enumerate(trainloader):

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # 开启自动混合精度上下文
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # 缩放损失并反向传播
            scaler.scale(loss).backward()

            # 缩放器更新优化器参数
            scaler.step(optimizer)
            scaler.update()

            # 调度器在每个 batch 后更新学习率
            scheduler.step()

            # 统计训练指标
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        # 5. 每个 Epoch 结束后的测试集评估
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testloader):

                inputs, labels = inputs.to(device), labels.to(device)
                # 推理阶段也可以使用 AMP 加速
                with autocast():
                    outputs = model(inputs)
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()

        epoch_time = time.time() - epoch_start_time
        train_acc = 100. * correct_train / total_train
        test_acc = 100. * correct_test / total_test

        print(f"Epoch {epoch + 1}/{epochs} [{epoch_time:.1f}s] | "
              f"Train Loss: {running_loss / len(trainloader):.3f} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    total_time = time.time() - total_start_time
    #print(f"训练完成！总耗时: {total_time:.2f} 秒。")
    print(f"Training complete! Total time: {total_time:.2f} seconds")
