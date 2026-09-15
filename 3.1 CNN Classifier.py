import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people

# 从Part 2获取数据
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.images
y = lfw_people.target
n_classes = lfw_people.target_names.shape[0]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 将数据转换为PyTorch需要的四维格式 (N, 1, H, W)
X_train_reshaped = X_train[:, np.newaxis, :, :]
X_test_reshaped = X_test[:, np.newaxis, :, :]

# 转换为PyTorch Tensors
X_train_tensor = torch.tensor(X_train_reshaped, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 构建DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F


class LFW_CNN(nn.Module):
    def __init__(self, num_classes):
        super(LFW_CNN, self).__init__()

        # 第一层卷积: 3x3, 32 filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二层卷积: 3x3, 32 filters
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层 (Dense layers)
        # 注意: 这里的 32 * 12 * 9 需要根据你输入图像的实际长宽来计算
        # LFW (resize=0.4) 默认尺寸通常是 50x37
        # 经过两次 2x2 池化后，尺寸大致变为 12x9
        self.fc1 = nn.Linear(32 * 12 * 9, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # 展平操作
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


import torch.optim as optim

# 初始化模型、损失函数和优化器
model = LFW_CNN(num_classes=n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters()) # 使用要求的Adam优化器

# 设定训练轮数 (Epochs)
num_epochs = 20

print("开始训练...")
# === 1. 训练阶段 ===
for epoch in range(num_epochs):
    model.train()  # 将模型设置为训练模式
    running_loss = 0.0

    # 从 DataLoader 中按批次 (Batch) 提取数据
    for inputs, labels in train_loader:
        # a. 梯度清零
        optimizer.zero_grad()

        # b. 前向传播 (Forward Pass)
        outputs = model(inputs)

        # c. 计算损失 (Calculate Loss)
        loss = criterion(outputs, labels)

        # d. 反向传播 (Backward Pass)
        loss.backward()

        # e. 更新权重 (Update Weights)
        optimizer.step()

        running_loss += loss.item()

    # 打印当前 epoch 的平均损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

print("训练完成。\n开始测试评估...")

# === 2. 评估阶段 ===
model.eval()  # 将模型设置为评估模式
correct = 0
total = 0

# 关闭梯度计算以节省内存并加速运算
with torch.no_grad():
    # 对测试集进行前向传播
    outputs = model(X_test_tensor)

    # 获取预测类别 (在第1维度上取最大得分的索引)
    _, predicted = torch.max(outputs.data, 1)

    # 统计正确数量
    total = y_test_tensor.size(0)
    correct = (predicted == y_test_tensor).sum().item()

# 计算并打印最终准确率
accuracy = correct / total
print(f"测试集总数: {total}")
print(f"预测正确数: {correct}")
print(f"CNN分类准确率: {accuracy * 100:.2f}%")










class BSTNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def insert(node, newNode):
    """ Insert newNode into the subtree with root node """
    if newNode.value >= node.value:
        if node.right is None:
            node.right = newNode
        else:
            insert(node.right, newNode)
    else: # newNode.value < node.value
        if node.left is None:
            node.left = newNode
        else:
            insert(node.left, newNode)

def buildBST(A):
    """ A is an array of stuff """
    assert len(A) != 0
    root = BSTNode(A[0])
    for v in A[1:]:
        node = BSTNode(v)
        insert(root, node)
    return root

