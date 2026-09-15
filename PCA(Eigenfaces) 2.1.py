from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

# 1. 准备和加载数据
# Download the data, if not already on disk and load it as numpy arrays
# 加载数据集，过滤掉照片少于70张的人，并将图像缩小到原来的40%（为了计算更快）

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
# 获取单张图片的高度(h)和宽度(w)，方便后面画图
#lfw_people.images 是一个三维的 Numpy 数组，包含所有图像数据。
#.shape 返回三维数组的维度信息, (样本总数, h, w)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
# 提取特征矩阵 X：模型不看二维图，所以每张人脸图都被展平（flatten）成了一维数组
X = lfw_people.data
n_features = X.shape[1]# 计算一张图有多少个像素点（即初始特征维度）

# the label to predict is the id of the person
# 提取标签 y：每张脸对应的真实身份 ID
y = lfw_people.target
target_names = lfw_people.target_names# 身份 ID 对应的真实名字
n_classes = target_names.shape[0]# 算一下总共有几个人（类别）

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# Split into a training set and a test set using a stratified k fold
# 划分训练集(75%)和测试集(25%)。random_state=42 保证每次运行切分的方式都一样
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 2. PCA 特征提取 (Eigenfaces)
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
# 设定降维目标：把几千维的像素，浓缩成 150 个最重要的特征（即 150 张特征脸）
n_components = 150

# Center data
# 1. 中心化 (Center data)：求所有训练集人脸的平均值，得到一张“平均脸”
mean = np.mean(X_train, axis=0)
# 让所有脸减去“平均脸”。PCA算法要求数据必须先做中心化，提取每个人脸区别于大众脸的“个性”
X_train -= mean
X_test -= mean

# Eigen-decomposition (修复了 PDF 中遗漏的 np.linalg.svd)
# 2. 特征分解：使用 SVD（奇异值分解）计算 PCA。这是线性代数的核心操作。
# U 是左奇异矩阵，S 是奇异值，V 是右奇异矩阵（这里面装的就是我们需要的主成分方向）
U, S, V = np.linalg.svd(X_train, full_matrices=False)
# 取出 V 的前 150 行，这就是我们提取出的 150 个最重要的“特征向量”
components = V[:n_components]
# 将这 150 个一维向量，重新折叠回二维图像的形状 (h, w)，它们就是所谓的“特征脸 (eigenfaces)”
eigenfaces = components.reshape((n_components, h, w))

# project into PCA subspace
# 3. 投影降维：用矩阵点乘 (np.dot)
# 把原本高维的训练集 X_train 投影到这 150 个特征方向上。
# 现在，每一张人脸不再是几千个像素，而是变成了 150 个数字！
X_transformed = np.dot(X_train, components.T)
print("X_transformed shape:", X_transformed.shape)

X_test_transformed = np.dot(X_test, components.T)
print("X_test_transformed shape:", X_test_transformed.shape)

# 3. 绘制 Eigenfaces (特征脸)
# Qualitative evaluation of the predictions using matplotlib
# 画出降维后的“特征脸”
# 这里的 plot_gallery 是个自定义的画图函数，用 plt.subplot 循环把 12 张图画在一起
# plt.imshow(..., cmap=plt.cm.gray) 表示以灰度图显示
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    # 修复了 PDF 中缺失的乘号
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.show()

# 4. 绘制紧凑度 (Compactness)
# 评估降维的性能
# 画出“紧凑度 (Compactness)”曲线
# 利用奇异值 S 计算每个主成分包含的信息量（方差）
explained_variance = (S ** 2) / (n_samples - 1)
total_var = explained_variance.sum()
# 计算前 150 个特征包含的信息量占总信息量的百分比，并用 np.cumsum 累加
explained_variance_ratio = explained_variance / total_var
ratio_cumsum = np.cumsum(explained_variance_ratio)
print("ratio_cumsum shape:", ratio_cumsum.shape)
# 画折线图：横坐标是特征数量（0到150），纵坐标是保留的信息百分比（0.0到1.0）
eigenvalueCount = np.arange(n_components)
plt.plot(eigenvalueCount, ratio_cumsum[:n_components])
plt.title('Compactness')
plt.show()

# 5. 构建并评估随机森林分类器
# build random forest
estimator = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=150)
estimator.fit(X_transformed, y_train) # expects X as [n_samples, n_features]

predictions = estimator.predict(X_test_transformed)
# 修复了 PDF 中的语法错误
correct = (predictions == y_test)
total_test = len(X_test_transformed)

# print("Gnd Truth:", y_test)
print("Total Testing:", total_test)
print("Predictions:", predictions)
print("Which Correct:", correct)
print("Total Correct:", np.sum(correct))
print("Accuracy:", np.sum(correct) / total_test)
print(classification_report(y_test, predictions, target_names=target_names))