import numpy as np
import matplotlib.pyplot as plt

def k_medoids(data, k, max_iter=1000000):
    # 随机选择 k 个数据点作为聚类中心
    centers = data[np.random.choice(len(data), size=k, replace=False)]
    for _ in range(max_iter):
        # 计算每个数据点到所有聚类中心的距离 计算欧几里得距离
        distances = np.linalg.norm(data[:, np.newaxis, :] - centers, axis=-1)
        # 将每个数据点分配到距离最近的聚类中心所在的簇 np.argmin取各向量的最小值
        labels = np.argmin(distances, axis=1)

        # 对于每个簇，选择其中一个数据点作为新的聚类中心
        new_centers = np.zeros_like(centers)
        for i in range(k):
            # 数据集中属于指定簇 i 的数据点提取出来
            cluster = data[labels == i]
            if len(cluster) > 0:
                # 三维数组，每个维度的差异
                costs = np.sum(np.abs(cluster[:, np.newaxis, :] - cluster), axis=-1)
                index = np.argmin(np.sum(costs, axis=1))
                new_centers[i] = cluster[index]
            else:
                new_centers[i] = centers[i]
        # 如果新的聚类中心与旧的聚类中心相同，则停止迭代
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels
def load_data():
    f = open("./data mining ex3/data.txt", "r", encoding="UTF-8")
    res = []
    for i in f.readlines():
        res.append([float(x) for x in i.split()])
    return np.array(res)

X = load_data()
k = 20

# 聚类
centers, labels = k_medoids(X, k=k)

# 打印结果
print('中心点：', centers)
print('标签：', labels)

# 绘制聚类结果
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#FFFF00', '#00FFFF',
          '#FFA500', '#800080', '#00FF7F', '#FFD700', '#8B0000', '#228B22',
          '#4169E1', '#FF1493', '#B8860B', '#9932CC', '#DC143C', '#008080', '#FFC0CB', '#7FFF00']
for i in range(k):
    cluster = X[labels == i]
    center = centers[i]
    ax.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], label=f'Cluster {i+1}', s=50)
    ax.scatter(center[0], center[1], marker='*', c=colors[i], s=200, edgecolor='black')
ax.legend()
plt.show()

'''
    k-means 和 k-medoids 是两种常用的聚类算法，它们在质心的选择和计算方式上有所不同。

    1. K-means:
    - 质心选择：K-means 算法使用数据点的均值作为质心。
    - 质心更新：在每次迭代中，K-means 计算每个数据点与当前质心之间的距离，并将数据点分配给最近的质心所属的簇。然后，重新计算每个簇的质心位置，使用簇内数据点的均值更新质心位置。
    - 优化目标：K-means 的优化目标是最小化每个数据点与所属质心之间的距离的平方和（平方误差和）。

    2. K-medoids:
    - 质心选择：K-medoids 算法选择簇内实际的数据点作为质心，而不是使用均值。
    - 质心更新：在每次迭代中，K-medoids 计算每个数据点与当前质心之间的距离，并将数据点分配给最近的质心所属的簇。然后，重新选择簇内其他数据点作为新的质心，以最小化簇内数据点与质心之间的距离之和。
    - 优化目标：K-medoids 的优化目标是最小化每个簇内数据点与所属质心之间的距离的和。

    区别总结：
    - K-means 使用均值作为质心，而 K-medoids 使用实际的数据点作为质心。
    - K-means 使用均值更新质心位置，而 K-medoids 重新选择簇内其他数据点作为新的质心。
    - K-means 最小化平方误差和作为优化目标，而 K-medoids 最小化数据点与质心距离的和作为优化目标。

    在实践中，K-medoids 对离群点更具鲁棒性，因为它使用实际的数据点作为质心，而不受离群点的影响。然而，由于 K-medoids 需要计算所有数据点之间的距离，并且选择新的质心，因此在计算复杂度上相对较高。相比之下，K-means 在计算上更高效，但对离群点敏感。选择使用哪种算法应根据具体的问题和数据集特点来决定。
'''