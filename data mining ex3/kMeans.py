import numpy as np
import matplotlib.pyplot as plt
def k_means(data, k, max_iter=100):
    # 随机选取k个中心点
    centroids = data[np.random.choice(len(data), k)]
    for i in range(max_iter):
        # 计算每个点到最近的中心点的距离 ||x||₂ = sqrt(x[0]² + x[1]² + ... + x[n-1]²)
        # np.newaxis改变了数组的维数，增加了一维
        distances = np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=-1)
        # 根据距离将数据点分配到最近的中心点所在的簇
        labels = np.argmin(distances, axis=1)
        # 计算每个簇的平均值，并将其设置为新的中心点
        new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(k)])
        # 如果新的中心点与旧的中心点相同，则停止迭代
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

def load_data():
    f = open("./data mining ex3/data.txt", "r", encoding="UTF-8")
    res = []
    for i in f.readlines():
        res.append([float(x) for x in i.split()])
    return np.array(res)

X = load_data()
centroids, labels = k_means(X, 20, 1000000)
# 打印结果
print('中心点：', centroids)
print('标签：', labels)
# 绘制聚类结果
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
plt.show()