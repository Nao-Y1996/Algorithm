# ライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import silhouette_score
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph


class DBSCAN():
    def __init__(self, data, eps, min_samples):
        self.data = data
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan_data = None
        self.cluster_num = None
        self.noise_num = None
        self.calc_time = None

    def calc(self):
        self.dbscan_data = cluster.DBSCAN(
            eps=self.eps, min_samples=self.min_samples, metric='euclidean').fit_predict(self.data)
        self.noise_num = sum(self.dbscan_data == -1)
        if self.noise_num == 0:
            self.cluster_num = len(set(self.dbscan_data))
        else:
            self.cluster_num = len(set(self.dbscan_data)) - 1


def cluster_plots(data, colors='gray', title1='Dataset 1'):  # グラフ作成
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(title1, fontsize=14)
    ax1.scatter(data[:, 0], data[:, 1], s=8, lw=0, c=colors)
    plt.show()

# 塊のデータセット
dataset1 = datasets.make_blobs(
    n_samples=1000, random_state=10, centers=6, cluster_std=1.2)[0]
# 月のデータセット
# dataset2 = datasets.make_moons(n_samples=10000, noise=.05)[0]

dbscan = DBSCAN(dataset1, eps=1, min_samples=5)
dbscan.calc()
print(f"クラスタ：{dbscan.cluster_num}, ノイズ：{dbscan.noise_num}")
cluster_plots(dataset1, dbscan.dbscan_data)
