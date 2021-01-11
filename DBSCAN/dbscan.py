# ライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
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
    # plt.xlim(0,1920)
    # plt.ylim(0,1080)
    fig.savefig('figure/test.png')


# 塊のデータセット
# dataset1 = datasets.make_blobs(
#     n_samples=1000, random_state=10, centers=6, cluster_std=1.2)[0]
# 月のデータセット
# data = datasets.make_moons(n_samples=1000, noise=.05)[0]

with open('test.csv') as f:
    reader = csv.reader(f)
    l = [row for row in reader]
l = l[1::]
data = np.array([[int(v) for v in row] for row in l])
data = np.delete(data, [2, 5, 8, 11, 14, 17, 20,
                        23, 26, 29, 32, 35, 38, 41, 44], 1)

print(f'データ形状：{np.shape(data)}')
dbscan = DBSCAN(data, eps=130, min_samples=30)
dbscan.calc()
print(dbscan.dbscan_data[0:10])

print(f"クラスタ：{dbscan.cluster_num}, ノイズ：{dbscan.noise_num}")

cluster_plots(data, dbscan.dbscan_data)
