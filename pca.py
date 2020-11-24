#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
import numpy as np
import itertools
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class my_PCA():
    def __init__(self, data):
        self.n = np.shape(data)[0]  # データ数
        self.dimension = np.shape(data)[1]  # 次元数
        self.eigenvalue = None  # 固有値
        self.components = None  # 固有ベクトル(主成分ベクトル)
        self.explained_variance_ratio = None  # 寄与率
        # データの中心化
        self.data = data - self.average(data)
    # 各次元の平均の計算
    def average(self, data):
        """
        Arg:
          list: 2 dimensional array
        return:
          np.array: list of Average per column
        """
        averages = []
        for dim in range(np.shape(data)[1]):
            averages.append(np.average(data[:, dim]))
        return np.array(averages)
    # 分散の計算
    def calc_variance(self, data1, data2):
        # 中心化しているので平均=0
        return np.sum(((data1-0) * (data2-0))/(self.n - 1), axis=0)
    # 分散共分散行列
    def get_Covariance_Matrix(self):
        """
        description:
          np.cov(pca.data, rowvar=0, bias=0)と等しい
          bias:データ数nで割る(標本データ)時1, n-1でわる時0
          今回は標本データだが、すでに分散を求める際にn-1で割っているためbias=0と等しくなる
        return:
          np.array: Covariance Matrix
        """
        # cvm = np.array([[0] * self.dimension for _ in range(self.dimension)], dtype=float)
        cvm = np.zeros((self.dimension, self.dimension), dtype=float)
        for row in range(self.dimension):
            for col in range(self.dimension):
                # cvm[row][col] = self.calc_variance(self.data[:,row],self.data[:,col])
                if row <= col:
                    cvm[row][col] = self.calc_variance(
                        self.data[:, row], self.data[:, col])
                else:
                    cvm[row][col] = cvm[col][row]
        return cvm
    # 固有値、固有ベクトルの計算
    def get_components(self, cvm):
        """
        Arg:
          np.array: Covariance Matrix
        returns:
          np.array: eigenvalue
          np.array: eigenvector
        """
        # 固有値eigenvalue,  固有ベクトルeigenvector(主成分ベクトル)
        # 第n主成分に対応する固有ベクトルは第n列ベクトル
        eigenvalue, eigenvector = np.linalg.eig(cvm)
        return eigenvalue, eigenvector
    # データをモデルにfitさせる(PCAの計算を行う)
    def fit(self):
        # 分散共分散行列を求めて、固有値、固有ベクトルを求める
        cvm = self.get_Covariance_Matrix()
        self.eigenvalue, self.components = self.get_components(cvm)
        # 寄与率の取得
        self.explained_variance_ratio = self.eigenvalue/sum(self.eigenvalue)
    # データの線形写像(次元削減)
    def transform(self, n_component):
        return np.dot(self.data, self.components[:, :n_component])


iris = load_iris()

# print('--------ライブラリでPCA-------')
# # 第n主成分に対応するベクトルはn行ベクトル
# pca = PCA()
# pca.fit(iris.data)
# print(f'分散共分散：\n{pca.get_covariance()}')
# print(f'主成分ベクトル：\n{pca.components_}')

print('--------自作クラスでPCA-------')
pca = my_PCA(iris.data)
pca.fit()

print(f'主成分ベクトル：\n{pca.components}')
print(f'寄与率：{pca.explained_variance_ratio}')
n_component = 2
transformed_data = pca.transform(n_component)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('pc_1')
ax.set_ylabel('pc_2')
ax.scatter(transformed_data[:50, 0], transformed_data[:50, 1],color='r')
ax.scatter(transformed_data[50:100, 0], transformed_data[50:100, 1],color='g')
ax.scatter(transformed_data[100:150, 0], transformed_data[100:150, 1],color='b')
plt.show()

