#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
import numpy as np
import itertools
# from statistics import variance # 自作コードの評価用

from sklearn.decomposition import PCA

class my_PCA():
    def __init__(self,data):
      self.n = np.shape(data)[0] # データ数
      self.dimension = np.shape(data)[1] # 次元数
      self.components = None # 固有ベクトル(主成分)
      # データの中心化
      self.data = data - self.average(data)
    #各次元の平均を計算
    def average(self,data):
      """
      Arg:
        list: 2 dimensional array
      return:
        np.array: list of Average per column
      """
      averages = []
      for dim in range(np.shape(data)[1]):
          averages.append(np.average(data[:,dim]))
      return np.array(averages)
    # 各次元の分散を計算
    def calc_variance(self, data1, data2):
        return np.sum(( (data1-0) * (data2-0) )/(self.n - 1), axis=0)# 中心化しているので平均=0
    #分散共分散行列
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
      cvm = np.zeros((self.dimension, self.dimension) , dtype=float)
      for row in range(self.dimension):
        for col in range(self.dimension):
          # cvm[row][col] = self.calc_variance(self.data[:,row],self.data[:,col])
          if row <= col:
              cvm[row][col] = self.calc_variance(self.data[:,row],self.data[:,col])
          else:
              cvm[row][col] = cvm[col][row]
      return cvm
    # 固有ベクトル（主成分）
    def get_components(self,cvm):
      w, eigenvector = np.linalg.eig(cvm) # 固有値w,  固有ベクトルeigenvector(主成分)
      eigenvector = eigenvector.T # 第n主成分に対応する固有ベクトルは第n列ベクトルなので転置
      return eigenvector

    def fit(self):
      cvm = self.get_Covariance_Matrix()
      self.components = self.get_components(cvm)



iris = load_iris()

print('--------ライブラリでPCA-------')
# 第n主成分に対応するベクトルはn行ベクトル
pca = PCA()
pca.fit(iris.data)
# print(f'分散共分散：\n{pca.get_covariance()}')
print(f'主成分ベクトル：\n{pca.components_}')

print('--------自作クラスででPCA-------')
my_pca = my_PCA(iris.data)
my_pca.fit()
print(f'主成分ベクトル：\n{my_pca.components}')