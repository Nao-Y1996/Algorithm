#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
import numpy as np
import numpy.linalg as LA
import itertools
from statistics import variance # 自作コードの評価用


class PCA():
    def __init__(self,data):
      self.n = np.shape(data)[0] # データ数
      self.dimension = np.shape(data)[1] # 次元数
      # データの中心化
      self.data = data - self.average(data)
    #各次元の平均を計算
    def average(self,data):
      """
      Arg:
        list: 2 dimensional array
      return:
        np.array : list of Average per column
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
      np.cov(pca.data, rowvar=0, bias=0)と等しい
      np.covでは標本データではbias=1とするがすでに分散を求める際にn-1で割っているためbias=0と等しい
      """
      cvm = np.array([[0] * self.dimension for _ in range(self.dimension)], dtype=float)
      for row in range(self.dimension):
        for col in range(self.dimension):
          # cvm[row][col] = self.calc_variance(self.data[:,row],self.data[:,col])
          if row > col:
              cvm[row][col] = cvm[col][row]
          else:
              cvm[row][col] = self.calc_variance(self.data[:,row],self.data[:,col])
      return cvm


# iris = load_iris()
# pca = PCA(iris.data)
#共分散
# cvm = pca.get_Covariance_Matrix()
# print(cvm)

# w,v = LA.eig(cvm)
# print(w,v,sep='\n')