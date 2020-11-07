#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ニューラルガスの実装
import numpy as np
import matplotlib.pyplot as plt
import itertools

# テストデータの作成
def test_data(center_x, center_y, s, data_size):
    x_list = np.random.normal(0, s, (data_size)) + center_x
    y_list = np.random.normal(0, s, (data_size)) + center_y
    data = []
    for x, y in zip(x_list, y_list):
        data.append([x, y])
    return data

standard_deviation = 1
data_sie = 100
data1 = test_data(3, 10, standard_deviation, data_sie)
data2 = test_data(12, 10, standard_deviation, data_sie)
data3 = test_data(7, 3, standard_deviation, data_sie)
input_data = np.concatenate([data1, data2, data3], 0)
# plt.scatter(input_data[0][0:100], input_data[1][0:100], c="pink")
# plt.scatter(input_data[0][100:200], input_data[1][100:200], c="green")
# plt.scatter(input_data[0][200:300], input_data[1][200:300], c="blue")
# plt.xlim(-5, 15)
# plt.ylim(-5, 15)
# plt.show()

def calc_d(vector): return np.linalg.norm(vector)


def get_ranking(array):
    """
    Arg:
      array
    Returns:
      list: a list of how small an array element is in the array
    """
    ranking_info = sorted(list(enumerate(sorted(list(enumerate(array)),
                                                key=lambda x: x[1]))),
                          key=lambda y: y[1][0])
    ranking_list = []
    for i in ranking_info:
        ranking_list.append(i[0])
    return ranking_list

# 参照ベクトルの初期化
w_num = 30
data_dimension = 2
a, b = 0, 15  # 各参照ベクトルの要素はa~bまでのランダムな整数
w = (np.random.rand(w_num * data_dimension)
     * (b-a) + a).reshape(-1, data_dimension)
# 結合関係の初期化
edges = np.zeros((w_num, w_num), int)
# 結合年齢の初期化
ages = np.zeros((w_num, w_num), int)
# 入力ベクトルと全ての参照ベクトルのユークリッド距離を計算
d_list = input_data[0] - w
Euclidean_d_list = list(map(calc_d, d_list))
# 入力ベクトルとの近さランキング
ranking = get_ranking(Euclidean_d_list)
# print(ranking)
# ランキングに基づいて参照ベクトルを更新
