#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Self-Organization-Map
# 入力データを何周学習するか指定できるプログラム

import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import random


class som_clustering():
    def __init__(self, data_dimension, ml, data_num, loop):
        self.dimension = data_dimension # データ(ノード)の次元
        self.ml = ml # map liner (mapの一辺のノード数)
        self.data_num = data_num #入力データの数
        self.loop = loop # 入力データを何周学習させるか
        self.t_max = self.data_num * self.loop # 学習回数
    #勝者ノード(best match unit)の決定
    def bmu_index(self, Map, input_vector):
        input_vector_map = np.full((self.ml, self.ml, 3), input_vector)
        norm_vec =[]
        for i in range(self.ml):
            for j in range(self.ml):
                norm = np.linalg.norm(Map[i][j] - input_vector_map[i][j], ord=2)
                norm_vec.append(norm)
        bmu = np.unravel_index(np.argmin(norm_vec),(self.ml,self.ml))
        return bmu
    #近傍半径の定義
    def neighborhood_radius(self, t):
        halflife = float(self.t_max/4)
        initial  = float(self.ml/2)
        return initial * np.exp(-t/halflife)
        #return 1 + (initial-1)*((t_max-t)/t_max)
    #学習率の定義
    def learning_ratio(self, t):
        halflife = float(self.t_max/4)
        initial  = 0.9
        return initial * np.exp(-t/halflife)
        #return initial*(1-(t/t_max))
    #近傍関数の定義
    def neighborhood_function(self, t, d):
        # d is distance from BMU
        r = self.neighborhood_radius(t)
        alpha = self.learning_ratio(t)
        return alpha * np.exp(-d**2/(2*r**2))

#モデルの作成
som = som_clustering(data_dimension=3, ml=20, data_num=1000, loop=1)

#input_dataの作成
input_data_path = 'csv/input.csv'
if not os.path.exists(input_data_path):
    data = np.random.rand(som.data_num * som.dimension)
    with open(input_data_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(data)
data = np.loadtxt(input_data_path, delimiter=',')
input_data = data.reshape(som.data_num, som.dimension)
print(np.shape(input_data))

#初期Mapの作成
Map_path = 'csv/initial_Map.csv'
if not os.path.exists(Map_path):
    data = np.random.rand(som.ml * som.ml * som.dimension)
    with open(Map_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(data)
data = np.loadtxt(Map_path, delimiter=',')
Map = data.reshape(som.ml, som.ml, som.dimension)
print(np.shape(Map))

#初期マップの図
initial_Map = Map
plt.imshow(initial_Map, interpolation='none')
plt.savefig("fig/0.png")
plt.clf()
plt.close()

#近郷半径の変化図
t = np.linspace(0,som.t_max,100)
y = som.neighborhood_radius(t)
plt.figure(figsize=(6, 3))
plt.plot(t,y)
plt.title('neighborhood_radius')
plt.savefig("fig/neighborhood_radius.png")
# plt.show()
plt.clf()
plt.close()
#学習率の変化図
t = np.linspace(0,som.t_max,100)
y = som.learning_ratio(t)
plt.plot(t,y) 
plt.title('learning_ratio')
plt.savefig("fig/learning_ratio.png")
# plt.show()
plt.clf()
plt.close()


t = 0
for l in range(som.loop):
    print("学習{}周目".format(l))
    # 何周かする場合はinput_dataを並び替える
    if som.loop>1:
        shuffled_input_data_path = 'csv/data1_shuffled'+str(l)+'.csv'
        if not os.path.exists(shuffled_input_data_path):
            data = input_data.reshape(som.data_num * som.dimension)
            data = random.sample(data, len(data))# 並び替え　(ランダムに複数の要素を選択)
            with open(shuffled_input_data_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(data)
        data = np.loadtxt(shuffled_input_data_path, delimiter=',')
        input_data = data.reshape(som.data_num,som.dimension)

    # 学習
    for input_data in  input_data:
        t+=1
        bmu = som.bmu_index(Map, input_data)
        for x in range(som.ml):
            for y in range(som.ml):
                unit = np.array([x,y])# coordinate of unit
                d = np.linalg.norm(unit-bmu, ord=2)
                for dim in range(som.dimension): #TODO clear up using numpy function
                    Map[x,y,dim] += som.neighborhood_function(t,d)*(input_data[dim] - Map[x][y][dim])
        if t%100==0: #out put for t<100 or each 1000 iteration
            plt.imshow(Map, interpolation='none')
            plt.savefig("fig/" + str(l) + "周目-" + str(t) + ".png")
# plt.imshow(Map, interpolation='none')
plt.savefig("fig/final.png")
plt.clf()
plt.close()




