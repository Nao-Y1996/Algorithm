# -*- coding: utf-8 -*-
import numpy as np
import cv2
import random
import time
from sklearn import datasets
import matplotlib.pyplot as plt
import os



class GrowingNwuralGas():
    def __init__(self, feature_dim, max_node_num, max_age, adjacency_matrix_image_height=None):
        self.max_node_num = max_node_num
        self.feature_dim = feature_dim
        self.max_age = max_age

        # ノードの初期化
        self.nodes = []
        for i in range(2):
            feature = np.random.rand(self.feature_dim)
            self.nodes.append(feature.tolist())
        # 隣接行列の初期化
        self.adjacency_matrix = [[-1 for i in range(len(self.nodes))] for j in range(len(self.nodes))]
        # ノードの積算誤差の初期化
        self.errors = [0 for i in range(len(self.nodes))]
        self.eta1, self.eta2, self.mu, self.nu = None, None, None, None
        self.image = None
        self.file_path = os.path.dirname(__file__)
        
        # visualize_adjacency_matrix_init
        if adjacency_matrix_image_height is not None:
            height = adjacency_matrix_image_height
            width = height
            image = np.zeros((height+100, width+100, 3)) +255 # ←全ゼロデータに255を足してホワイトにする
            cv2.imwrite(os.path.join(self.file_path, f'blanck_{height}_{width}.jpg'), image)
            self.image = cv2.imread(os.path.join(self.file_path, f'blanck_{height}_{width}.jpg'))
            self.ages = list(range(self.max_age+1))
            self.ages_color = ((np.array(self.ages)/self.max_age)*255).tolist()
            self.step = int(height / self.max_node_num)
            # 外枠の表示
            cv2.rectangle(self.image, (0, 0), (height, width),(0, 0, 0), thickness=3)
            # マス目の表示
            for i in range(self.max_node_num):
                cv2.line(self.image, (0, self.step*i), (height, self.step*i), (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                cv2.line(self.image, (self.step*i, 0), (self.step*i, height), (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    def gng_init(self, data, eta1, eta2, mu, nu):
        self.data = data
        self.eta1, self.eta2, self.mu, self.nu = eta1, eta2, mu, nu
        self.adjacency_matrix[0][1] = 0
        self.adjacency_matrix[1][0] = 0

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.set_title('GNG', fontsize=14)

    def visualize_adjacency_matrix(self):

        for i in range(len(self.adjacency_matrix)):
            for j in range(len(self.adjacency_matrix)):
                if self.adjacency_matrix[i][j] == -1:
                    color = (255,255,255)
                else:
                    self.ages_color = ((np.array(self.ages)/self.max_age)*200).tolist()
                    try: # adjacency_matrixの要素のageに比例して明るい赤色にしていく
                        c = int(self.ages_color[self.ages.index(self.adjacency_matrix[i][j])])
                        color = (0,0,c+55) #BGR
                    # adjacency_matrixの要素がmax_egeを超えていて削除される直前の状態は青色にする
                    except ValueError:
                        color = (255,0,0)
                cv2.rectangle(self.image, (self.step*j+1, self.step*i+1), (self.step*(j+1)-1, self.step*(i+1)-1), (color), thickness=-1)
                cv2.imshow('im', self.image)
                cv2.waitKey(1)

    def find_winners_index(self, input_index):
        # STEP2
        _DistList = [0.0 for i in range(len(self.nodes))]
        dist_list = [0.0 for i in range(len(self.nodes))]
        diffs = np.array(self.nodes) - np.array(self.data[input_index])
        for i, diff in enumerate(diffs):
            _DistList[i] = np.sum(np.power(diff, 2)) ** 0.5  # ユークリッド距離
            dist_list[i] = _DistList[i]
        win1 = np.argmin(_DistList)
        _DistList.pop(win1)
        _win2 = np.argmin(_DistList)
        if win1 <= _win2:
            win2 = _win2+1
        else:
            win2 = _win2
        # STEP3 update errors
        self.errors[win1] += dist_list[win1]**2
        
        # self.plots(input_index=input_index, win1=win1, win2=win2)
        # self.visualize_adjacency_matrix()
        
        return win1, win2

    def update_conect(self, win1, win2, input_index):
        # STEP4
        #   update feature of winner1
        self.nodes[win1] = (self.nodes[win1] + self.eta1 * (np.array(self.data[input_index])-self.nodes[win1])).tolist()
        #   update feature of node which is connected to winner1
        win1_con_list = []  # list of indexes of node which is connected to winner1
        for i, node_feature in enumerate(self.nodes):
            if (self.adjacency_matrix[win1][i] >= 0):# and i != win1:
                self.nodes[i] = ( self.nodes[i] + self.eta2 * ( np.array(self.data[input_index])-self.nodes[i] ) ).tolist()
                win1_con_list.append(i)
        
        # self.plots(input_index=input_index, win1=win1, win2=win2)
        # self.visualize_adjacency_matrix()
        
        # STEP5
        self.adjacency_matrix[win1][win2] = 0
        self.adjacency_matrix[win2][win1] = 0
        
        # self.plots(input_index=input_index, win1=win1, win2=win2)
        # self.visualize_adjacency_matrix()
        
        # STEP6
        win1_con_list.append(win2)
        for i in win1_con_list:
            self.adjacency_matrix[win1][i] += 1
            self.adjacency_matrix[i][win1] += 1
        
        # self.plots(input_index=input_index, win1=win1, win2=win2)
        # self.visualize_adjacency_matrix()
        
        # STEP7
        #   remove edges that's age is over max_age
        delete_node_list = []
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if self.adjacency_matrix[i][j] > self.max_age:
                    self.adjacency_matrix[i][j] = -1
                    self.adjacency_matrix[j][i] = -1
        #   remove nodes
            if (self.adjacency_matrix[i] == [-1 for k in range(len(self.nodes))]):
                delete_node_list.append(i)
        count = 0
        for i in delete_node_list:
            # print('~~~~~~~~~~~~~~~~~~~~ノードを削除')
            i -= count
            self.nodes = np.delete(self.nodes, i, axis=0).tolist()
            self.adjacency_matrix = np.delete(self.adjacency_matrix, i, axis=0).tolist()
            self.adjacency_matrix = np.delete(self.adjacency_matrix, i, axis=1).tolist()
            self.errors = np.delete(self.errors, i).tolist()
            count += 1

        # self.plots(input_index=input_index, win1=win1, win2=win2)
        # self.visualize_adjacency_matrix()

    def add_node(self):
        # STEP8
        # print('-------------ノードを追加')
        u = np.argmax(self.errors)
        _max_error = 0
        f = 0
        for i in range(len(self.errors)):
            if (self.adjacency_matrix[i][u] >= 0) and i != u:
                if _max_error < self.errors[i]:
                    _max_error = self.errors[f]
                    f = i
        r_node_feature = ((np.array(self.nodes[u]) + np.array(self.nodes[f]))*0.5).tolist()
        self.nodes.append(r_node_feature)
        self.adjacency_matrix = (np.insert(self.adjacency_matrix, len(self.adjacency_matrix), [-1 for k in range(len(self.nodes)-1)], axis=1)).tolist()
        self.adjacency_matrix = (np.insert(self.adjacency_matrix, len(self.adjacency_matrix), [-1 for k in range(len(self.nodes))], axis=0)).tolist()
        # self.errors.append(0)
        # ノードu, f の接続を切る
        self.adjacency_matrix[u][f] = -1
        self.adjacency_matrix[f][u] = -1
        # ノードu,rとノードf,rを接続
        self.adjacency_matrix[u][len(self.nodes)-1] = 0
        self.adjacency_matrix[f][len(self.nodes)-1] = 0
        self.adjacency_matrix[len(self.nodes)-1][f] = 0
        self.adjacency_matrix[len(self.nodes)-1][u] = 0
        
        # ノードu, f の積算誤差を更新
        self.errors[u] = self.errors[u] * (1 - self.mu)# * self.errors[u]
        self.errors[f] = self.errors[f] * (1 - self.mu)# - self.mu * self.errors[f]
        # 追加した新たなノードの積算誤差を設定
        self.errors.append((self.errors[u] + self.errors[f]) * 0.5)

    def reduce_error(self):
        # STEP9
        self.errors = (np.array(self.errors) * (1.0 - self.nu)).tolist()

    def plots(self, input_index=None, win1=None, win2=None, is_saving=False):  # グラフ作成
        self.ax1.scatter(self.data[:, 0], self.data[:, 1], s=15, lw=0, c='gray')
        self.ax1.scatter(np.array(self.nodes)[:, 0], np.array(self.nodes)[:, 1], s=100, lw=0, c="blue")
        if input_index is not None:
            self.ax1.scatter(self.data[input_index][0], self.data[input_index][1],s=50, color="black")
        for i, feature in enumerate(self.nodes):
            for j in range(len(self.nodes)):
                if self.adjacency_matrix[i][j] >= 0:
                    self.ax1.plot([self.nodes[i][0], self.nodes[j][0]], [self.nodes[i][1], self.nodes[j][1]], color='black', linestyle='-', linewidth=2)
                    # plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2)
        if (win1 is not None) & (win2 is not None):
            self.ax1.scatter(self.nodes[win1][0], self.nodes[win1][1], s=100, lw=0, c="yellow")
            self.ax1.scatter(self.nodes[win2][0], self.nodes[win2][1], s=100, lw=0, c="green")
        plt.pause(0.01)
        if is_saving:
            plt.savefig(os.path.join(self.file_path, 'gng_result/final_result.png'))
        plt.cla()




if __name__ == "__main__":

    # dataset = datasets.make_moons(n_samples=10000, noise=0.1)[0]-9
    # dataset = datasets.make_circles(n_samples=500,shuffle=True,noise=0.1,random_state=None,factor = 0.7)[0]
    dataset = datasets.make_blobs(n_samples=10000,shuffle=True,centers=3,cluster_std=0.80,random_state=3)[0]

    gng = GrowingNwuralGas(feature_dim=2, max_node_num=100, max_age=20) #, adjacency_matrix_image_height=600)
    gng.gng_init(data=dataset, eta1=0.1, eta2=0.01, mu=0.5, nu=0.005)
    # gng.visualize_adjacency_matrix_init(image_height=1000)

    input_count = 1
    while len(gng.nodes) <= gng.max_node_num:
    # while input_count <= 10000:
        input_index = random.randrange(0, len(dataset))
        win1, win2 = gng.find_winners_index(input_index)
        gng.update_conect(win1, win2, input_index)
        if input_count%500 == 0 :
            gng.add_node()
            gng.plots()
        gng.reduce_error()
        input_count+=1
        print(f'\rinput_num : {input_count}',end='')
        # print(f'\rnumber of node：{len(gng.nodes)}',end='')
    gng.plots(is_saving=True)

