# -*- coding: utf-8 -*-
#!/usr/bin/env python2

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   #fig.add_subplot(1,1,1, projection='3d')に必要
from PIL import Image
import os




def create_gif(data, start_azim, end_azim, reduction_rate=1):
    num_data = np.shape(data)[0]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_zlim(0, 1)

    for i in range(num_data):
        ax.scatter(data[i][0],data[i][1],data[i][2], c='green')

    images = []
    fig_num = 0
    for i in range(start_azim, end_azim):
        ax.view_init(elev=30, azim=i)

        if i%reduction_rate==0:
            fig_num = int(i/reduction_rate)
            plt.savefig("gif/"+str(fig_num)+".png")
            img = Image.open("gif/"+str(fig_num)+".png")
            images.append(img)
            os.remove("gif/"+str(fig_num)+".png")
        else:
            continue
    images[0].save('gif/pillow_imagedraw.gif',
                   save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)

if __name__ == "__main__":
    data = np.random.randn(100,3)
    create_gif(data,0,90,3)