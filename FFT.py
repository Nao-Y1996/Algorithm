#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 21:39:28 2020

@author: NaoYamada
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

N = 32
x = np.arange(N)
freq = 3 # 周期
y = np.sin(freq * x * (2*np.pi/N))

plt.figure(figsize=(8, 4))
plt.xlabel('n')
plt.ylabel('Signal')
plt.plot(x, y)

F = np.fft.fft(y) # 高速フーリエ変換(FFT)

F_abs = np.abs(F)
plt.plot(F_abs[:int(N/2)+1])# 周きを確認できるのはデータ数の半分まで
