#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
import numpy as np

a = np.array([1, 2, 3])
b = list(a)
b.append(4)
print(b)
b = np.array(b)
print(type(b))
