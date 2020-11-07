#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def triangle_fuzzy(a,c,b,F):#a<c<bとする
    if a<F and F<=c: #(a,0),(c,1)の式
        return 1/(c-a)*(F-a)
    if c<F and F<b:  #(c,1),(b,0)の式
        return 1/(c-b)*(F-b)
    else:
        return 0

def l_edge_fuzzy(c,b,F):#c<bとする
    if F<=c:
        return 1
    if c<F and F<b:  #(c,1),(b,0)の式
        return 1/(c-b)*(F-b)
    else:
        return 0

def r_edge_fuzzy(a,c,F):#c<bとする
    if a<F and F<=c: #(a,0),(c,1)の式
        return 1/(c-a)*(F-a)
    if c<F:
        return 1
    else:
        return 0



class Fuzzy_inference():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.next_ek = 0


    def fuzzy4position(self, x, dx):
        w1 = l_edge_fuzzy(-1.5,0.0,x) * l_edge_fuzzy(-0.7,0.0,dx)
        w2 = l_edge_fuzzy(-1.5,0.0,x) * triangle_fuzzy(-0.2,0.0,0.2,dx)
        w3 = l_edge_fuzzy(-1.5,0.0,x) * r_edge_fuzzy(0.0,0.7,x)

        w4 = triangle_fuzzy(-0.5,0.0,0.5,x) * l_edge_fuzzy(-0.7,0.0,dx)
        w5 = triangle_fuzzy(-0.5,0.0,0.5,x) * triangle_fuzzy(-0.2,0.0,0.2,dx)
        w6 = triangle_fuzzy(-0.5,0.0,0.5,x) * r_edge_fuzzy(0.0,0.7,dx)

        w7 = r_edge_fuzzy(0.0,1.5,x) * l_edge_fuzzy(-0.7,0.0,dx)
        w8 = r_edge_fuzzy(0.0,1.5,x) * triangle_fuzzy(-0.2,0.0,0.2,dx)
        w9 = r_edge_fuzzy(0.0,1.5,x) * r_edge_fuzzy(0.0,0.7,dx)

        w_vec = np.array([w1,w2,w3,w4,w5,w6,w7,w8,w9])

        then_vec = np.array([ -0.15, -0.1, -0.05, 0.0,0.0,0.0, 0.05, 0.1, 0.15])
        print(w_vec)
        return np.dot(w_vec, then_vec)/np.sum(w_vec)

    def fuzzy4angle(self, x,dx):

        w1 = l_edge_fuzzy(-3.0,-2.0,x) * l_edge_fuzzy(-0.7,0.0,dx)
        w2 = l_edge_fuzzy(-3.0,-2.0,x) * triangle_fuzzy(-0.2,0.0,0.2,dx)
        w3 = l_edge_fuzzy(-3.0,-2.0,x) * r_edge_fuzzy(-0.7,0.0,dx)

        w4 = r_edge_fuzzy(2.0,3.0,x) * l_edge_fuzzy(-0.7,0.0,dx)
        w5 = r_edge_fuzzy(2.0,3.0,x) * triangle_fuzzy(-0.2,0.0,0.2,dx)
        w6 = r_edge_fuzzy(2.0,3.0,x) * r_edge_fuzzy(0.0,0.7,dx)

        w_vec = np.array([w1,w2,w3,w4,w5,w6])
        print(w_vec)
        then_vec = np.array([-0.78, -0.52, -0.26, 0.26, 0.52, 0.78])
        return np.dot(w_vec, then_vec)/np.sum(w_vec)


class Fuzzy_inference_new():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.next_ek = 0


    def fuzzy4position(self, x, dx):
        #x = arg1/(arg1+arg2)
        #dx = arg2/(arg1+arg2)

        w1 = l_edge_fuzzy(-0.5,0.0,x) * l_edge_fuzzy(-0.2,0.0,dx)
        w2 = l_edge_fuzzy(-0.5,0.0,x) * triangle_fuzzy(-0.2,0.0,0.2,dx)
        w3 = l_edge_fuzzy(-0.5,0.0,x) * r_edge_fuzzy(0.0,0.2,x)
        w4 = triangle_fuzzy(-0.5,0.0,0.5,x) * l_edge_fuzzy(-0.2,0.0,dx)
        w5 = triangle_fuzzy(-0.5,0.0,0.5,x) * triangle_fuzzy(-0.2,0.0,0.2,dx)
        w6 = triangle_fuzzy(-0.5,0.0,0.5,x) * r_edge_fuzzy(0.0,0.2,dx)

        w7 = r_edge_fuzzy(0.0,0.5,x) * l_edge_fuzzy(-0.2,0.0,dx)
        w8 = r_edge_fuzzy(0.0,0.5,x) * triangle_fuzzy(-0.2,0.0,0.2,dx)
        w9 = r_edge_fuzzy(0.0,0.5,x) * r_edge_fuzzy(0.0,0.2,dx)

        w_vec = np.array([w1,w2,w3,w4,w5,w6,w7,w8,w9])

        then_vec = np.array([ -0.15, -0.1, -0.05, 0.0,0.0,0.0, 0.05, 0.1, 0.15])
        print(w_vec)
        return np.dot(w_vec, then_vec)/np.sum(w_vec)

    def fuzzy4angle(self, x,dx):

        w1 = l_edge_fuzzy(-4.0,-3.0,x) * l_edge_fuzzy(-0.7,0.0,dx)
        w2 = l_edge_fuzzy(-4.0,-3.0,x) * triangle_fuzzy(-0.2,0.0,-0.2,dx)
        w3 = l_edge_fuzzy(-4.0,-3.0,x) * r_edge_fuzzy(-0.7,0.0,dx)

        w4 = r_edge_fuzzy(3.0,4.0,x) * l_edge_fuzzy(-0.7,0.0,dx)
        w5 = r_edge_fuzzy(3.0,4.0,x) * triangle_fuzzy(-0.2,0.0,-0.2,dx)
        w6 = r_edge_fuzzy(3.0,4.0,x) * r_edge_fuzzy(0.0,0.7,dx)

        w_vec = np.array([w1,w2,w3,w4,w5,w6])
        #print(w_vec)
        then_vec = np.array([-0.78, -0.52, -0.26, 0.26, 0.52, 0.78])
        return np.dot(w_vec, then_vec)/np.sum(w_vec)



if __name__=="__main__":


    fuzzy = Fuzzy_inference_new()
    ans = fuzzy.fuzzy4position(0.6,0.15)
    print(ans)
    #plt.plot(x, ans, marker="o", color = "red", linestyle = "--")