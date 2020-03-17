# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 00:45:32 2020

@author: Walter Aburime
"""
import numpy as np
import sigmoid

def predict(theta, X):
    h = sigmoid.sigmoid(X @ theta)
    p = np.zeros((h.shape[0],1))
    for k in range(h.shape[0]):
        if h[k] >= 0.5:
            p[k,0] = 1
        else:
            p[k,0] = 0
    
    return p