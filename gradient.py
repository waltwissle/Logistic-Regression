# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 13:16:45 2020

@author: Walter Aburime
"""

import numpy as np
import sigmoid

def gradient(theta, X, Y):
    
    m,n = X.shape
    theta = theta.reshape((n,1))
    Y = Y.reshape((m,1))
    #grad = np.zeros((theta.shape))
    h_theta = sigmoid.sigmoid(X @ theta) #the hypothesis h(theta) = 1/(1 + e**(z))
    grad = (1/m) * (h_theta - Y).T @ X
    return grad
    