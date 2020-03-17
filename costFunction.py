# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 22:47:19 2020

@author: Walter Aburime
"""
import numpy as np
import sigmoid


def costFunction(theta, X, Y):
    """
    %COSTFUNCTION Compute cost and gradient for logistic regression
       J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
       parameter for logistic regression and the gradient of the cost
       w.r.t. to the parameters.
    """
    m = len(Y)
    J  = 0
    
    h_theta = sigmoid.sigmoid(X @ theta) #the hypothesis h(theta) = 1/(1 + e**(z))
    a = np.log(h_theta)
    b = 1 - Y
    c = np.log(1 - h_theta)
    J = 1/m * (np.sum((-Y.T @ a) - (b.T @ c)))
    
    return J